// Punch detector. Reads MediaPipe hand landmarks (21 points in normalized [0,1] frame coords)
// and classifies rapid arm extensions as LEFT hook, RIGHT hook, UPPERCUT, or JAB.
//
// A raw "palm got bigger in frame" (dW/dt > threshold) fires on every casual reach or wave —
// slow approaches accumulate enough growth to trip a naive threshold. We layer real
// discriminators on top so gestures pass through and only strikes land:
//
//   1. FRACTIONAL APPROACH   (dW/dt)/W — how many times the palm doubles per second.
//                            Depth-invariant. A hobbyist jab clocks ~8–15/s peak; a wave sits
//                            below 2/s. Depth-blind absolute growth (the old dW/dt) can't
//                            tell those apart because far-hand-fast and close-hand-slow both
//                            produce similar dW/dt.
//   2. SUSTAINED APPROACH    growth was positive across the whole recent window (not one
//                            spike from tracker jitter). Punches have a monotonic extension
//                            phase; waves oscillate.
//   3. FIST                  fistScore(lm) — real punches land with the fingers curled. Open
//                            palms mean "wave", "reach", "point"; not a strike.
//   4. PEAK VELOCITY         2D wrist speed in image coords. A punch's wrist covers real image
//                            distance fast; a slow reach-in doesn't.
//
// Side/type at trigger time comes from where the hand is and how it's moving:
//   vy < -minVerticalUp  → UPPERCUT
//   x > 0.5 + sideSlack  → LEFT hook (video is mirrored via CSS, so raw-right = user-left)
//   x < 0.5 - sideSlack  → RIGHT hook
//   else                 → JAB
//
// References: MediaPipe HandLandmarker documentation; typical hobbyist punch peak wrist speed
// runs 5–8 m/s (arm ~65 cm, straightens in ~80–120 ms), which in normalized image coords
// depends on distance to camera but drives dW/dt/W into the 8–15 /s range at close range.

import { fistScore } from './physics.js';

export const DEFAULT_TUNING = {
  minWidth: 0.05, // palm width in normalized image units (~1 m from a wide-FOV webcam)
  minFractionalGrowth: 2.6, // (dW/dt)/W — depth-invariant approach speed threshold, /s
  minFistScore: 0.35, // at least ~1.5/4 fingertips curled — enough to reject open-palm waves
  minPeakSpeed: 0.65, // 2D image-space wrist speed at trigger, /s (sqrt(dx²+dy²))
  minSustainedFraction: 0.55, // window-averaged fractional growth must clear this fraction of the trigger threshold
  minVerticalUp: 0.9, // dY/dt (negative = up on screen) for uppercut priority
  cooldownMs: 170, // per-type debounce
  historyMs: 100, // sliding window for derivative estimates
  sideSlack: 0.08, // buffer around center before we flip between hook/jab
};

export class SlapDetector {
  constructor(tuning = {}) {
    this.tuning = { ...DEFAULT_TUNING, ...tuning };
    this.reset();
  }

  reset() {
    this.history = [];
    // One global cooldown — a single physical swing sweeps across sides in one motion
    // and would otherwise trigger multiple hits (right → jab → left).
    this.lastTrigger = -Infinity;
    this.state = this.#idle('waiting for hand');
    this.lastEvent = null;
  }

  #idle(reason) {
    return {
      handDetected: false,
      palmWidth: 0,
      growth: 0,
      vy: 0,
      vx: 0,
      centerX: 0.5,
      centerY: 0.5,
      reason,
      triggered: null,
      landmarks: null,
    };
  }

  observe(landmarks, timestampMs) {
    const cutoff = timestampMs - this.tuning.historyMs * 4;
    while (this.history.length && this.history[0].t < cutoff) this.history.shift();
    if (!landmarks || !landmarks.length) {
      this.state = this.#idle('no hand visible');
      return null;
    }
    // Closest hand wins — biggest palm width.
    let best = null;
    for (const lm of landmarks) {
      const w = Math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y);
      if (!best || w > best.w)
        best = {
          w,
          x: (lm[0].x + lm[9].x) / 2,
          y: (lm[0].y + lm[9].y) / 2,
          points: lm,
        };
    }
    this.history.push({ t: timestampMs, x: best.x, y: best.y, w: best.w });
    // Derivative vs sample ~historyMs ago (pick the newest sample no newer than target).
    const target = timestampMs - this.tuning.historyMs;
    let prior = null;
    for (const s of this.history) {
      if (s.t <= target) prior = s;
      else break;
    }
    let growth = 0,
      vy = 0,
      vx = 0,
      fractional = 0,
      peakSpeed = 0;
    if (prior && timestampMs > prior.t) {
      const dt = (timestampMs - prior.t) / 1000;
      growth = (best.w - prior.w) / dt;
      vy = (best.y - prior.y) / dt;
      vx = (best.x - prior.x) / dt;
      const wMean = 0.5 * (best.w + prior.w);
      fractional = growth / Math.max(wMean, 1e-3);
      peakSpeed = Math.hypot(vx, vy);
    }
    // Sustained approach over the most recent ~120 ms — long enough to filter single-frame
    // MediaPipe jitter, short enough that punch onset clears the check within one detector window
    // instead of waiting out the whole ~400 ms history buffer. The old version averaged over all
    // history, which delayed triggers by 100-200 ms because most of the window was pre-punch.
    let sustained = 0;
    const sustainedStart = timestampMs - 120;
    let recentOldest = null;
    for (const s of this.history) {
      if (s.t >= sustainedStart) {
        recentOldest = s;
        break;
      }
    }
    if (recentOldest && recentOldest !== this.history[this.history.length - 1]) {
      const newest = this.history[this.history.length - 1];
      const spanDt = (newest.t - recentOldest.t) / 1000;
      if (spanDt > 0.02) {
        const meanW = 0.5 * (recentOldest.w + newest.w);
        sustained = (newest.w - recentOldest.w) / spanDt / Math.max(meanW, 1e-3);
      }
    }
    const fist = fistScore(best.points);
    let reason = 'ready',
      triggered = null;
    if (best.w < this.tuning.minWidth) {
      reason = `palm far (w=${best.w.toFixed(2)} < ${this.tuning.minWidth})`;
    } else if (fractional < this.tuning.minFractionalGrowth) {
      reason = `slow approach (${fractional.toFixed(1)}/s < ${this.tuning.minFractionalGrowth}/s)`;
    } else if (
      sustained <
      this.tuning.minFractionalGrowth * this.tuning.minSustainedFraction
    ) {
      reason = `unsustained (${sustained.toFixed(1)}/s below ${(this.tuning.minFractionalGrowth * this.tuning.minSustainedFraction).toFixed(1)}/s)`;
    } else if (fist < this.tuning.minFistScore) {
      reason = `open hand (fist=${fist.toFixed(2)} < ${this.tuning.minFistScore}) — wave, not punch`;
    } else if (peakSpeed < this.tuning.minPeakSpeed) {
      reason = `slow wrist (${peakSpeed.toFixed(2)}/s < ${this.tuning.minPeakSpeed}/s)`;
    } else {
      let side;
      // Uppercut needs (a) fast upward vy, (b) vertical dominates horizontal drift,
      // (c) hand is near-center in x. Otherwise a hook with slight rise gets stolen.
      const isUppercut =
        vy < -this.tuning.minVerticalUp &&
        Math.abs(vy) > Math.abs(vx) * 1.4 &&
        Math.abs(best.x - 0.5) < 0.28;
      // Camera preview is mirrored via CSS (scaleX(-1)), so the user sees their right hand on
      // the right of the screen. MediaPipe still reports unmirrored coords, so we flip the side
      // classification here: raw x > 0.5 is MediaPipe's "right of frame" = user's LEFT hand.
      if (isUppercut) side = 'up';
      else if (best.x > 0.5 + this.tuning.sideSlack) side = 'left';
      else if (best.x < 0.5 - this.tuning.sideSlack) side = 'right';
      else side = 'jab';
      if (timestampMs - this.lastTrigger < this.tuning.cooldownMs) {
        reason = `cooldown ${Math.round(this.tuning.cooldownMs - (timestampMs - this.lastTrigger))}ms`;
      } else {
        this.lastTrigger = timestampMs;
        const type = side === 'up' ? 'uppercut' : side === 'jab' ? 'jab' : 'hook';
        triggered = {
          type,
          side,
          growth,
          vy,
          vx,
          palmWidth: best.w,
          centerX: best.x,
          centerY: best.y,
          fractional,
          fist,
          peakSpeed,
          timestampMs,
        };
        this.lastEvent = triggered;
        reason = `${type.toUpperCase()} · ${side.toUpperCase()}`;
      }
    }
    this.state = {
      handDetected: true,
      palmWidth: best.w,
      growth,
      vy,
      vx,
      fractional,
      fist,
      peakSpeed,
      centerX: best.x,
      centerY: best.y,
      reason,
      triggered,
      landmarks: best.points,
    };
    return triggered;
  }
}
