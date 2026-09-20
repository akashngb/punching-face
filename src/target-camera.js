import {
  FistPoseEstimator,
  PinholeCamera,
  handApparentSpan,
  closureOf,
} from './fist-pose.js';
import { PunchExtractor, DEFAULTS } from './punch-events.js';

// Target-perspective capture shell.
//
// This camera sits at the target and looks back at the puncher. It measures "where on the face"
// and "when did it arrive" IN ITS IMAGE PLANE — the direction cameras measure best — where the
// first-person camera has to estimate the same quantities along its own view axis, the direction
// any monocular method estimates worst. Nothing here is fused with the phone: this camera
// *defines* the target frame, so there is no extrinsic transform between two devices to calibrate,
// which matters because the phone is strapped to a moving human.
//
// The decisions live in src/punch-events.js (see docs/target-cv-pipeline.md). This file owns the
// plumbing only: camera + worker lifecycle, converting each worker result into observations (the
// rigid 6-DOF fist fit for the hand stream, pass-through for the motion-blob stream), and the
// stage-by-stage probe that says WHICH link of the chain broke when nothing fires.

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const KNUCKLE_SPAN_PRIOR = 0.084;

/**
 * Lateral offset of an image point, in metres, at the depth implied by apparent size.
 *
 * Worth noting: this is independent of the assumed field of view. Lateral offset is
 * (2u-1)*depth*k and depth from apparent size is proportional to 1/k, so k cancels. Impact
 * LOCATION therefore needs no camera calibration at all. Depth itself does not cancel, so approach
 * SPEED scales linearly with the assumed FOV -- that one is a single global gain on intensity.
 */
export function lateralOffset(point, depth, camera) {
  return {
    x: (point.x * 2 - 1) * depth * camera.kx,
    y: (1 - point.y * 2) * depth * camera.ky,
  };
}

export class TargetTracking {
  constructor(
    video,
    onStatus,
    {
      onImpact,
      fovDegrees = 60,
      targetWidth = 0.22,
      headWidth = 0.3,
      graceMs = 160,
      motionFloor = 0.01,
      headRadii = [0.15, 0.14, 0.09],
      mirrored = false,
      ...tuning
    } = {},
  ) {
    Object.assign(this, {
      video,
      onStatus,
      onImpact,
      fovDegrees,
      targetWidth,
      headWidth,
      motionFloor,
      headRadii,
      mirrored,
    });
    this.camera = new PinholeCamera({
      fovDegrees,
      viewAspect: 4 / 3,
      sourceAspect: 4 / 3,
    });
    this.extractor = new PunchExtractor({ ...DEFAULTS, graceMs, ...tuning });
    this.active = false;
    this.inFlight = 0;
    this.frameCallback = null;
    this.results = null;
    this.appliedTimestamp = 0;
    this.rate = 0;
    this.lastResultAt = null;
    // Estimators persist across dropouts and are keyed per hand label. A single shared one was a
    // real bug: the shape learner Kabsch-aligns to a seeded template, so alternating left and
    // right punches dragged one template between two mirror-image hands and it never converged.
    this.estimators = new Map();
    this.counts = { frames: 0, detected: 0 };
    this.lastDepth = null;
    this.debugSpan = 0;
    this.debugMotion = 0;
    this.lastHands = 0;
    // Stage-by-stage counters. Each one isolates a different way the chain can die, so a failure
    // says which link broke instead of collapsing to "nothing detected".
    this.probe = {
      results: 0,
      rawHands: 0,
      withWorld: 0,
      fitOk: 0,
      fitFail: 0,
      lastFit: '',
      detectError: '',
      crudeDepth: null,
      crudeClosing: 0,
      crudeTime: 0,
    };
  }
  // The extractor's knobs, surfaced as plain properties so the debug sliders can poke them.
  get minPeak() {
    return this.extractor.options.minPeak;
  }
  set minPeak(value) {
    if (Number.isFinite(value)) this.extractor.configure({ minPeak: value });
  }
  get startClosing() {
    return this.extractor.options.startClosing;
  }
  set startClosing(value) {
    if (Number.isFinite(value)) this.extractor.configure({ startClosing: value });
  }
  get minTravel() {
    return this.extractor.options.minTravel;
  }
  set minTravel(value) {
    if (Number.isFinite(value)) this.extractor.configure({ minTravel: value });
  }
  get strikeRange() {
    return this.extractor.options.strikeRange;
  }
  set strikeRange(value) {
    if (Number.isFinite(value) && value > 0.1)
      this.extractor.configure({ strikeRange: value });
  }
  get contactDepth() {
    return this.extractor.options.contactDepth;
  }
  setContactDepth(value) {
    if (Number.isFinite(value) && value > 0.02)
      this.extractor.configure({ contactDepth: value });
  }
  setTargetWidth(value) {
    if (Number.isFinite(value) && value > 0.05) this.targetWidth = value;
  }
  setFov(value) {
    if (Number.isFinite(value) && value > 20 && value < 140) {
      this.fovDegrees = value;
      this.camera.set({ fovDegrees: value });
    }
  }
  setHeadWidth(value) {
    if (Number.isFinite(value) && value > 0.05) this.headWidth = value;
  }
  // Whether the camera delivers horizontally mirrored (selfie-style) frames. This cannot be
  // auto-detected: a mirrored world is geometrically self-consistent, so every measurement —
  // fitted chirality, wrist trail, sweep direction, even MediaPipe's label, which is itself
  // computed from image geometry — flips COHERENTLY. Built-in webcams deliver raw (unmirrored)
  // frames, but virtual cameras and some drivers mirror; when they do, every hook reads as the
  // other hand and lands on the other cheek. The flag flips the interpretation at the one
  // boundary where observations enter the extractor.
  setMirrored(value) {
    if (this.mirrored === !!value) return;
    this.mirrored = !!value;
    this.extractor.reset(); // buffers must not blend two coordinate conventions
  }
  #label(raw) {
    return this.mirrored
      ? raw === 'Left'
        ? 'Right'
        : raw === 'Right'
          ? 'Left'
          : raw
      : raw;
  }
  setHeadRadii(radii) {
    if (
      Array.isArray(radii) &&
      radii.length === 3 &&
      radii.every((v) => Number.isFinite(v) && v > 0.01)
    )
      this.headRadii = [...radii];
  }
  get gain() {
    return this.headWidth / Math.max(this.targetWidth, 1e-3);
  }
  get slots() {
    return this.extractor.slots;
  }
  // Legacy name, kept because "how many identities exist right now" is a question the readout and
  // the tests keep asking under this name.
  get tracks() {
    return this.extractor.slots;
  }
  get stats() {
    return {
      ...this.extractor.stats,
      frames: this.counts.frames,
      detected: this.counts.detected,
    };
  }
  get debug() {
    const d = this.extractor.debug;
    return {
      hands: this.lastHands,
      span: this.debugSpan,
      motion: this.debugMotion,
      depth: this.lastDepth,
      range: d.r,
      closing: d.closing ?? 0,
      phase: d.phase ?? 'idle',
      peak: d.peak ?? 0,
      travel: d.travel ?? 0,
      minDepth: d.minRange ?? null,
      key: d.slot ?? '—',
    };
  }
  async cameras() {
    return (await navigator.mediaDevices.enumerateDevices()).filter(
      (device) => device.kind === 'videoinput',
    );
  }
  async start(deviceId = '') {
    this.onStatus('Starting target camera…');
    try {
      // Deliberately smaller and faster than the first-person capture. An incoming fist fills a
      // large part of this frame, so resolution buys little, while frame rate buys a lot: impact
      // speed error was bounded by inference rate, not by the estimator.
      const video = {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 60, min: 30 },
      };
      if (deviceId) video.deviceId = { exact: deviceId };
      else video.facingMode = { ideal: 'user' };
      this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.worker = new Worker('/target-worker.js');
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(
          () => reject(new Error('Target CV models timed out while loading.')),
          25000,
        );
        this.worker.onmessage = ({ data }) => {
          if (data.type === 'ready') {
            clearTimeout(timeout);
            resolve();
          } else if (data.type === 'error') {
            clearTimeout(timeout);
            reject(new Error(data.message));
          }
        };
        this.worker.onerror = (event) => {
          clearTimeout(timeout);
          reject(new Error(event.message));
        };
        this.worker.postMessage({ type: 'init', origin: location.origin });
      });
      this.worker.onmessage = ({ data }) => {
        this.inFlight = Math.max(0, this.inFlight - 1);
        if (data.type === 'result') {
          if (!this.results || data.timestamp > this.results.timestamp)
            this.results = data;
          const gap = data.timestamp - (this.lastResultAt ?? data.timestamp);
          if (gap > 0)
            this.rate = this.rate
              ? this.rate + (1000 / gap - this.rate) * 0.1
              : 1000 / gap;
          this.lastResultAt = data.timestamp;
          this.latency = performance.now() - data.timestamp;
          this.scheduleFrame();
        } else if (data.type === 'error') {
          this.onStatus(data.message);
          this.stop();
        }
      };
      this.active = true;
      this.onStatus('Target camera live.');
      this.scheduleFrame();
    } catch (error) {
      this.stop();
      throw error;
    }
  }
  stop() {
    this.active = false;
    if (this.frameCallback !== null && this.video.cancelVideoFrameCallback)
      this.video.cancelVideoFrameCallback(this.frameCallback);
    this.worker?.terminate();
    this.worker = null;
    this.inFlight = 0;
    this.results = null;
    this.appliedTimestamp = 0;
    this.frameCallback = null;
    this.extractor.reset();
    this.estimators.clear();
    this.rate = 0;
    this.lastResultAt = null;
    this.lastDepth = null;
    this.debugSpan = 0;
    this.lastHands = 0;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }
  scheduleFrame() {
    if (
      !this.active ||
      !this.worker ||
      this.inFlight >= 2 ||
      this.frameCallback !== null
    )
      return;
    const post = async () => {
      this.frameCallback = null;
      if (!this.active || !this.worker || this.inFlight >= 2) return;
      if (this.video.readyState < 2) {
        this.scheduleFrame();
        return;
      }
      this.inFlight++;
      try {
        const bitmap = await createImageBitmap(this.video);
        // Stamped after the await: capture completion order is what the worker sees, and stamping
        // before it let two in-flight frames arrive with inverted timestamps.
        this.worker.postMessage(
          { type: 'frame', bitmap, timestamp: performance.now() },
          [bitmap],
        );
      } catch {
        this.inFlight--;
        this.scheduleFrame();
      }
    };
    this.frameCallback = this.video.requestVideoFrameCallback
      ? this.video.requestVideoFrameCallback(post)
      : setTimeout(post, 4);
  }
  estimatorFor(key) {
    let estimator = this.estimators.get(key);
    // Never rebuilt on a dropout: the hand leaves view between punches, and throwing the learned
    // hand size away each time meant it never finished converging.
    if (!estimator) {
      estimator = new FistPoseEstimator({ camera: this.camera });
      this.estimators.set(key, estimator);
    }
    return estimator;
  }
  #solveOptions() {
    return { gain: this.gain, headRadii: this.headRadii };
  }
  tick(now) {
    if (!this.active) return null;
    this.scheduleFrame();
    const results = this.results;
    // No new frame: the extractor still runs, because censoring (a fist that blurred out at full
    // extension) is a property of time passing, not of frames arriving.
    if (!results || results.timestamp === this.appliedTimestamp)
      return this.extractor.tick(now, this.#solveOptions());
    this.appliedTimestamp = results.timestamp;
    this.counts.frames++;
    const sourceAspect =
      this.video.videoWidth && this.video.videoHeight
        ? this.video.videoWidth / this.video.videoHeight
        : 4 / 3;
    // No cover crop: this camera is a measuring instrument, read in its own frame, not a backdrop.
    this.camera.set({
      fovDegrees: this.fovDegrees,
      viewAspect: sourceAspect,
      sourceAspect,
    });
    this.debugMotion = results.motion?.energy ?? 0;
    const raw = results.landmarks ?? [];
    this.lastHands = raw.length;
    this.probe.results++;
    this.probe.rawHands = raw.length;
    this.probe.detectError = results.detectError ?? '';

    // Everything entering the extractor is expressed in the unmirrored convention; `sign`/`mu`
    // undo a mirrored feed in one place.
    const sign = this.mirrored ? -1 : 1,
      mu = (u) => (this.mirrored ? 1 - u : u);
    // The motion stream goes to the core first, so this very tick can already bridge with it.
    // The blob's block-matched flow (du/dv, image units per second) rides along: the core uses it
    // to steer a slot's predicted position through landmark-dead blur, which is what lets the
    // reacquired landmarks rejoin their own identity instead of forking a duplicate.
    for (const blob of results.motion?.blobs ?? []) {
      this.extractor.pushBlob({
        t: results.timestamp,
        u: mu(blob.u),
        v: blob.v,
        mass: blob.mass,
        spread: blob.spread,
        expand: blob.expand,
        du: Number.isFinite(blob.du) ? sign * blob.du : undefined,
        dv: blob.dv,
        kx: this.camera.kx,
        ky: this.camera.ky,
      });
    }

    // The hand's long axis (wrist -> knuckle centroid) measured IN THE IMAGE, in units of the
    // fist's own apparent width. This is the jab/uppercut discriminator, and it is deliberately
    // computed here from raw landmarks rather than taken from the 3D pose: a straight punch's
    // hand axis points along the view axis, so in 3D it lives almost entirely in the DEPTH
    // component — the one MediaPipe measures worst. Depth compression then shrinks that
    // component and normalisation inflates the small honest vertical, so a jab's fitted axis
    // tilts "up" and reads as an uppercut. In the image there is no such failure: a fist aimed
    // at the camera has its axis foreshortened to nearly nothing (short vector), while an
    // uppercut's points clearly up the frame (long vector). Both are pure image-plane geometry.
    // Scaled into head-frame image units (+du = head +x, +dv = up) and made aspect-correct so
    // the ratio means the same thing horizontally and vertically.
    const axisOf = (landmarks) => {
      if (landmarks?.length !== 21) return null;
      const mcp = [5, 9, 13, 17];
      const mx = mcp.reduce((s, i) => s + landmarks[i].x, 0) / 4,
        my = mcp.reduce((s, i) => s + landmarks[i].y, 0) / 4;
      const dx = (mx - landmarks[0].x) * sourceAspect,
        dy = my - landmarks[0].y;
      const span = Math.hypot(
        (landmarks[5].x - landmarks[17].x) * sourceAspect,
        landmarks[5].y - landmarks[17].y,
      );
      const len = Math.hypot(dx, dy),
        scale = Math.max(span, len);
      if (!(scale > 1e-4)) return null;
      // Normalised by the hand's LARGER apparent dimension, not by the knuckle span alone: an
      // edge-on fist (guard — back of the fist to the side) collapses the span, and dividing by
      // it inflated a resting guard fist into an emphatic "axis up". `facing` says which face
      // shows: knuckle-row span relative to hand scale — ~1.0 whenever the back (or front) of
      // the fist is to the camera, ~.1 for a guard fist seen edge-on.
      return { du: (-sign * dx) / scale, dv: -dy / scale, facing: span / scale };
    };
    const detections = raw
      .map((landmarks, index) => ({
        landmarks,
        index,
        world: results.worldLandmarks?.[index],
        span: handApparentSpan(landmarks),
        axis: axisOf(landmarks),
        knuckles:
          landmarks?.length === 21
            ? Math.hypot(
                landmarks[5].x - landmarks[17].x,
                landmarks[5].y - landmarks[17].y,
              )
            : 0,
        raw: results.handedness?.[index]?.[0]?.categoryName ?? 'hand',
        confidence: results.handedness?.[index]?.[0]?.score ?? 1,
        centre:
          landmarks?.length === 21
            ? {
                x: mu((landmarks[5].x + landmarks[17].x) / 2),
                y: (landmarks[5].y + landmarks[17].y) / 2,
              }
            : { x: 0.5, y: 0.5 },
      }))
      .filter((detection) => detection.span);
    this.probe.withWorld = detections.filter(
      (detection) => detection.world?.length === 21,
    ).length;

    // Crude apparent-size depth, computed straight from the landmarks with no rigid fit involved.
    // It carries exactly the foreshortening bias this pipeline exists to remove, so it never feeds
    // detection -- but it guarantees the readout shows a live number even when the fit is failing,
    // which is what makes a broken fit visible instead of silent.
    const nearest = detections.reduce(
      (best, d) => (!best || d.knuckles > best.knuckles ? d : best),
      null,
    );
    if (nearest && nearest.knuckles > 1e-4) {
      const span =
        this.estimators.get(this.#label(nearest.raw))?.shape?.span ||
        KNUCKLE_SPAN_PRIOR;
      const crude = span / (2 * nearest.knuckles * this.camera.kx);
      const elapsed = (results.timestamp - this.probe.crudeTime) / 1000;
      this.probe.crudeClosing =
        this.probe.crudeDepth !== null && elapsed > 1e-3 && elapsed < 0.5
          ? (this.probe.crudeDepth - crude) / elapsed
          : 0;
      this.probe.crudeDepth = crude;
      this.probe.crudeTime = results.timestamp;
    } else {
      this.probe.crudeDepth = null;
      this.probe.crudeClosing = 0;
    }

    const usable = detections.filter((detection) => detection.world?.length === 21);
    if (usable.length) this.counts.detected++;
    // Don't leave the last good reading on screen when detection has dropped — a stale number
    // reads as a working pipeline, which is exactly the wrong thing while debugging one.
    else {
      this.lastDepth = null;
      this.debugSpan = 0;
    }

    const batch = usable.map((d) => ({
      u: d.centre.x,
      v: d.centre.y,
      span: d.span,
      det: d,
    }));
    const assignment = this.extractor.assign(batch, results.timestamp);
    for (const item of batch) {
      const slot = assignment.get(item),
        d = item.det;
      const label = this.#label(d.raw);
      // The estimator is keyed on the slot's majority handedness label so a single flickering
      // frame cannot swap the learned hand shape between two mirror-image hands.
      const estimator = this.estimatorFor(slot.label !== 'hand' ? slot.label : label);
      // The rigid fit consumes the RAW frame; its outputs are converted to the unmirrored
      // convention on the way into the extractor, alongside the already-converted image u.
      const solved = estimator.estimate(d.landmarks, d.world);
      if (solved) {
        this.probe.fitOk++;
        this.probe.lastFit = '';
        this.lastDepth = -solved.centre[2];
        this.debugSpan = d.span;
        this.extractor.push(slot, {
          t: results.timestamp,
          u: d.centre.x,
          v: d.centre.y,
          // Render space (-z in front) -> camera frame (z = distance in front).
          p: [sign * solved.centre[0], solved.centre[1], -solved.centre[2]],
          closure: solved.closure,
          kn: solved.knuckleNormal
            ? [
                sign * solved.knuckleNormal[0],
                solved.knuckleNormal[1],
                -solved.knuckleNormal[2],
              ]
            : null,
          label,
          score: d.confidence,
          quality: 'rigid',
          axis: d.axis,
          // Metric lateral wrist offset from the solved pose — the forearm trails toward its own
          // shoulder, and metric space is free of the perspective pull toward image centre that
          // makes the raw landmark offset read as the wrong hand for laterally-offset fists.
          wristDx: sign * (solved.points[0][0] - solved.centre[0]),
          chirality: sign * solved.chirality,
        });
      } else {
        this.probe.fitFail++;
        this.probe.lastFit =
          'solveTranslation returned null (hand too close, or degenerate)';
        // A degraded sample keeps the trajectory alive through what the rigid fit could not
        // solve — but ONLY while a recent rigid fit can lend it depth. Its DEPTH is inertial,
        // carried from that anchor, never derived from the apparent span: apparent size
        // conflates rotation with distance by construction (a stationary fist rotating as the
        // elbow lifts halves its knuckle span, which span-depth read as a tens-of-centimetres
        // phantom approach; blur widening it the other way fabricated superhuman speeds).
        //
        // With no anchor there is no honest depth to be had, so nothing enters the trajectory:
        // the hand is merely OBSERVED, which keeps its identity alive (association, liveness,
        // rest lineage) without inventing a position. A punch lasts ~200 ms, so "no rigid fit
        // for 400 ms" never happens mid-punch — it happens to a fist parked at the lens, where
        // apparent-size depth manufactured an endless stream of phantom jabs.
        const anchor =
          slot.samples.findLast?.(
            (s) => s.quality === 'rigid' && results.timestamp - s.t < 400,
          ) ??
          [...slot.samples]
            .reverse()
            .find((s) => s.quality === 'rigid' && results.timestamp - s.t < 400);
        if (anchor) {
          const depth = anchor.p[2];
          this.extractor.push(slot, {
            t: results.timestamp,
            u: d.centre.x,
            v: d.centre.y,
            p: [
              (d.centre.x * 2 - 1) * depth * this.camera.kx,
              (1 - d.centre.y * 2) * depth * this.camera.ky,
              depth,
            ],
            closure: closureOf(d.world),
            kn: null,
            label,
            score: d.confidence,
            quality: 'span',
            axis: d.axis,
          });
        } else
          this.extractor.observe(slot, {
            t: results.timestamp,
            u: d.centre.x,
            v: d.centre.y,
          });
      }
    }
    return this.extractor.tick(now, this.#solveOptions());
  }
}
