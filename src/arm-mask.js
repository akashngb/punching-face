// Loaded two ways: bundled for tests, and imported by URL from public/pov-worker.js, which has to
// stay a classic worker so MediaPipe's WASM loader can use importScripts. Vite inlines this file
// for the `?url` import, so it must stay dependency-free.
export const LINKS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const FALLBACK_ALPHA = 165;

export function handScale(landmarks) {
  if (landmarks?.length !== 21) return 0;
  return Math.max(
    distance(landmarks[5], landmarks[17]),
    distance(landmarks[0], landmarks[9]) * 1.25,
  );
}

function segmentDistance(point, a, b) {
  const dx = b.x - a.x,
    dy = b.y - a.y,
    t = clamp(
      ((point.x - a.x) * dx + (point.y - a.y) * dy) / Math.max(dx * dx + dy * dy, 1e-9),
      0,
      1,
    );
  return Math.hypot(point.x - a.x - dx * t, point.y - a.y - dy * t);
}

function armGeometry(landmarks) {
  const scale = handScale(landmarks);
  if (!scale) return null;
  const wrist = landmarks[0],
    palm = [5, 9, 13, 17].reduce(
      (sum, index) => ({
        x: sum.x + landmarks[index].x / 4,
        y: sum.y + landmarks[index].y / 4,
      }),
      { x: 0, y: 0 },
    );
  let dx = wrist.x - palm.x,
    dy = wrist.y - palm.y,
    length = Math.hypot(dx, dy);
  if (length < 0.01) {
    dx = 0;
    dy = 1;
    length = 1;
  }
  dx /= length;
  dy /= length;
  const candidates = [];
  if (dx > 0) candidates.push((1 - wrist.x) / dx);
  else if (dx < 0) candidates.push(-wrist.x / dx);
  if (dy > 0) candidates.push((1 - wrist.y) / dy);
  else if (dy < 0) candidates.push(-wrist.y / dy);
  const valid = candidates.filter((value) => Number.isFinite(value) && value > 0),
    travel = valid.length ? Math.min(...valid) : 1;
  return {
    landmarks,
    palm,
    wrist,
    end: { x: wrist.x + dx * travel, y: wrist.y + dy * travel },
    scale,
  };
}

// Anti-loop anchor for the below-chin camera, which stares straight at the laptop screen and so
// sees the app's own rendered arms again. The physical truth it leans on: the guard mandates that
// every real arm enters the frame through the bottom-corner regions, while an on-screen ghost
// lives in a floating rectangle mid-frame. `anchored` is the person mass flood-reachable from the
// corner seed strips this frame. `connected` additionally grows from pixels holding temporal
// credit, so a momentary segmentation gap in the arm's neck of pixels does not decapitate it —
// the old any-border grounding's fatal flaw. Credit refills only on true corner connection and
// drains otherwise, so a ghost patch that got bridged while a fist passed in front of the screen
// (transitively corner-connected for those frames) dies within ~a second of separating.
export class ArmAnchor {
  constructor({ maxCredit = 26, bottomSpan = 0.38, sideStart = 0.55 } = {}) {
    this.maxCredit = maxCredit;
    this.bottomSpan = bottomSpan;
    this.sideStart = sideStart;
    this.credit = null;
  }
  update(person, width, height) {
    if (!this.credit || this.credit.length !== person.length)
      this.credit = new Uint8Array(person.length);
    const credit = this.credit,
      anchored = new Uint8Array(person.length),
      connected = new Uint8Array(person.length),
      stack = [];
    const seed = (index, map) => {
      if (person[index] && !map[index]) {
        map[index] = 1;
        stack.push(index);
      }
    };
    const fill = (map) => {
      while (stack.length) {
        const index = stack.pop(),
          x = index % width,
          y = (index - x) / width;
        if (x > 0) seed(index - 1, map);
        if (x < width - 1) seed(index + 1, map);
        if (y > 0) seed(index - width, map);
        if (y < height - 1) seed(index + width, map);
      }
    };
    for (let x = 0; x < width; x++)
      if (x <= width * this.bottomSpan || x >= width * (1 - this.bottomSpan))
        seed((height - 1) * width + x, anchored);
    for (let y = Math.floor(height * this.sideStart); y < height; y++) {
      seed(y * width, anchored);
      seed(y * width + width - 1, anchored);
    }
    fill(anchored);
    for (let i = 0; i < person.length; i++) {
      if (anchored[i]) connected[i] = 1;
      else if (credit[i] && person[i]) {
        connected[i] = 1;
        stack.push(i);
      }
    }
    fill(connected);
    for (let i = 0; i < credit.length; i++)
      credit[i] = anchored[i] ? this.maxCredit : credit[i] ? credit[i] - 1 : 0;
    return { anchored, connected };
  }
}

// The multiclass model labels body skin and clothes but does not distinguish one arm from the
// torso. Hand landmarks seed a hand skeleton and a tapered wrist-to-frame corridor, so only the
// first-person arms attached to detected hands win. The hand region accepts any person pixel
// (shadowed knuckles flicker between the segmenter's classes), and a tighter landmark-only core
// keeps partial alpha even when segmentation misses the fist entirely — landmarks are the
// trustworthy signal, so the fist may dim but never vanishes.
export function armCutout(labels, width, height, hands, support, anchor) {
  const alpha = new Uint8ClampedArray(width * height),
    person = new Uint8Array(alpha.length);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const label = labels[y * width + x];
      if (label !== 2 && label !== 4) continue;
      for (let oy = -1; oy <= 1; oy++)
        for (let ox = -1; ox <= 1; ox++) {
          const px = x + ox,
            py = y + oy;
          if (px < 0 || px >= width || py < 0 || py >= height) continue;
          person[py * width + px] = 1;
        }
    }
  const region = anchor ? anchor.update(person, width, height) : null;
  const geometry = hands.map(armGeometry);
  // A hand whose landmarks sit nowhere near corner-connected person mass is the screen's copy of
  // an arm, not an arm: it contributes no shapes, no fallback core, and the page must not track it.
  const anchoredHands = geometry.map((arm) => {
    if (!arm) return false;
    if (!region) return true;
    for (const index of [0, 5, 9, 13, 17]) {
      const point = arm.landmarks[index];
      const px = Math.min(width - 1, Math.max(0, Math.round(point.x * width - 0.5))),
        py = Math.min(height - 1, Math.max(0, Math.round(point.y * height - 0.5)));
      for (let oy = -1; oy <= 1; oy++)
        for (let ox = -1; ox <= 1; ox++) {
          const qx = px + ox,
            qy = py + oy;
          if (qx < 0 || qx >= width || qy < 0 || qy >= height) continue;
          if (region.connected[qy * width + qx]) return true;
        }
    }
    return false;
  });
  for (let armIndex = 0; armIndex < geometry.length; armIndex++) {
    const arm = geometry[armIndex];
    if (!arm || !anchoredHands[armIndex]) continue;
    // The corridor's widest capture radius is .97·scale and no hand test reaches past .72·scale,
    // so a scale-padded bounding box over every seed point contains the whole arm shape.
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const point of [arm.wrist, arm.end, arm.palm, ...arm.landmarks]) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }
    const x0 = Math.max(0, Math.floor((minX - arm.scale) * width)),
      x1 = Math.min(width - 1, Math.ceil((maxX + arm.scale) * width));
    const y0 = Math.max(0, Math.floor((minY - arm.scale) * height)),
      y1 = Math.min(height - 1, Math.ceil((maxY + arm.scale) * height));
    const dx = arm.end.x - arm.wrist.x,
      dy = arm.end.y - arm.wrist.y,
      lengthSq = Math.max(dx * dx + dy * dy, 1e-9);
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        const index = y * width + x;
        if (alpha[index] === 255) continue;
        const p = { x: (x + 0.5) / width, y: (y + 0.5) / height };
        const backed = person[index] && (!region || region.connected[index]);
        if (backed) {
          const t = ((p.x - arm.wrist.x) * dx + (p.y - arm.wrist.y) * dy) / lengthSq;
          if (
            t >= -0.12 &&
            t <= 1.04 &&
            segmentDistance(p, arm.wrist, arm.end) <=
              arm.scale * (0.42 + 0.55 * clamp(t, 0, 1))
          ) {
            alpha[index] = 255;
            continue;
          }
        }
        let inHand = distance(p, arm.palm) <= arm.scale * (backed ? 0.72 : 0.6);
        if (!inHand) {
          const reach = arm.scale * (backed ? 0.13 : 0.11);
          for (const [a, b] of LINKS)
            if (segmentDistance(p, arm.landmarks[a], arm.landmarks[b]) <= reach) {
              inHand = true;
              break;
            }
        }
        if (inHand)
          alpha[index] = backed ? 255 : Math.max(alpha[index], FALLBACK_ALPHA);
      }
  }
  // Sustain pass: hand detection is the least stable signal exactly where the guard lives — a
  // fist half-clipped by the frame edge drops in and out of the landmarker — while the
  // segmenter's person label for a big, close arm is steady. So landmarks only *bootstrap* an
  // arm region; once established (in the smoothed history the worker passes back in), any person
  // pixel touching that region stays lit even on frames with no detection at all. Growth is
  // bounded to one mask-pixel per frame, so a still arm persists indefinitely and a slow-moving
  // one is tracked, while nothing appears where landmarks never confirmed an arm.
  if (support && support.length === alpha.length) {
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const index = y * width + x;
        if (
          alpha[index] === 255 ||
          !person[index] ||
          (region && !region.connected[index])
        )
          continue;
        let near = false;
        for (let oy = -1; oy <= 1 && !near; oy++)
          for (let ox = -1; ox <= 1; ox++) {
            const px = x + ox,
              py = y + oy;
            if (px < 0 || px >= width || py < 0 || py >= height) continue;
            if (support[py * width + px] >= 80) {
              near = true;
              break;
            }
          }
        if (near) alpha[index] = 255;
      }
  }
  return { alpha, anchored: anchoredHands };
}

// Fraction of a guard arc's on-frame area that the mask actually lights. Mask pixels are square
// in height-normalised units (width px = aspect × height px), so a display-space circle is a
// pixel-space circle and the off-frame part of a corner arc simply never enters the count. This
// is what calibration judges — the same fill the user sees — instead of landmark geometry, whose
// off-frame estimated joints made honest guard fists read as oversized.
export function arcCoverage(alpha, width, height, target, radius) {
  const r = radius * height,
    cx = target.x * width,
    cy = target.y * height;
  const x0 = Math.max(0, Math.floor(cx - r)),
    x1 = Math.min(width - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r)),
    y1 = Math.min(height - 1, Math.ceil(cy + r));
  let inside = 0,
    lit = 0;
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx,
        dy = y + 0.5 - cy;
      if (dx * dx + dy * dy > r * r) continue;
      inside++;
      if (alpha[y * width + x] >= 128) lit++;
    }
  return inside ? lit / inside : 0;
}

// Asymmetric temporal filter: new arm pixels appear fast (punches stay responsive) while absent
// pixels fade over a few frames, so single-frame segmenter flicker dims the mask instead of
// carving holes in it.
export class MaskSmoother {
  constructor({ attack = 0.65, decay = 0.3 } = {}) {
    this.attack = attack;
    this.decay = decay;
    this.state = null;
  }
  apply(alpha) {
    if (!this.state || this.state.length !== alpha.length) {
      this.state = new Uint8ClampedArray(alpha);
      return this.state;
    }
    const state = this.state;
    for (let i = 0; i < alpha.length; i++) {
      const target = alpha[i],
        current = state[i];
      const next =
        current + (target - current) * (target > current ? this.attack : this.decay);
      state[i] = next < 6 ? 0 : next;
    }
    return state;
  }
}
