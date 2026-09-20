// Rigid first-person fist pose.
//
// The old estimator unprojected all 21 landmarks independently: 42 image observations against 63
// unknowns, closed with two heuristics (apparent hand size -> depth, MediaPipe relative z -> per
// landmark depth). Both conflate distance with orientation, which is fatal from the puncher's own
// camera because the fist rotates through ~50 deg during a punch. No scalar image measurement of a
// rigid body is invariant to its rotation, so no tuning of that family can work.
//
// A closed fist is a rigid body: 6 DOF against 42 observations, hugely overdetermined. This module
// solves it that way. MediaPipe's metric `worldLandmarks` supply shape and orientation; the image
// landmarks supply the ray each point must lie on; a 3-DOF Gauss-Newton solve places the hand along
// those rays. Rotation is solved for rather than absorbed into depth, so pitching the wrist no
// longer looks like the fist flying away.

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

// MediaPipe hands back landmarks as {x,y,z} objects, not [x,y,z] arrays. Indexing them positionally
// yields undefined -> NaN and every downstream solve silently returns null, which is exactly how
// this module failed in the browser while passing every test: the synthetic fixtures used arrays.
// Both shapes are accepted here so the boundary can never drift again.
export function toVectors(landmarks) {
  if (!Array.isArray(landmarks)) return null;
  const out = new Array(landmarks.length);
  for (let i = 0; i < landmarks.length; i++) {
    const p = landmarks[i];
    if (!p) return null;
    const x = Array.isArray(p) ? p[0] : p.x,
      y = Array.isArray(p) ? p[1] : p.y,
      z = Array.isArray(p) ? p[2] : p.z;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    out[i] = [x, y, z];
  }
  return out;
}
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// Bootstrap shape only. `FistShape` replaces this with the wearer's own hand within a second of
// tracking, so these numbers set the starting point and the handedness convention, nothing more.
// Frame: +x pinky->index across the knuckles, +y wrist->knuckles, +z palm->back of hand. Metres,
// origin at the fist's centroid, sized for an 84 mm knuckle span (population mean).
export const REFERENCE_SPAN = 0.084;
function authoredFist() {
  const p = Array.from({ length: 21 }, () => [0, 0, 0]);
  p[0] = [0.0, -0.055, -0.005];
  const mcp = [
    [5, 0.042, 0.018, 0.002],
    [9, 0.014, 0.022, 0.0],
    [13, -0.014, 0.02, -0.002],
    [17, -0.04, 0.014, -0.005],
  ];
  for (const [i, x, y, z] of mcp) p[i] = [x, y, z];
  // Fingers curl onto the palm: proximal phalanx forward, PIP the far edge, middle and distal
  // phalanges folding back down the palm side (-z). The MCP heads stay the striking surface.
  for (const [i, x] of [
    [5, 0.041],
    [9, 0.014],
    [13, -0.014],
    [17, -0.039],
  ]) {
    p[i + 1] = [x * 1.02, 0.032, -0.008];
    p[i + 2] = [x * 1.0, 0.018, -0.024];
    p[i + 3] = [x * 0.96, 0.0, -0.026];
  }
  p[1] = [0.04, -0.03, -0.01];
  p[2] = [0.05, -0.008, -0.016];
  p[3] = [0.038, 0.008, -0.022];
  p[4] = [0.012, 0.012, -0.026];
  return p;
}
function authoredOpen() {
  const p = authoredFist();
  for (const [i, x] of [
    [5, 0.042],
    [9, 0.014],
    [13, -0.014],
    [17, -0.04],
  ]) {
    p[i + 1] = [x * 1.03, 0.052, 0.004];
    p[i + 2] = [x * 1.05, 0.076, 0.006];
    p[i + 3] = [x * 1.06, 0.096, 0.006];
  }
  p[1] = [0.046, -0.034, -0.006];
  p[2] = [0.068, -0.008, -0.008];
  p[3] = [0.082, 0.012, -0.008];
  p[4] = [0.094, 0.03, -0.008];
  return p;
}
export const CANONICAL_FIST = authoredFist(),
  CANONICAL_OPEN = authoredOpen();

// Apparent size in the image, used ONLY to order simultaneously visible hands by nearness -- the
// closest fist is the one about to land. This is deliberately not a depth estimator: apparent size
// conflates distance with orientation, which is the whole reason this module exists. Comparing two
// hands in the same frame is safe because the comparison is relative and instantaneous.
export function handApparentSpan(landmarks) {
  if (landmarks?.length !== 21) return 0;
  let span = 0;
  for (const [a, b] of [
    [5, 17],
    [0, 9],
    [5, 9],
    [13, 17],
  ])
    span = Math.max(
      span,
      Math.hypot(landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y),
    );
  return span;
}

const centroid = (points) => {
  const c = [0, 0, 0];
  for (const p of points) for (let k = 0; k < 3; k++) c[k] += p[k] / points.length;
  return c;
};
export function recentre(points) {
  const c = centroid(points);
  return points.map((p) => [p[0] - c[0], p[1] - c[1], p[2] - c[2]]);
}

// --- rotation ------------------------------------------------------------------------------
// Horn's quaternion method rather than an SVD Kabsch: the largest eigenvector of the 4x4 profile
// matrix is always a *proper* rotation, so a noisy or near-degenerate frame can never come back as
// a reflection. Power iteration seeded from the previous frame's quaternion converges in a few
// steps and keeps the sign temporally consistent.
export function alignRotation(from, to, seed) {
  const a = recentre(from),
    b = recentre(to),
    S = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let i = 0; i < a.length; i++)
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) S[r * 3 + c] += a[i][r] * b[i][c];
  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = S;
  const N = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
  ];
  // Gershgorin shift makes N positive definite so power iteration converges to the *largest*
  // eigenvalue rather than the largest magnitude one.
  let shift = 0;
  for (let r = 0; r < 4; r++)
    shift = Math.max(
      shift,
      N[r].reduce((sum, value) => sum + Math.abs(value), 0),
    );
  for (let r = 0; r < 4; r++) N[r][r] += shift;
  let q = seed && Number.isFinite(seed[0]) ? [...seed] : [1, 0, 0, 0];
  for (let iteration = 0; iteration < 64; iteration++) {
    const next = [0, 1, 2, 3].map((r) =>
      N[r].reduce((sum, value, c) => sum + value * q[c], 0),
    );
    const length = Math.hypot(...next);
    if (length < 1e-12) {
      q = [1, 0, 0, 0];
      break;
    }
    const normalised = next.map((value) => value / length);
    const delta = Math.hypot(...normalised.map((value, i) => value - q[i]));
    q = normalised;
    if (delta < 1e-10) break;
  }
  if (q[0] < 0) q = q.map((value) => -value); // keep the scalar part positive so slerps never flip
  return { quaternion: q, matrix: quaternionMatrix(q) };
}
export function quaternionMatrix([w, x, y, z]) {
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - z * w),
    2 * (x * z + y * w),
    2 * (x * y + z * w),
    1 - 2 * (x * x + z * z),
    2 * (y * z - x * w),
    2 * (x * z - y * w),
    2 * (y * z + x * w),
    1 - 2 * (x * x + y * y),
  ];
}
export const applyMatrix = (m, p) => [
  m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
  m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
  m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
];
export const applyTranspose = (m, p) => [
  m[0] * p[0] + m[3] * p[1] + m[6] * p[2],
  m[1] * p[0] + m[4] * p[1] + m[7] * p[2],
  m[2] * p[0] + m[5] * p[1] + m[8] * p[2],
];
export function slerp(a, b, t) {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
    target = b;
  if (dot < 0) {
    target = b.map((value) => -value);
    dot = -dot;
  }
  if (dot > 0.9995) {
    const out = a.map((value, i) => value + (target[i] - value) * t),
      length = Math.hypot(...out);
    return out.map((value) => value / length);
  }
  const theta = Math.acos(clamp(dot, -1, 1)),
    sin = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sin,
    wb = Math.sin(t * theta) / sin;
  const out = a.map((value, i) => value * wa + target[i] * wb),
    length = Math.hypot(...out);
  return out.map((value) => value / length);
}

// --- personal hand shape -------------------------------------------------------------------
// MediaPipe's worldLandmarks carry their own per-frame shape jitter, so posing the raw ones still
// leaves ~21% frame-to-frame bone-length variance. Averaging them in a canonical frame recovers a
// stable metric template of *this* wearer's hand — which is also what makes the old "hold fists
// still to calibrate" ritual unnecessary: hand size arrives with the measurement.
export class FistShape {
  constructor({ rate = 0.012, seed = CANONICAL_FIST } = {}) {
    this.template = seed.map((p) => [...p]);
    this.rate = rate;
    this.samples = 0;
    this.quaternion = [1, 0, 0, 0];
  }
  get span() {
    const a = this.template[5],
      b = this.template[17];
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }
  // Unit vector the knuckles face, in template space: wrist -> MCP centroid.
  get strikeAxis() {
    const wrist = this.template[0],
      mcp = [0, 1, 2].map((k) =>
        [5, 9, 13, 17].reduce((sum, i) => sum + this.template[i][k] / 4, 0),
      );
    const d = [0, 1, 2].map((k) => mcp[k] - wrist[k]),
      length = Math.hypot(...d) || 1;
    return d.map((value) => value / length);
  }
  // Returns the rotation taking the template into the observed frame. `update` folds the
  // observation into the running template; it is withheld for an open hand so the fist template
  // never absorbs splayed fingers.
  observe(world, { update = true } = {}) {
    if (!world || world.length !== 21) return null;
    const observed = recentre(world);
    const { quaternion, matrix } = alignRotation(
      this.template,
      observed,
      this.quaternion,
    );
    this.quaternion = quaternion;
    if (!update) return { quaternion, matrix, converged: this.samples >= 12 };
    // Pull the observation back into template space and average there, on a schedule: the first
    // frames move the template hard so it reaches the wearer's real hand in well under a second,
    // then the rate collapses. That last part is what buys rigidity -- a template that keeps
    // chasing worldLandmark noise puts that noise straight back into the bone lengths.
    const rate = this.samples < 12 ? 0.35 : this.samples < 60 ? 0.06 : this.rate;
    for (let i = 0; i < 21; i++) {
      const local = applyTranspose(matrix, observed[i]);
      for (let k = 0; k < 3; k++)
        this.template[i][k] += (local[k] - this.template[i][k]) * rate;
    }
    const drift = centroid(this.template);
    for (const p of this.template) for (let k = 0; k < 3; k++) p[k] -= drift[k];
    this.samples++;
    return { quaternion, matrix, converged: this.samples >= 12 };
  }
  posed(matrix) {
    return this.template.map((p) => applyMatrix(matrix, p));
  }
}

// Viewpoint-independent closure, measured in metres in the hand's own frame rather than from
// fingertip distances in the image. That is what lets it survive the self-occluded fist: the
// fingertips do not need to be *visible*, only estimated in 3D.
export function closureOf(input) {
  const world = toVectors(input);
  if (!world || world.length !== 21) return 0;
  const knuckles = [5, 9, 13, 17].map((i) => world[i]);
  const palm = [0, 1, 2].map((k) => knuckles.reduce((sum, p) => sum + p[k] / 4, 0));
  const span =
    Math.hypot(
      knuckles[0][0] - knuckles[3][0],
      knuckles[0][1] - knuckles[3][1],
      knuckles[0][2] - knuckles[3][2],
    ) || 1e-6;
  const reach =
    [8, 12, 16, 20].reduce(
      (sum, i) =>
        sum +
        Math.hypot(world[i][0] - palm[0], world[i][1] - palm[1], world[i][2] - palm[2]),
      0,
    ) / 4;
  // Curled tips sit ~0.45 spans from the palm centre, an open hand ~1.25 spans.
  return clamp(1 - (reach / span - 0.45) / 0.8, 0, 1);
}

// --- camera --------------------------------------------------------------------------------
// Matches the render camera the debug scene uses, including the cover crop that maps the source
// frame onto a differently-shaped viewport, so the fitted hand lands exactly where it is drawn.
export class PinholeCamera {
  constructor({ fovDegrees = 60, viewAspect = 16 / 9, sourceAspect = 16 / 9 } = {}) {
    this.set({ fovDegrees, viewAspect, sourceAspect });
  }
  set({
    fovDegrees = this.fovDegrees,
    viewAspect = this.viewAspect,
    sourceAspect = this.sourceAspect,
  } = {}) {
    this.fovDegrees = fovDegrees;
    this.viewAspect = viewAspect;
    this.sourceAspect = sourceAspect;
    this.tan = Math.tan((fovDegrees * Math.PI) / 360);
    this.kx = this.tan * viewAspect;
    this.ky = this.tan;
  }
  // source-normalised -> view-normalised, identical to the cover fit used for the video layer
  cover(point) {
    const { sourceAspect: s, viewAspect: v } = this;
    if (!(s > 0) || !(v > 0)) return { x: point.x, y: point.y };
    return s > v
      ? { x: 0.5 + ((point.x - 0.5) * s) / v, y: point.y }
      : { x: point.x, y: 0.5 + ((point.y - 0.5) * v) / s };
  }
  project(p) {
    return {
      x: 0.5 + p[0] / (2 * p[2] * this.kx),
      y: 0.5 - p[1] / (2 * p[2] * this.ky),
    };
  }
  // view-normalised + depth -> render space (three.js camera at the origin looking down -Z)
  unproject(point, depth) {
    return [
      (point.x * 2 - 1) * depth * this.kx,
      (1 - point.y * 2) * depth * this.ky,
      -depth,
    ];
  }
}

// Per-axis image noise in normalised units. y is noisier per unit because the frame is shorter.
const NOISE_X = 2.5 / 1280,
  NOISE_Y = 2.5 / 720;

function solve3(A, b) {
  const d =
    A[0] * (A[4] * A[8] - A[5] * A[7]) -
    A[1] * (A[3] * A[8] - A[5] * A[6]) +
    A[2] * (A[3] * A[7] - A[4] * A[6]);
  if (!Number.isFinite(d) || Math.abs(d) < 1e-16) return null;
  const inv = [
    (A[4] * A[8] - A[5] * A[7]) / d,
    (A[2] * A[7] - A[1] * A[8]) / d,
    (A[1] * A[5] - A[2] * A[4]) / d,
    (A[5] * A[6] - A[3] * A[8]) / d,
    (A[0] * A[8] - A[2] * A[6]) / d,
    (A[2] * A[3] - A[0] * A[5]) / d,
    (A[3] * A[7] - A[4] * A[6]) / d,
    (A[1] * A[6] - A[0] * A[7]) / d,
    (A[0] * A[4] - A[1] * A[3]) / d,
  ];
  return {
    x: [
      inv[0] * b[0] + inv[1] * b[1] + inv[2] * b[2],
      inv[3] * b[0] + inv[4] * b[1] + inv[5] * b[2],
      inv[6] * b[0] + inv[7] * b[1] + inv[8] * b[2],
    ],
    inverse: inv,
  };
}

// Place an already-oriented metric hand along the rays its landmarks were seen on. 42 residuals,
// 3 unknowns; Gauss-Newton converges in two or three steps from a size-ratio seed. The normal
// matrix it accumulates is the information matrix of the estimate, so the covariance handed to the
// filter downstream is measured rather than assumed.
export function solveTranslation(shape, points, camera, seed) {
  const n = Math.min(shape.length, points.length);
  if (n < 6) return null;
  let t;
  if (seed && Number.isFinite(seed[2]) && seed[2] > 0.02) t = [...seed];
  else {
    let metric = 0,
      image = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        metric = Math.max(
          metric,
          Math.hypot(shape[i][0] - shape[j][0], shape[i][1] - shape[j][1]),
        );
        image = Math.max(
          image,
          Math.hypot(
            (points[i].x - points[j].x) * 2 * camera.kx,
            (points[i].y - points[j].y) * 2 * camera.ky,
          ),
        );
      }
    const depth = clamp(metric / Math.max(image, 1e-6), 0.02, 3);
    let cx = 0,
      cy = 0;
    for (let i = 0; i < n; i++) {
      cx += points[i].x / n;
      cy += points[i].y / n;
    }
    t = [
      (cx - 0.5) * 2 * depth * camera.kx,
      -(cy - 0.5) * 2 * depth * camera.ky,
      depth,
    ];
  }
  let residual = Infinity,
    A = null;
  for (let iteration = 0; iteration < 8; iteration++) {
    A = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    const b = [0, 0, 0];
    let sum = 0,
      used = 0;
    for (let i = 0; i < n; i++) {
      const px = shape[i][0] + t[0],
        py = shape[i][1] + t[1],
        pz = shape[i][2] + t[2];
      // A fist held at the lens is genuinely a few centimetres away, and its near landmarks are
      // nearer still. Rejecting those made close hands unsolvable on the target camera.
      if (!(pz > 0.005)) continue;
      const u = 0.5 + px / (2 * pz * camera.kx),
        v = 0.5 - py / (2 * pz * camera.ky);
      const ru = (points[i].x - u) / NOISE_X,
        rv = (points[i].y - v) / NOISE_Y;
      const ju = [
        1 / (2 * pz * camera.kx) / NOISE_X,
        0,
        -px / (2 * pz * pz * camera.kx) / NOISE_X,
      ];
      const jv = [
        0,
        -1 / (2 * pz * camera.ky) / NOISE_Y,
        py / (2 * pz * pz * camera.ky) / NOISE_Y,
      ];
      for (const [j, r] of [
        [ju, ru],
        [jv, rv],
      ]) {
        for (let a = 0; a < 3; a++) {
          b[a] += j[a] * r;
          for (let c = 0; c < 3; c++) A[a * 3 + c] += j[a] * j[c];
        }
      }
      sum += ru * ru + rv * rv;
      used++;
    }
    if (used < 6) return null;
    residual = Math.sqrt(sum / (2 * used));
    const step = solve3(A, b);
    if (!step) return null;
    t = [t[0] + step.x[0], t[1] + step.x[1], t[2] + step.x[2]];
    if (Math.hypot(...step.x) < 1e-7) break;
  }
  if (!(t[2] > 0.01) || !Number.isFinite(t[0])) return null;
  const inverted = solve3(A, [0, 0, 0]);
  // diag(A^-1) is the per-axis variance of the fit; depth is always the loose axis.
  const variance = inverted
    ? [
        Math.abs(inverted.inverse[0]),
        Math.abs(inverted.inverse[4]),
        Math.abs(inverted.inverse[8]),
      ]
    : [1e-4, 1e-4, 1e-3];
  return { translation: t, variance, residual };
}

// MediaPipe reports world landmarks with y down; the render frame is y up and looks down -Z. The
// z sign has flipped between task releases, so rather than hard-coding a guess it is measured.
//
// Reprojection residual cannot decide it: a fist is only ~30 mm thick, so at guard distance a
// depth-mirrored hand reprojects almost identically. What *does* decide it is the image landmarks'
// own z channel, whose convention is fixed (larger = farther from the camera). Correlating that
// against the candidate world z over all 21 points is a strong signal even though each individual
// z is noisy, because only the sign of the sum is being asked for.
const AXIS_OPTIONS = [
  [1, -1, 1],
  [1, -1, -1],
];
export class AxisResolver {
  constructor({ samples = 6 } = {}) {
    this.needed = samples;
    this.score = 0;
    this.samples = 0;
    this.locked = null;
  }
  get signs() {
    return this.locked ?? (this.score >= 0 ? AXIS_OPTIONS[0] : AXIS_OPTIONS[1]);
  }
  get resolved() {
    return this.locked !== null;
  }
  consider(input, landmarks) {
    if (this.locked) return this.locked;
    const world = toVectors(input);
    if (!world) return this.signs;
    const meanWorld = world.reduce((sum, p) => sum + p[2], 0) / world.length;
    const meanImage =
      landmarks.reduce((sum, p) => sum + (p.z || 0), 0) / landmarks.length;
    let covariance = 0;
    for (let i = 0; i < world.length; i++)
      covariance += (world[i][2] - meanWorld) * ((landmarks[i].z || 0) - meanImage);
    // AXIS_OPTIONS[0] keeps world z as-is; a positive covariance means it already points away from
    // the camera, matching the image channel.
    this.score += covariance;
    if (++this.samples >= this.needed) this.locked = this.signs;
    return this.signs;
  }
}

// Direction the knuckles face, from posed points, converted to render space.
function strikeAxis(posed) {
  const wrist = posed[0],
    mcp = [0, 1, 2].map((k) =>
      [5, 9, 13, 17].reduce((sum, i) => sum + posed[i][k] / 4, 0),
    );
  const d = [mcp[0] - wrist[0], mcp[1] - wrist[1], -(mcp[2] - wrist[2])],
    length = Math.hypot(...d) || 1;
  return d.map((value) => value / length);
}

// Signed chirality of the observed metric hand: the triple product of wrist->index-MCP,
// wrist->pinky-MCP and wrist->thumb-MCP, normalised by knuckle span cubed. Chirality is exactly
// the property of a hand that no rotation can change, so this is a handedness measurement that
// does not care how the fist is oriented — unlike MediaPipe's label, which is guessed per frame
// and flickers on a chirally ambiguous head-on fist.
//
// Sign convention: this is evaluated in the sign-resolved fit frame (x image-right, y up,
// +z AWAY from the camera — the frame AxisResolver pins against the image z channel). That basis
// is left-handed relative to render, so a physical RIGHT hand measures POSITIVE, about +0.18 for
// a clean fist. The caller aggregates the signed value over a window and applies a deadband.
export function chiralityOf(mapped) {
  if (!mapped || mapped.length !== 21) return 0;
  const [w, i, p, t] = [mapped[0], mapped[5], mapped[17], mapped[2]];
  const a = [i[0] - w[0], i[1] - w[1], i[2] - w[2]],
    b = [p[0] - w[0], p[1] - w[1], p[2] - w[2]],
    c = [t[0] - w[0], t[1] - w[1], t[2] - w[2]];
  const cross = [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const span = Math.hypot(i[0] - p[0], i[1] - p[1], i[2] - p[2]);
  if (!(span > 1e-6)) return 0;
  return (cross[0] * c[0] + cross[1] * c[1] + cross[2] * c[2]) / (span * span * span);
}

// --- the estimator -------------------------------------------------------------------------
export class FistPoseEstimator {
  constructor({ camera = new PinholeCamera(), shape = new FistShape() } = {}) {
    this.camera = camera;
    this.shape = shape;
    this.axes = new AxisResolver();
    this.translation = null;
    this.matrix = IDENTITY;
  }
  reset() {
    this.translation = null;
    this.shape = new FistShape();
    this.matrix = IDENTITY;
  }
  // landmarks: MediaPipe normalised image landmarks. world: MediaPipe metric worldLandmarks.
  // Returns metric render-space points plus the rigid pose they came from, or null if unsolvable.
  estimate(landmarks, world) {
    if (!landmarks || landmarks.length !== 21 || !world || world.length !== 21)
      return null;
    const metric = toVectors(world);
    if (!metric) return null;
    const points = landmarks.map((point) => this.camera.cover(point));
    const signs = this.axes.consider(metric, landmarks);
    const mapped = metric.map((p) => [
      p[0] * signs[0],
      p[1] * signs[1],
      p[2] * signs[2],
    ]);
    const closure = closureOf(mapped);
    // Rigidity is worth having only while the hand is a fist — which is the only state that can
    // land a punch. An open hand is positioned from its own measured shape instead, so the learned
    // fist template stays clean and the skeleton still tracks.
    const closed = closure >= 0.55;
    const aligned = this.shape.observe(mapped, { update: closed });
    if (!aligned) return null;
    const posed = closed ? this.shape.posed(aligned.matrix) : recentre(mapped);
    const fit = solveTranslation(posed, points, this.camera, this.translation);
    if (!fit) return null;
    this.translation = fit.translation;
    this.matrix = aligned.matrix;
    const [tx, ty, tz] = fit.translation;
    // Render space: x right, y up, -z into the scene.
    const toRender = (p) => [p[0] + tx, p[1] + ty, -(p[2] + tz)];
    return {
      points: posed.map(toRender),
      centre: toRender([0, 0, 0]),
      quaternion: aligned.quaternion,
      matrix: aligned.matrix,
      closure,
      rigid: closed,
      // The knuckle face is the striking surface: wrist -> MCP centroid of the posed hand, so it is
      // read off the actual geometry rather than an assumed axis and holds in both branches.
      knuckleNormal: strikeAxis(posed),
      chirality: chiralityOf(mapped),
      span: this.shape.span,
      variance: [fit.variance[0], fit.variance[1], fit.variance[2]],
      residual: fit.residual,
      converged: aligned.converged && this.axes.resolved,
    };
  }
}
