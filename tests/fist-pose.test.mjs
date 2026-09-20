import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  alignRotation,
  applyMatrix,
  solveTranslation,
  closureOf,
  FistShape,
  AxisResolver,
  PinholeCamera,
  FistPoseEstimator,
  recentre,
  slerp,
  toVectors,
} from '../src/fist-pose.js';
import { PoseFilter } from '../src/fist-filter.js';
import { HeadCollider, StrikeTracker, strikeMode } from '../src/strike-system.js';
import { observe, makeRandom, TRUTH_FIST, rotate } from './helpers/synthetic-hand.mjs';
import { SCENARIOS, HEAD } from './helpers/ab-scenarios.mjs';

const camera = new PinholeCamera({
  fovDegrees: 60,
  viewAspect: 16 / 9,
  sourceAspect: 16 / 9,
});
const LINKS = [
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 13],
  [13, 17],
  [0, 17],
];

test('a rotation is recovered as a rotation, never a reflection', () => {
  const rotated = TRUTH_FIST.map((p) =>
    rotate(p, { pitch: 0.9, yaw: -0.7, roll: 0.4 }),
  );
  const { matrix } = alignRotation(TRUTH_FIST, rotated);
  const determinant =
    matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
    matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
    matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  assert.ok(
    Math.abs(determinant - 1) < 1e-6,
    `proper rotation, got det ${determinant}`,
  );
  for (let i = 0; i < 21; i++) {
    const mapped = applyMatrix(matrix, recentre(TRUTH_FIST)[i]);
    assert.ok(Math.hypot(...mapped.map((v, k) => v - recentre(rotated)[i][k])) < 1e-6);
  }
});

test('translation is recovered from the rays the landmarks were seen on, at any wrist angle', () => {
  for (const rotation of [
    {},
    { pitch: 1.0 },
    { yaw: 0.8 },
    { pitch: 0.5, yaw: 0.6, roll: 0.4 },
  ]) {
    const random = makeRandom(3),
      position = [0.04, -0.03, -0.27];
    const o = observe({ position, rotation, closure: 1 }, camera, random, {
      noise: false,
    });
    const shape = recentre(o.truth.points.map((p) => [p[0], p[1], -p[2]]));
    const fit = solveTranslation(
      shape,
      o.landmarks.map((p) => camera.cover(p)),
      camera,
    );
    assert.ok(fit, 'solvable');
    const centre = [fit.translation[0], fit.translation[1], -fit.translation[2]];
    const error = Math.hypot(...centre.map((v, k) => v - o.truth.centre[k]));
    assert.ok(
      error < 0.002,
      `${JSON.stringify(rotation)} recovered within 2 mm, got ${(error * 1000).toFixed(1)} mm`,
    );
  }
});

test('the world-axis convention is resolved from the image z channel, not guessed', () => {
  const resolver = new AxisResolver(),
    random = makeRandom(9);
  for (let i = 0; i < 8; i++) {
    const o = observe(
      { position: [0, 0, -0.26], rotation: { pitch: 0.4 }, closure: 1 },
      camera,
      random,
    );
    resolver.consider(o.worldLandmarks, o.landmarks);
  }
  assert.ok(resolver.resolved);
  // +z of the mapped frame must point away from the camera for the fit to be well posed
  assert.deepEqual(resolver.signs, [1, -1, 1]);
});

test('the shape prior really is the wrong hand to start with', () => {
  const truth = Math.hypot(
    ...[0, 1, 2].map((k) => TRUTH_FIST[5][k] - TRUTH_FIST[17][k]),
  );
  assert.ok(
    Math.abs(new FistShape().span - truth) > 0.005,
    'otherwise the convergence tests prove nothing',
  );
});

test('bone lengths hold steady once the hand template has settled', () => {
  const estimator = new FistPoseEstimator({ camera }),
    random = makeRandom(6),
    frames = [];
  for (let i = 0; i < 260; i++) {
    const o = observe(
      {
        position: [0.03, -0.02, -0.26],
        rotation: { pitch: 0.35 + 0.2 * Math.sin(i / 6) },
        closure: 1,
      },
      camera,
      random,
    );
    const solved = estimator.estimate(o.landmarks, o.worldLandmarks);
    if (solved && i >= 120) frames.push(solved.points);
  }
  assert.ok(frames.length > 100);
  for (const [a, b] of LINKS) {
    const lengths = frames.map((f) =>
      Math.hypot(...[0, 1, 2].map((k) => f[a][k] - f[b][k])),
    );
    const mean = lengths.reduce((s, v) => s + v, 0) / lengths.length;
    const sd = Math.sqrt(
      lengths.reduce((s, v) => s + (v - mean) ** 2, 0) / lengths.length,
    );
    // The legacy per-landmark unprojection runs 40-60% here; anything at this scale is a hand,
    // not a cloud of independent points.
    assert.ok(
      sd / mean < 0.025,
      `bone ${a}-${b} varies ${((100 * sd) / mean).toFixed(1)}% — a settled fist must not stretch`,
    );
  }
});

test('hand size converges inside a second of tracking, with no held calibration pose', () => {
  const estimator = new FistPoseEstimator({ camera }),
    random = makeRandom(4);
  const truth = Math.hypot(
    ...[0, 1, 2].map((k) => TRUTH_FIST[5][k] - TRUTH_FIST[17][k]),
  );
  for (let i = 0; i < 26; i++) {
    // 26 frames ~= 1 s at the pipeline's inference rate
    const o = observe(
      { position: [0, 0, -0.25], rotation: { pitch: 0.3 }, closure: 1 },
      camera,
      random,
    );
    estimator.estimate(o.landmarks, o.worldLandmarks);
  }
  assert.ok(
    Math.abs(estimator.shape.span - truth) < 0.004,
    `converged to ${(estimator.shape.span * 1000).toFixed(1)} mm vs truth ${(truth * 1000).toFixed(1)} mm in 1 s`,
  );
});

test('closure is read in 3D, so a fist stays closed even pointed away from the camera', () => {
  const random = makeRandom(8);
  const fist = observe(
    { position: [0, 0, -0.25], rotation: { pitch: -Math.PI / 2 }, closure: 1 },
    camera,
    random,
    { noise: false },
  );
  const open = observe(
    { position: [0, 0, -0.25], rotation: { pitch: -Math.PI / 2 }, closure: 0 },
    camera,
    random,
    { noise: false },
  );
  const signs = [1, -1, 1];
  const map = (w) =>
    toVectors(w).map((p) => [p[0] * signs[0], p[1] * signs[1], p[2] * signs[2]]);
  assert.ok(
    closureOf(map(fist.worldLandmarks)) > 0.8,
    'self-occluded fist still reads closed',
  );
  assert.ok(closureOf(map(open.worldLandmarks)) < 0.4, 'open hand reads open');
});

test('the filter recovers a constant velocity and stays quiet on a resting hand', () => {
  const moving = new PoseFilter();
  for (let k = 0; k < 20; k++)
    moving.update([0, 0, -(0.25 + 4 * k * 0.038)], [1e-6, 1e-6, 1e-5], k * 38);
  const v = moving.at(19 * 38).velocity;
  assert.ok(
    Math.abs(Math.hypot(...v) - 4) < 0.4,
    `tracked 4 m/s, got ${Math.hypot(...v).toFixed(2)}`,
  );

  const random = makeRandom(12),
    still = new PoseFilter();
  let peak = 0;
  for (let k = 0; k < 160; k++) {
    still.update(
      [
        random.normal() * 0.002,
        random.normal() * 0.002,
        -0.25 + random.normal() * 0.004,
      ],
      [4e-6, 4e-6, 1.4e-5],
      k * 38,
    );
    peak = Math.max(peak, Math.hypot(...still.at(k * 38).velocity));
  }
  assert.ok(
    peak < 2.0,
    `a still hand must not manufacture speed, peaked at ${peak.toFixed(2)} m/s`,
  );
});

test('attack type comes from where the knuckles point, not travel direction alone', () => {
  assert.equal(strikeMode([-0.08, 0.08, -0.99], [-0.14, 0.15, -0.98]), 'jab');
  assert.equal(strikeMode([-0.86, 0.05, -0.5], [-0.95, 0, -0.32]), 'hook');
  assert.equal(strikeMode([0, 0.88, -0.47], [0, 0.99, -0.15]), 'uppercut');
  // the same lateral travel with the knuckles still facing down-range is not a hook
  assert.equal(strikeMode([-0.86, 0.05, -0.5], [-0.1, 0.05, -0.99]), 'jab');
  // without a pose the original direction-only rule is preserved for legacy callers
  assert.equal(strikeMode([-0.86, 0.05, -0.5]), 'hook');
});

test('the capsule sweep hits the head and the broad phase rejects a sweep that cannot', () => {
  const geometry = new THREE.SphereGeometry(1, 48, 32);
  geometry.scale(...HEAD.radii);
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  const root = new THREE.Group();
  root.position.z = -HEAD.distance;
  root.scale.setScalar(HEAD.scale);
  root.add(mesh);
  root.updateMatrixWorld(true);
  const collider = new HeadCollider([mesh], root, () => null);

  const hit = collider.sweep({
    from: [0, 0, -0.3],
    to: [0, 0, -0.55],
    radius: 0.042 * HEAD.scale,
    axis: [1, 0, 0],
    halfLength: 0.042,
    velocity: [0, 0, -4],
    knuckleNormal: [0, 0, -1],
  });
  assert.ok(hit, 'a punch down the centre line connects');
  assert.ok(
    Number.isFinite(hit.t) && hit.t >= 0 && hit.t <= 1,
    'contact time is kept, not discarded',
  );
  assert.ok(hit.normalSpeed > 3, `compression speed reported, got ${hit.normalSpeed}`);
  assert.ok(hit.obliquity < 25, 'a square landing reads as square');

  const before = collider.stats.vertexTests;
  const miss = collider.sweep({
    from: [1.6, 0, -0.3],
    to: [1.6, 0, -0.55],
    radius: 0.042 * HEAD.scale,
    velocity: [0, 0, -4],
  });
  assert.equal(miss, null);
  assert.equal(
    collider.stats.vertexTests,
    before,
    'a sweep nowhere near the head costs no vertex tests',
  );
});

test('a glancing blow is distinguishable from a square one', () => {
  const geometry = new THREE.SphereGeometry(1, 48, 32);
  geometry.scale(...HEAD.radii);
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  const root = new THREE.Group();
  root.position.z = -HEAD.distance;
  root.scale.setScalar(HEAD.scale);
  root.add(mesh);
  root.updateMatrixWorld(true);
  const collider = new HeadCollider([mesh], root, () => null);
  const square = collider.sweep({
    from: [0, 0, -0.3],
    to: [0, 0, -0.55],
    radius: 0.042 * HEAD.scale,
    axis: [1, 0, 0],
    halfLength: 0.042,
    velocity: [0, 0, -4],
    knuckleNormal: [0, 0, -1],
  });
  const rake = collider.sweep({
    from: [0, 0, -0.3],
    to: [0, 0, -0.55],
    radius: 0.042 * HEAD.scale,
    axis: [1, 0, 0],
    halfLength: 0.042,
    velocity: [3.4, 0, -2],
    knuckleNormal: [0, 0, -1],
  });
  assert.ok(
    rake.tangentSpeed > square.tangentSpeed + 1,
    'a raking hit carries far more tangential speed',
  );
  assert.ok(rake.normalSpeed < square.normalSpeed, 'and less compression');
});

test('a still fist never fires, through the whole pipeline', () => {
  const scenario = SCENARIOS.find((s) => s.name === 'wrist rotation only');
  const geometry = new THREE.SphereGeometry(1, 48, 32);
  geometry.scale(...HEAD.radii);
  geometry.computeVertexNormals();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  const root = new THREE.Group();
  root.position.z = -HEAD.distance;
  root.scale.setScalar(HEAD.scale);
  root.add(mesh);
  root.updateMatrixWorld(true);
  const collider = new HeadCollider([mesh], root, () => null);
  const sweep = (s) =>
    collider.sweep({
      ...s,
      radius: (s.radius ?? 0.042) * HEAD.scale,
      halfLength: s.halfLength ?? 0,
    });
  const estimator = new FistPoseEstimator({ camera }),
    filter = new PoseFilter(),
    random = makeRandom(21);
  const strikes = new StrikeTracker({ minFist: 0.15, startSpeed: 0.4, maxGapMs: 240 });
  let fired = 0;
  for (let ms = -26 * 38; ms <= scenario.duration * 1000; ms += 38) {
    const o = observe(scenario.at(Math.max(0, ms) / 1000), camera, random);
    const solved = estimator.estimate(o.landmarks, o.worldLandmarks);
    if (!solved) continue;
    filter.update(solved.centre, solved.variance, ms, solved.quaternion);
    if (ms < 0) continue;
    const state = filter.at(ms);
    const bar = [0, 1, 2].map((k) => solved.points[5][k] - solved.points[17][k]),
      barLength = Math.hypot(...bar) || 1e-6;
    const event = strikes.update(
      {
        hand: -1,
        position: state.position,
        target: [0, 0, -HEAD.distance],
        timestamp: ms,
        closed: solved.closure,
        confidence: 1,
        velocity: state.velocity,
        axis: bar.map((v) => v / barLength),
        halfLength: (barLength / 2) * HEAD.scale,
        knuckleNormal: solved.knuckleNormal,
      },
      sweep,
    );
    if (event) fired++;
  }
  assert.equal(fired, 0, 'rotating the wrist in place is not a punch');
});

test('slerp keeps orientation continuous across a sign flip', () => {
  const a = [1, 0, 0, 0],
    b = [-0.9999, 0.0141, 0, 0];
  const mid = slerp(a, b, 0.5);
  assert.ok(
    Math.hypot(...mid) > 0.999 && Math.hypot(...mid) < 1.001,
    'stays unit length',
  );
  assert.ok(
    Math.abs(mid[0]) > 0.99,
    'takes the short way round rather than spinning 360 deg',
  );
});

test('landmarks arrive from MediaPipe as {x,y,z} objects and must be read as such', () => {
  // This is the shape the browser actually delivers. Reading it positionally yields undefined ->
  // NaN, and every solve returns null while the test suite stays green. Regression guard.
  const objects = [
    { x: 1, y: 2, z: 3, visibility: 1 },
    { x: 4, y: 5, z: 6, visibility: 1 },
  ];
  assert.deepEqual(toVectors(objects), [
    [1, 2, 3],
    [4, 5, 6],
  ]);
  assert.deepEqual(
    toVectors([
      [1, 2, 3],
      [4, 5, 6],
    ]),
    [
      [1, 2, 3],
      [4, 5, 6],
    ],
    'arrays still accepted',
  );
  assert.equal(
    toVectors([{ x: 1, y: 2 }]),
    null,
    'a malformed landmark is rejected, not turned into NaN',
  );

  const camera = new PinholeCamera({
    fovDegrees: 60,
    viewAspect: 4 / 3,
    sourceAspect: 4 / 3,
  });
  const estimator = new FistPoseEstimator({ camera }),
    random = makeRandom(31);
  let solved = 0;
  for (let i = 0; i < 20; i++) {
    const o = observe(
      { position: [0, 0, -0.35], rotation: { pitch: Math.PI / 2 }, closure: 1 },
      camera,
      random,
    );
    assert.ok(
      !Array.isArray(o.worldLandmarks[0]),
      'fixture must emit objects, like MediaPipe does',
    );
    if (estimator.estimate(o.landmarks, o.worldLandmarks)) solved++;
  }
  assert.equal(
    solved,
    20,
    'every frame must solve when the landmark shape is handled correctly',
  );
});
