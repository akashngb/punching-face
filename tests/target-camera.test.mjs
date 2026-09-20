import test from 'node:test';
import assert from 'node:assert/strict';
import { TargetTracking, lateralOffset } from '../src/target-camera.js';
import { PinholeCamera } from '../src/fist-pose.js';
import { observe, makeRandom } from './helpers/synthetic-hand.mjs';

const FACING = Math.PI / 2; // the synthetic hand's +y is wrist->knuckles; +90deg aims it at this camera

function rig(options = {}) {
  const video = { videoWidth: 640, videoHeight: 480, readyState: 4 };
  const tracking = new TargetTracking(video, () => {}, { fovDegrees: 60, ...options });
  tracking.active = true;
  tracking.scheduleFrame = () => {};
  return tracking;
}
/**
 * Drive the tracker over a depth trajectory. `at(t)` returns metres of depth in front of the
 * camera. `path` gives a full 3D trajectory (hooks, uppercuts); `blob` synthesises the worker's
 * motion stream.
 */
function drive(
  tracking,
  at,
  {
    duration = 0.8,
    step = 1 / 60,
    seed = 5,
    vanishBelow = null,
    lateral = [0, 0],
    label = 'Right',
    motion = 0.05,
    path = null,
    flicker = false,
    blob = null,
    mirror = false,
    pitch = FACING,
    yaw = 0,
    roll = 0,
  } = {},
) {
  const camera = new PinholeCamera({
    fovDegrees: tracking.fovDegrees,
    viewAspect: 4 / 3,
    sourceAspect: 4 / 3,
  });
  const random = makeRandom(seed),
    impacts = [];
  for (let ms = 0; ms <= duration * 1000; ms += step * 1000) {
    const state = path ? path(ms / 1000) : { lateral, depth: at(ms / 1000) };
    const depth = state.depth,
      here = state.lateral;
    const gone = vanishBelow !== null && depth < vanishBelow;
    const o = observe(
      {
        position: [here[0], here[1], -depth],
        rotation: { pitch, yaw, roll },
        closure: 1,
      },
      camera,
      random,
    );
    // A horizontally mirrored camera: what MediaPipe would emit for the flipped frame.
    if (mirror) {
      o.landmarks = o.landmarks.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
      o.worldLandmarks = o.worldLandmarks.map((p) => ({
        x: -p.x,
        y: p.y,
        z: p.z,
        visibility: 1,
      }));
    }
    // `flicker` mimics MediaPipe relabelling a fist mid-punch, which is chirally ambiguous head-on.
    const shown =
      flicker && Math.floor(ms / (1000 / 60)) % 3 === 0
        ? label === 'Right'
          ? 'Left'
          : 'Right'
        : label;
    const blobs = blob ? blob(ms / 1000, state) : [];
    tracking.results = {
      landmarks: gone ? [] : [o.landmarks],
      worldLandmarks: gone ? [] : [o.worldLandmarks],
      handedness: gone ? [] : [[{ categoryName: shown, score: 0.95 }]],
      motion: { energy: motion, x: 0.5, y: 0.5, peak: 0.4, blobs },
      timestamp: ms,
    };
    const contact = tracking.tick(ms + 10);
    if (contact) impacts.push(contact);
  }
  return impacts;
}
// A punch: fast out, brief hold at full reach, then retract.
const punch =
  ({ from = 0.7, to = 0.34, out = 0.13, hold = 0.08, back = 0.22, at = 0 } = {}) =>
  (t) => {
    const local = t - at;
    if (local < 0) return from;
    if (local < out) return from - (from - to) * (local / out) ** 0.7;
    if (local < out + hold) return to;
    const r = Math.min(1, (local - out - hold) / back);
    return to + (from - to) * r;
  };

test('impact location needs no camera calibration — it is independent of the assumed field of view', () => {
  const offsets = [45, 60, 90].map((fov) => {
    const camera = new PinholeCamera({
      fovDegrees: fov,
      viewAspect: 4 / 3,
      sourceAspect: 4 / 3,
    });
    const span = 0.3,
      depth = 0.092 / (2 * span * camera.kx);
    return lateralOffset({ x: 0.72, y: 0.4 }, depth, camera);
  });
  for (const o of offsets.slice(1))
    assert.ok(
      Math.abs(o.x - offsets[0].x) < 1e-9 && Math.abs(o.y - offsets[0].y) < 1e-9,
      'must not move with the FOV assumption',
    );
  assert.ok(
    Math.abs(offsets[0].x) > 0.05,
    'and the test point is genuinely off-centre',
  );
});

test('a punch that stops in the air still registers — it never has to reach the laptop', () => {
  const tracking = rig({ contactDepth: 0.15 });
  // stops 34 cm out, far short of the 15 cm contact plane the old version required
  const impacts = drive(tracking, punch({ from: 0.7, to: 0.34 }));
  assert.equal(
    impacts.length,
    1,
    `a punch stopping in mid-air must count (got ${impacts.length})`,
  );
  assert.ok(
    impacts[0].depth > 0.25,
    `fired at the apex it actually reached, ${(impacts[0].depth * 100).toFixed(0)} cm`,
  );
  assert.ok(
    impacts[0].closing > 1.5,
    `peak closing speed recorded, got ${impacts[0].closing.toFixed(2)} m/s`,
  );
});

test('one punch fires once; the fist must withdraw before the next can register', () => {
  const tracking = rig();
  assert.equal(drive(tracking, punch(), { duration: 0.9 }).length, 1);
  const two = rig();
  const first = punch({ at: 0 }),
    second = punch({ at: 0.5 });
  assert.equal(
    drive(two, (t) => (t < 0.5 ? first(t) : second(t)), { duration: 1.1 }).length,
    2,
    'two swings, two impacts',
  );
});

test('a slow reach toward the camera is not a punch', () => {
  const tracking = rig();
  const impacts = drive(
    tracking,
    (t) => Math.max(0.25, 0.7 - 0.45 * Math.min(1, t / 3)),
    { duration: 2.6 },
  );
  assert.equal(impacts.length, 0, 'never crosses the closing-speed floor');
});

test('a twitch that goes nowhere is not a punch', () => {
  const tracking = rig({ minTravel: 0.08 });
  const impacts = drive(
    tracking,
    punch({ from: 0.5, to: 0.465, out: 0.05, hold: 0.04, back: 0.1 }),
    { duration: 0.6 },
  );
  assert.equal(impacts.length, 0, 'fast but only 3.5 cm of travel');
});

test('a fist that blurs out at full extension still lands', () => {
  const tracking = rig();
  // landmarks vanish below 0.40 m and never come back — the punch is lost exactly at full reach
  const impacts = drive(tracking, punch({ from: 0.75, to: 0.3 }), {
    duration: 0.9,
    vanishBelow: 0.4,
  });
  assert.equal(
    impacts.length,
    1,
    'losing the hand mid-extension is evidence of a punch, not absence of one',
  );
  assert.equal(impacts[0].stale, true);
});

test('the motion stream carries a blur-dead punch to a confirmed apex through the shell', () => {
  const tracking = rig();
  const camera = new PinholeCamera({
    fovDegrees: 60,
    viewAspect: 4 / 3,
    sourceAspect: 4 / 3,
  });
  let lastDepth = null,
    lastU = null,
    lastV = null,
    lastT = null;
  const impacts = drive(
    tracking,
    punch({ from: 0.72, to: 0.32, out: 0.14, hold: 0.06, back: 0.2 }),
    {
      duration: 1.0,
      vanishBelow: 0.52,
      blob: (t, state) => {
        if (state.depth >= 0.55) return [];
        const u = 0.5 + state.lateral[0] / (2 * state.depth * camera.kx);
        const v = 0.5 - state.lateral[1] / (2 * state.depth * camera.ky);
        const expand =
          lastDepth !== null && t > lastT
            ? Math.log(lastDepth / state.depth) / (t - lastT)
            : 0;
        const du = lastU !== null && t > lastT ? (u - lastU) / (t - lastT) : 0;
        const dv = lastV !== null && t > lastT ? (v - lastV) / (t - lastT) : 0;
        lastDepth = state.depth;
        lastU = u;
        lastV = v;
        lastT = t;
        return [{ u, v, mass: 0.4, spread: 0.1, expand, du, dv }];
      },
    },
  );
  assert.equal(
    impacts.length,
    1,
    `the blob stream must land the punch exactly once (got ${impacts.length})`,
  );
  assert.ok(
    impacts[0].bridged,
    'and the event records that the motion stream carried it',
  );
});

test('left and right hands keep independent shape models', () => {
  const tracking = rig();
  drive(tracking, punch(), { label: 'Right', duration: 0.9 });
  drive(tracking, punch(), { label: 'Left', seed: 11, duration: 0.9 });
  assert.equal(
    tracking.estimators.size,
    2,
    'one learned hand template per hand, not one shared between them',
  );
  for (const [key, estimator] of tracking.estimators)
    assert.ok(
      estimator.shape.span > 0.06 && estimator.shape.span < 0.13,
      `${key} template stayed a plausible hand (${(estimator.shape.span * 1000).toFixed(0)} mm)`,
    );
});

// Where you punch is where the marker shows up, from your own point of view. The target camera
// looks AT you, so its image x runs opposite yours; the scene is rendered from your eyes, where +x
// is screen right. That makes the flip a fact of the setup, not a setting.
test('punch to your right and the marker appears on your right', () => {
  const right = drive(
    rig({ targetWidth: 0.22, headWidth: 0.22 }),
    punch({ from: 0.7, to: 0.34 }),
    { lateral: [-0.09, 0] },
  );
  assert.equal(right.length, 1);
  assert.ok(
    right[0].point[0] > 0.03,
    `a punch to your right must land at scene +x (screen right), got ${right[0].point[0].toFixed(3)}`,
  );

  const left = drive(
    rig({ targetWidth: 0.22, headWidth: 0.22 }),
    punch({ from: 0.7, to: 0.34 }),
    { lateral: [0.09, 0] },
  );
  assert.equal(left.length, 1);
  assert.ok(
    left[0].point[0] < -0.03,
    `and a punch to your left at scene -x, got ${left[0].point[0].toFixed(3)}`,
  );
});

test('punch high and the marker is high; the vertical axis never flips', () => {
  const high = drive(
    rig({ targetWidth: 0.22, headWidth: 0.22 }),
    punch({ from: 0.7, to: 0.34 }),
    { lateral: [0, 0.09] },
  );
  const low = drive(
    rig({ targetWidth: 0.22, headWidth: 0.22 }),
    punch({ from: 0.7, to: 0.34 }),
    { lateral: [0, -0.09] },
  );
  assert.ok(
    high[0].point[1] > 0.03,
    `raised fist lands high, got ${high[0].point[1].toFixed(3)}`,
  );
  assert.ok(
    low[0].point[1] < -0.03,
    `lowered fist lands low, got ${low[0].point[1].toFixed(3)}`,
  );
});

test('a right hook travelling leftward across you draws its arrow leftward on screen', () => {
  // thrown from your right, swinging in across the target: on screen the fist travels right-to-left
  const tracking = rig();
  const from = { lateral: [-0.34, 0.02], depth: 0.36 },
    to = { lateral: [-0.06, 0.02], depth: 0.3 };
  const impacts = drive(tracking, null, { path: arc(from, to), duration: 0.85 });
  assert.equal(impacts.length, 1);
  assert.ok(
    impacts[0].point[0] > 0.05,
    "lands on the right of your screen — the target's left cheek",
  );
  assert.ok(
    impacts[0].direction[0] < -0.5,
    `and drives leftward across the screen (dx ${impacts[0].direction[0].toFixed(2)})`,
  );
  assert.equal(impacts[0].mode, 'hook', 'classified from the full window');
  assert.equal(
    impacts[0].hand,
    'right',
    "entry side and azimuth say the puncher's right hand",
  );
});

test('a mirrored feed, declared mirrored, names the hand that threw the hook and lands it on the correct cheek', () => {
  // A mirrored camera flips every geometric measurement coherently — fitted chirality, wrist
  // trail, sweep, entry side, even MediaPipe's label, which is computed from the same flipped
  // image. Live symptom: a left hook read as a RIGHT hook and landed on the wrong side of the
  // face. Declaring the interpretation un-flips everything at the shell boundary.
  const right = rig();
  right.setMirrored(true);
  // Physical right hook: from the puncher's right, sweeping across. The mirrored image of a
  // right hand has left-hand geometry, which this landmarker (it names what it sees — the
  // physical-looking hand, no selfie assumption) labels 'Left'; the label mirrors along with
  // everything else, and the shell's declared-mirrored flip restores it.
  const rightImpacts = drive(right, null, {
    mirror: true,
    label: 'Left',
    duration: 0.85,
    path: arc(
      { lateral: [-0.34, 0.02], depth: 0.36 },
      { lateral: [-0.06, 0.02], depth: 0.3 },
    ),
  });
  assert.equal(rightImpacts.length, 1);
  assert.equal(rightImpacts[0].mode, 'hook');
  assert.equal(
    rightImpacts[0].hand,
    'right',
    'the hand that threw it, not its mirror image',
  );
  assert.ok(
    rightImpacts[0].point[0] > 0.05,
    `lands screen-right, the target's left cheek (got ${rightImpacts[0].point[0].toFixed(3)})`,
  );
  assert.ok(rightImpacts[0].direction[0] < -0.5);
  const left = rig();
  left.setMirrored(true);
  const leftImpacts = drive(left, null, {
    mirror: true,
    label: 'Right',
    duration: 0.85,
    path: arc(
      { lateral: [0.34, 0.02], depth: 0.36 },
      { lateral: [0.06, 0.02], depth: 0.3 },
    ),
  });
  assert.equal(leftImpacts.length, 1);
  assert.equal(leftImpacts[0].hand, 'left');
  assert.ok(
    leftImpacts[0].point[0] < -0.05,
    `the left hook lands screen-left (got ${leftImpacts[0].point[0].toFixed(3)})`,
  );
});

test('the punch-area slider maps real offsets onto whatever head is loaded', () => {
  const tracking = rig({ targetWidth: 0.3, headWidth: 0.15 });
  assert.ok(Math.abs(tracking.gain - 0.5) < 1e-9);
  tracking.setTargetWidth(0.15);
  assert.ok(Math.abs(tracking.gain - 1) < 1e-9);
});

test('diagnostics say why a punch was rejected', () => {
  const tracking = rig({ minPeak: 6 });
  // held at guard first, so the track is not born mid-flight — a born-fast track legitimately
  // gets softened gates, and this test is about the FULL gate naming its rejection
  drive(tracking, punch({ at: 0.35 }), { duration: 1.3 });
  assert.match(
    tracking.stats.rejected,
    /peak/,
    `should name the failing gate, got "${tracking.stats.rejected}"`,
  );
  assert.ok(
    tracking.stats.frames > 30 && tracking.stats.detected > 30,
    'and confirm frames were actually processed',
  );
});

// A hook and an uppercut both finish well away from the lens: the hook out to the side with real
// depth still remaining, the uppercut below and short. Scored on depth alone neither of them closes
// at all; scored on range to the head centre, both do.
const arc =
  (from, to, { out = 0.16, hold = 0.07, back = 0.24 } = {}) =>
  (t) => {
    let s;
    if (t < out) s = (t / out) ** 0.7;
    else if (t < out + hold) s = 1;
    else s = Math.max(0, 1 - (t - out - hold) / back);
    const lerp = (a, b) => a + (b - a) * s;
    return {
      lateral: [
        lerp(from.lateral[0], to.lateral[0]),
        lerp(from.lateral[1], to.lateral[1]),
      ],
      depth: lerp(from.depth, to.depth),
    };
  };

test('a hook landing on the side of the head registers, though its depth barely changes', () => {
  const tracking = rig();
  // swings from out wide to the cheek. Depth moves only 5 cm — below the reach gate — while the
  // range to the head centre closes 16 cm. Depth-only scoring cannot see this punch at all.
  const from = { lateral: [0.34, 0.02], depth: 0.36 },
    to = { lateral: [0.11, 0.02], depth: 0.31 };
  const depthTravel = from.depth - to.depth;
  assert.ok(
    depthTravel < tracking.minTravel,
    `depth alone only moves ${(depthTravel * 100).toFixed(0)}cm — under the ${(tracking.minTravel * 100).toFixed(0)}cm gate`,
  );
  const impacts = drive(tracking, null, { path: arc(from, to), duration: 0.85 });
  assert.equal(
    impacts.length,
    1,
    'the hook must register on range even though depth hardly moves',
  );
  assert.ok(
    Math.abs(impacts[0].point[0]) > Math.abs(impacts[0].point[2]),
    `lands on the side, not the front (x ${impacts[0].point[0].toFixed(3)} vs z ${impacts[0].point[2].toFixed(3)})`,
  );
});

test('an uppercut arriving from below registers', () => {
  const tracking = rig();
  const from = { lateral: [0.04, -0.34], depth: 0.4 },
    to = { lateral: [0.04, -0.09], depth: 0.26 };
  // pitch π/4: an uppercut's knuckles angle UP at contact. Face-on knuckles (the FACING default)
  // are a straight punch's signature and correctly veto the uppercut classification.
  const impacts = drive(tracking, null, {
    path: arc(from, to),
    duration: 0.85,
    pitch: Math.PI / 6,
  });
  assert.equal(impacts.length, 1, 'rising punches close on range too');
  assert.ok(
    impacts[0].point[1] < -0.02,
    `and land low on the head (y ${impacts[0].point[1].toFixed(3)})`,
  );
  assert.equal(impacts[0].mode, 'uppercut');
});

test('a hand sweeping laterally past the head at constant range is not a punch', () => {
  const tracking = rig();
  // constant range, pure tangential motion — closing speed along the line to centre stays ~0
  const impacts = drive(tracking, null, {
    duration: 0.9,
    path: (t) => {
      const angle = -0.9 + 1.8 * Math.min(1, t / 0.45),
        radius = 0.42;
      return {
        lateral: [radius * Math.sin(angle), 0.02],
        depth: radius * Math.cos(angle),
      };
    },
  });
  assert.equal(impacts.length, 0, 'moving across the target is not moving into it');
});

// Where a punch LANDS is decided by the direction it arrived from, not by where the fist stopped.
// Every punch converges on the camera, so the stopping point is always near the face centre — which
// is exactly why hooks and uppercuts were both landing on the nose.
function zoneOf(impact, radii = [0.15, 0.14, 0.09]) {
  const [x, y, z] = impact.point,
    n = [x / radii[0], y / radii[1], z / radii[2]];
  const biggest = n
    .map(Math.abs)
    .reduce((best, v, i, arr) => (arr[best] > v ? best : i), 0);
  return biggest === 0
    ? x > 0
      ? 'right cheek'
      : 'left cheek'
    : biggest === 1
      ? y > 0
        ? 'forehead'
        : 'chin'
      : z > 0
        ? 'face front'
        : 'back of head';
}

test('an uppercut lands on the chin, not the middle of the face', () => {
  const tracking = rig();
  const from = { lateral: [0.03, -0.34], depth: 0.4 },
    to = { lateral: [0.03, -0.06], depth: 0.24 };
  const impacts = drive(tracking, null, {
    path: arc(from, to),
    duration: 0.85,
    pitch: Math.PI / 6,
  });
  assert.equal(impacts.length, 1);
  const impact = impacts[0];
  assert.equal(
    zoneOf(impact),
    'chin',
    `landed on the ${zoneOf(impact)} at [${impact.point.map((v) => v.toFixed(3))}]`,
  );
  assert.ok(
    impact.direction[1] > 0.45,
    `and drives upward (dy ${impact.direction[1].toFixed(2)})`,
  );
  // the fist came to rest near the centre; the impact must NOT be reported there
  assert.ok(
    Math.abs(impact.apexPoint[1]) < Math.abs(impact.point[1]),
    'reported below where the fist stopped',
  );
});

test('a hook lands on the cheek, not the middle of the face', () => {
  const tracking = rig();
  const from = { lateral: [0.34, 0.02], depth: 0.36 },
    to = { lateral: [0.06, 0.02], depth: 0.3 };
  const impacts = drive(tracking, null, { path: arc(from, to), duration: 0.85 });
  assert.equal(impacts.length, 1);
  const impact = impacts[0];
  assert.match(
    zoneOf(impact),
    /cheek/,
    `landed on the ${zoneOf(impact)} at [${impact.point.map((v) => v.toFixed(3))}]`,
  );
  assert.ok(
    Math.abs(impact.direction[0]) > 0.75,
    `and drives across (dx ${impact.direction[0].toFixed(2)})`,
  );
  assert.ok(
    Math.abs(impact.point[0]) > Math.abs(impact.apexPoint[0]),
    'reported further out than where the fist stopped',
  );
});

test('jab and uppercut separate on where the FIST POINTS, measured in the image', () => {
  // The user-visible fact: a jab arrives with its knuckles square at the camera, an uppercut
  // with them pointing up. The wrist->knuckle axis therefore FORESHORTENS to nearly nothing for
  // a jab and stands tall up the frame for an uppercut — and both readings are pure image
  // geometry. The 3D knuckle normal could not do this job: a jab's axis points along the view
  // axis, so it lives in the depth component, and MediaPipe's depth compression inflated the
  // small honest vertical until jabs read as uppercuts (and truncated uppercuts read as jabs).
  // Measured on real synthetic landmark geometry: a fist aimed at the camera reads ~0.16
  // fist-widths, one tilted 30° up ~0.22, a genuine uppercut 0.4-0.9.
  for (const [pitch, label] of [
    [FACING, 'knuckles square at the camera'],
    [Math.PI / 3, 'fist tilted 30° up'],
  ]) {
    const tracking = rig();
    const impacts = drive(tracking, punch({ from: 0.7, to: 0.3 }), {
      duration: 0.85,
      pitch,
    });
    assert.equal(impacts.length, 1, `${label}: the jab must register`);
    assert.equal(
      impacts[0].mode,
      'jab',
      `${label}: still a jab (got ${impacts[0].mode} via ${impacts[0].why}, axis ${JSON.stringify(impacts[0].handAxis)})`,
    );
  }
  // And the same trajectory thrown with the knuckles up IS an uppercut — orientation alone
  // flips the verdict, which is what lets a truncated uppercut register as one.
  const upright = rig();
  const rising = drive(upright, null, {
    duration: 0.85,
    pitch: Math.PI / 6,
    path: arc(
      { lateral: [0.03, -0.34], depth: 0.4 },
      { lateral: [0.03, -0.06], depth: 0.24 },
    ),
  });
  assert.equal(rising.length, 1);
  assert.equal(
    rising[0].mode,
    'uppercut',
    `knuckles up is an uppercut (via ${rising[0].why})`,
  );
  assert.ok(
    rising[0].handAxis.dv > 0.4,
    `and the fist measurably points up (${rising[0].handAxis.dv.toFixed(2)} fist-widths)`,
  );
});

test('fists idling in guard never log an uppercut — a guard fist is vertical but EDGE-ON', () => {
  // The live symptom: normal guard idling occasionally fired "uppercut". A guard fist shares an
  // uppercut's vertical knuckle axis, but shows the camera its EDGE (back of the fist to the
  // side), where an uppercut shows the BACK. The facing measure separates them (~.1 vs 1.0),
  // and no orientation verdict is allowed from an edge-on hand — however briskly the guard bobs.
  const tracking = rig();
  const bob = (t) => {
    const phase = (t % 1.0) / 1.0;
    let s;
    if (phase < 0.15) s = phase / 0.15;
    else if (phase < 0.4) s = 1;
    else if (phase < 0.55) s = 1 - (phase - 0.4) / 0.15;
    else s = 0;
    return { lateral: [-0.15, -0.02], depth: 0.52 - 0.16 * s };
  };
  const impacts = drive(tracking, null, {
    path: bob,
    duration: 3.0,
    pitch: 0,
    yaw: Math.PI / 2,
  });
  for (const impact of impacts)
    assert.notEqual(
      impact.mode,
      'uppercut',
      `an edge-on guard bob is never an uppercut (got ${impact.mode} via ${impact.why}, facing ${impact.handAxis?.facing?.toFixed(2)})`,
    );
});

test('a hook is recognised by its hand shape: back-on fist, knuckles to the side', () => {
  const tracking = rig();
  const from = { lateral: [0.3, 0.02], depth: 0.42 },
    to = { lateral: [0.06, 0.02], depth: 0.3 };
  const impacts = drive(tracking, null, {
    path: arc(from, to),
    duration: 0.85,
    pitch: 0,
    roll: -Math.PI / 2,
  });
  assert.equal(impacts.length, 1);
  assert.equal(
    impacts[0].mode,
    'hook',
    `(got ${impacts[0].mode} via ${impacts[0].why})`,
  );
  assert.ok(
    Math.abs(impacts[0].handAxis.du) > 0.5 && impacts[0].handAxis.facing > 0.55,
    `the side-pointing back-on fist was measured (axis ${JSON.stringify(impacts[0].handAxis)})`,
  );
});

test('a straight jab still lands on the front of the face', () => {
  const tracking = rig();
  const impacts = drive(tracking, punch({ from: 0.7, to: 0.3 }), { duration: 0.85 });
  assert.equal(impacts.length, 1);
  const impact = impacts[0];
  assert.equal(
    zoneOf(impact),
    'face front',
    `landed on the ${zoneOf(impact)} at [${impact.point.map((v) => v.toFixed(3))}]`,
  );
  assert.ok(
    impact.direction[2] < -0.8,
    `driving into the face (dz ${impact.direction[2].toFixed(2)})`,
  );
  assert.equal(impact.mode, 'jab');
});

test('the reported direction is the punch at speed, not the fist after it has stopped', () => {
  const tracking = rig();
  const impacts = drive(tracking, punch({ from: 0.7, to: 0.3 }), { duration: 0.85 });
  const impact = impacts[0];
  const magnitude = Math.hypot(...impact.direction);
  assert.ok(Math.abs(magnitude - 1) < 1e-6, 'direction is a unit vector');
  assert.ok(
    impact.closing > 1.5,
    `carried from peak closing (${impact.closing.toFixed(2)} m/s), not the apex where the fist is at rest`,
  );
});

test('one punch is one event, even when MediaPipe relabels the hand mid-strike', () => {
  const tracking = rig();
  // The front of a fist is nearly chirally ambiguous, so handedness flickers. Identity is
  // continuity in the image, never the label — a flicker cannot spawn a second event owner.
  const impacts = drive(tracking, punch({ from: 0.7, to: 0.3 }), {
    duration: 0.95,
    flicker: true,
  });
  assert.equal(impacts.length, 1, `one punch, one event (got ${impacts.length})`);
  assert.equal(
    tracking.tracks.size,
    1,
    'and one identity, not one per handedness label',
  );
  assert.equal(
    tracking.stats.refractory,
    0,
    'identity association handled it; the arbitration backstop never fired',
  );
});

test('a flickering label does not fragment the learned hand shape', () => {
  const tracking = rig();
  drive(tracking, punch(), { duration: 0.95, flicker: true });
  const slot = [...tracking.slots.values()][0];
  assert.ok(['Left', 'Right'].includes(slot.label), 'settles on a majority label');
  assert.ok(tracking.estimators.size <= 2, 'and does not spray estimators');
});

test('both fists up: only the one actually punching fires', () => {
  const tracking = rig();
  const camera = new PinholeCamera({
    fovDegrees: tracking.fovDegrees,
    viewAspect: 4 / 3,
    sourceAspect: 4 / 3,
  });
  const random = makeRandom(17),
    impacts = [];
  const swing = punch({ from: 0.7, to: 0.3 });
  for (let ms = 0; ms <= 950; ms += 1000 / 60) {
    const t = ms / 1000;
    // guard hand parked out to the left, punching hand driving in on the right
    const guard = observe(
      { position: [-0.18, -0.02, -0.55], rotation: { pitch: FACING }, closure: 1 },
      camera,
      random,
    );
    const live = observe(
      { position: [0.1, 0.01, -swing(t)], rotation: { pitch: FACING }, closure: 1 },
      camera,
      random,
    );
    tracking.results = {
      landmarks: [guard.landmarks, live.landmarks],
      worldLandmarks: [guard.worldLandmarks, live.worldLandmarks],
      handedness: [
        [{ categoryName: 'Left', score: 0.9 }],
        [{ categoryName: 'Right', score: 0.9 }],
      ],
      motion: { energy: 0.05, x: 0.5, y: 0.5, peak: 0.4, blobs: [] },
      timestamp: ms,
    };
    const contact = tracking.tick(ms + 10);
    if (contact) impacts.push(contact);
  }
  assert.equal(
    impacts.length,
    1,
    `the stationary guard hand must not score (got ${impacts.length})`,
  );
  assert.equal(tracking.tracks.size, 2, 'both hands are tracked separately');
});

test('lifting the elbows — fists still, hands rotating — is not a punch', () => {
  // The live symptom: forearms rotating to horizontal with the fists held in place fired
  // consecutive jab/hook events. Rotation foreshortens and un-foreshortens the knuckle span, and
  // when the rigid fit fails, apparent-size depth reads that as a tens-of-centimetres approach
  // and retreat — a complete phantom punch. Fallback samples are depth-inertial now: a failed
  // fit contributes the lateral the image measures honestly and manufactures no depth.
  const tracking = rig();
  const camera = new PinholeCamera({
    fovDegrees: 60,
    viewAspect: 4 / 3,
    sourceAspect: 4 / 3,
  });
  const random = makeRandom(23),
    impacts = [];
  for (let ms = 0; ms <= 1400; ms += 1000 / 60) {
    const t = ms / 1000;
    const o = observe(
      { position: [0.04, -0.02, -0.45], rotation: { pitch: FACING }, closure: 1 },
      camera,
      random,
    );
    const failing = t >= 0.5 && t < 0.86;
    if (failing) {
      // the "rotation": apparent span doubles and comes back while the rigid fit is unsolvable
      const swell = t < 0.68 ? 1 + (t - 0.5) / 0.18 : 2 - (t - 0.68) / 0.18;
      const cx = o.landmarks.reduce((s, p) => s + p.x, 0) / 21,
        cy = o.landmarks.reduce((s, p) => s + p.y, 0) / 21;
      for (const p of o.landmarks) {
        p.x = cx + (p.x - cx) * swell;
        p.y = cy + (p.y - cy) * swell;
      }
      for (const p of o.worldLandmarks) p.z = NaN; // unusable world: the rigid fit fails
    }
    tracking.results = {
      landmarks: [o.landmarks],
      worldLandmarks: [o.worldLandmarks],
      handedness: [[{ categoryName: 'Left', score: 0.9 }]],
      motion: { energy: 0.03, x: 0.5, y: 0.5, peak: 0.3, blobs: [] },
      timestamp: ms,
    };
    const contact = tracking.tick(ms + 10);
    if (contact) impacts.push(contact);
  }
  assert.ok(tracking.probe.fitFail > 5, 'the degraded-fit path was actually exercised');
  assert.equal(
    impacts.length,
    0,
    `a rotating stationary fist must not punch (got ${impacts.length}: ${impacts.map((i) => i.mode).join(', ')})`,
  );
});

// One fist can become two identities — a blurred frame, a big jump, a momentary loss. In the old
// per-track-lifecycle design each identity ran its own state machine and could fire its own event
// (the observed symptom was a hit followed ~200 ms later by a miss from the orphan). Identity is
// now a bucket of evidence, and candidate arbitration merges overlapping views of the same strike.
function fragmentedHook(tracking, { seed = 5, fps = 60, teleportAt = 120 } = {}) {
  const camera = new PinholeCamera({
    fovDegrees: 60,
    viewAspect: 4 / 3,
    sourceAspect: 4 / 3,
  });
  const random = makeRandom(seed),
    events = [];
  const swing = (t) => {
    const out = 0.16,
      hold = 0.07,
      back = 0.26;
    const s =
      t < out
        ? (t / out) ** 0.7
        : t < out + hold
          ? 1
          : Math.max(0, 1 - (t - out - hold) / back);
    return { lateral: [-0.42 + 0.36 * s, 0.02], depth: 0.38 - 0.08 * s };
  };
  for (let ms = 0; ms <= 1400; ms += 1000 / fps) {
    const state = swing(ms / 1000);
    const jump = teleportAt !== null && Math.abs(ms - teleportAt) < 1000 / fps / 2;
    const lateral = jump
      ? [state.lateral[0] + 0.28, state.lateral[1] + 0.18]
      : state.lateral;
    const o = observe(
      {
        position: [lateral[0], lateral[1], -state.depth],
        rotation: { pitch: FACING },
        closure: 1,
      },
      camera,
      random,
    );
    const centre = {
      x: (o.landmarks[5].x + o.landmarks[17].x) / 2,
      y: (o.landmarks[5].y + o.landmarks[17].y) / 2,
    };
    const visible =
      centre.x >= 0.02 && centre.x <= 0.98 && centre.y >= 0.02 && centre.y <= 0.98;
    tracking.results = {
      landmarks: visible ? [o.landmarks] : [],
      worldLandmarks: visible ? [o.worldLandmarks] : [],
      handedness: visible ? [[{ categoryName: 'Right', score: 0.9 }]] : [],
      motion: { energy: 0.05, x: 0.5, y: 0.5, peak: 0.4, blobs: [] },
      timestamp: ms,
    };
    const contact = tracking.tick(ms + 10);
    if (contact) events.push(contact);
  }
  return events;
}

test('a punch that fragments into two identities still emits one event', () => {
  for (const teleportAt of [120, 190, 260]) {
    const tracking = rig();
    const events = fragmentedHook(tracking, { teleportAt });
    assert.ok(
      events.length <= 1,
      `fragment at ${teleportAt}ms produced ${events.length} events`,
    );
    assert.equal(
      events.length,
      1,
      `fragment at ${teleportAt}ms must still land the punch`,
    );
  }
});

test('a strike spends every nearby view of itself, not only the identity that fired', () => {
  const tracking = rig();
  fragmentedHook(tracking, { teleportAt: 120 });
  for (const slot of tracking.slots.values())
    assert.ok(
      slot.rearm !== null || slot.consumedUntil > -1e8 || slot.samples.length < 3,
      'an unconsumed mid-approach orphan would fire its own version of the same punch once censored',
    );
});

test('the hit counter counts events that actually left, not ones merely formed', () => {
  const tracking = rig();
  const events = fragmentedHook(tracking, { teleportAt: 120 });
  assert.equal(
    tracking.stats.impacts,
    events.length,
    'a counter that includes suppressed duplicates cannot be used to diagnose duplicates',
  );
});
