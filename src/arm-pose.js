import * as THREE from 'three';

export const ARM_IDS = { left: [11, 13, 15], right: [12, 14, 16] };
export const ARM_PAIRS = [
  [0, 1],
  [1, 2],
  [2, 12],
  [3, 12],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 6],
  [8, 9],
  [9, 10],
  [10, 11],
  [11, 10],
  [12, 13],
  [13, 14],
  [14, 15],
  [15, 14],
  [16, 17],
  [17, 18],
  [18, 19],
  [19, 18],
  [20, 21],
  [21, 22],
  [22, 23],
  [23, 22],
];
const vector = (p) => new THREE.Vector3(p.x, p.y, p.z);
const finite = (p) => p && [p.x, p.y, p.z].every(Number.isFinite);
const visible = (p) =>
  finite(p) &&
  (p.visibility ?? 1) >= 0.6 &&
  (p.presence ?? 1) >= 0.6 &&
  p.x >= 0 &&
  p.x <= 1 &&
  p.y >= 0 &&
  p.y <= 1;
const usable = (world, image, ids) =>
  ids.every((i) => finite(world?.[i]) && visible(image?.[i]));

export function orientation(primary, across) {
  if (primary.lengthSq() < 1e-10) return null;
  const y = primary.clone().normalize(),
    x = across.clone().addScaledVector(y, -across.dot(y));
  if (x.lengthSq() < 1e-10) return null;
  x.normalize();
  return new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(x, y, new THREE.Vector3().crossVectors(x, y)),
  );
}

export function palmOrientation(points) {
  return orientation(
    points[9].clone().sub(points[0]),
    points[5].clone().sub(points[17]),
  );
}

// The eye origin and torso axes are inferred from monocular landmarks. They are
// a virtual camera estimate, not recovered first-person camera pixels.
export function calibrateBodyFrame(world, image, profiles = []) {
  if (!usable(world, image, [2, 5, 11, 12])) return null;
  const eye = vector(world[2]).add(vector(world[5])).multiplyScalar(0.5);
  const shoulders = vector(world[11]).add(vector(world[12])).multiplyScalar(0.5);
  const x = vector(world[12]).sub(vector(world[11])).normalize();
  // Eyes sit forward of the shoulders; that offset must not tilt the up axis.
  // Prefer the visible torso, otherwise assume an upright laptop camera.
  let vertical = usable(world, image, [23, 24])
    ? shoulders
        .clone()
        .sub(vector(world[23]).add(vector(world[24])).multiplyScalar(0.5))
    : new THREE.Vector3(0, -1, 0);
  const hasTorso = vertical.lengthSq() > 0.0225;
  if (!hasTorso) vertical.set(0, -1, 0);
  const y = vertical.addScaledVector(x, -vertical.dot(x));
  if (y.lengthSq() < 0.0025 || x.lengthSq() < 0.5) return null;
  y.normalize();
  const z = new THREE.Vector3().crossVectors(x, y).normalize();
  const scales = [];
  for (const profile of profiles) {
    const ids = ARM_IDS[profile.side];
    if (!usable(world, image, ids)) return null;
    const length = vector(world[ids[1]]).distanceTo(vector(world[ids[2]]));
    if (length > 0.08) scales.push(profile.forearmLength / length);
    else return null;
  }
  const scale = scales.length ? scales.reduce((a, b) => a + b, 0) / scales.length : 1;
  if (scale < 0.4 || scale > 2.5) return null;
  return {
    x,
    y,
    z,
    scale,
    orientationSource:
      hasTorso && usable(world, image, [23, 24])
        ? 'visible torso'
        : 'upright camera assumption',
  };
}

// Assign anatomy from body wrist proximity, including crossed hands. Independent
// handedness classifiers can flip when the palm is occluded or the image mirrors.
export function matchHandsToBody(hands, pose, aspect = 1) {
  const sides = ['left', 'right'].filter((side) => visible(pose?.[ARM_IDS[side][2]])),
    candidates = [];
  hands.forEach((hand, index) => {
    if (!finite(hand?.[0])) return;
    for (const side of sides) {
      const wrist = pose[ARM_IDS[side][2]];
      const cost = Math.hypot((hand[0].x - wrist.x) * aspect, hand[0].y - wrist.y);
      if (cost < 0.16) candidates.push({ index, side, cost });
    }
  });
  // At most two hands: select the valid assignment with most matches, then
  // minimum total wrist distance, never assigning both detections to one limb.
  let best = [],
    score = Infinity;
  const consider = (list) => {
    const cost = list.reduce((s, c) => s + c.cost, 0);
    if (list.length > best.length || (list.length === best.length && cost < score)) {
      best = list;
      score = cost;
    }
  };
  for (const c of candidates) {
    consider([c]);
    for (const d of candidates)
      if (c.index !== d.index && c.side !== d.side) consider([c, d]);
  }
  return new Map(best.map((c) => [c.index, c.side]));
}

export function capturedArmProfile(bundle) {
  if (
    !ARM_IDS[bundle.side] ||
    bundle.joints?.length !== 24 ||
    bundle.joints.some(
      (p) => !Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite),
    )
  )
    throw new Error(
      'Arm reconstruction needs 24 finite joint anchors and a left/right side.',
    );
  const rest = bundle.joints.map((p) => new THREE.Vector3(...p)),
    upperLength = rest[0].distanceTo(rest[1]),
    forearmLength = rest[1].distanceTo(rest[2]);
  const hand = rest.slice(3),
    palm = palmOrientation(hand);
  if (
    upperLength < 0.08 ||
    upperLength > 0.65 ||
    forearmLength < 0.1 ||
    forearmLength > 0.55 ||
    !palm
  )
    throw new Error(
      'Captured arm anchors are inconsistent. Inspect the reconstruction before rigging.',
    );
  return { side: bundle.side, rest, upperLength, forearmLength, palm };
}

export function retargetCapturedArm(profile, frame, world, image, handWorld) {
  const ids = ARM_IDS[profile.side];
  if (
    !frame ||
    !usable(world, image, [2, 5, ...ids]) ||
    handWorld?.length !== 21 ||
    !handWorld.every(finite)
  )
    return null;
  const eye = vector(world[2]).add(vector(world[5])).multiplyScalar(0.5);
  const rotate = (v) =>
    new THREE.Vector3(v.dot(frame.x), v.dot(frame.y), v.dot(frame.z));
  const body = ids.map((i) =>
    rotate(vector(world[i]).sub(eye)).multiplyScalar(frame.scale),
  );
  const upper = body[1].clone().sub(body[0]),
    lower = body[2].clone().sub(body[1]);
  if (upper.length() < 0.06 || lower.length() < 0.06) return null;
  // Preserve lengths recovered from the capture; use the webcam for directions.
  const shoulder = body[0],
    elbow = shoulder.clone().addScaledVector(upper.normalize(), profile.upperLength),
    wrist = elbow.clone().addScaledVector(lower.normalize(), profile.forearmLength);
  const observed = handWorld.map((p) => rotate(vector(p))),
    q = palmOrientation(observed);
  if (!q) return null;
  const palmRotation = q.clone().multiply(profile.palm.clone().invert());
  const restHand = profile.rest.slice(3),
    points = [wrist.clone()];
  for (const root of [1, 5, 9, 13, 17]) {
    points[root] = restHand[root]
      .clone()
      .sub(restHand[0])
      .applyQuaternion(palmRotation)
      .add(wrist);
    for (let j = root + 1; j <= root + 3; j++) {
      const direction = observed[j].clone().sub(observed[j - 1]),
        length = restHand[j].distanceTo(restHand[j - 1]);
      if (direction.lengthSq() < 1e-9 || length < 0.001 || length > 0.15) return null;
      points[j] = points[j - 1].clone().addScaledVector(direction.normalize(), length);
    }
  }
  return { joints: [shoulder, elbow, wrist, ...points], palmRotation, palm: q };
}

export function armBoneRotations(profile, pose) {
  const rest = profile.rest,
    target = pose.joints,
    restAcross = rest[8].clone().sub(rest[20]),
    across = target[8].clone().sub(target[20]);
  const restNormal = new THREE.Vector3(0, 0, 1).applyQuaternion(profile.palm),
    normal = new THREE.Vector3(0, 0, 1).applyQuaternion(pose.palm);
  return ARM_PAIRS.map(([a, b], i) => {
    if (i === 2 || i === 3) return pose.palmRotation.clone();
    const from = rest[b].clone().sub(rest[a]),
      to = target[b].clone().sub(target[a]);
    const before = orientation(from, i < 2 ? restAcross : restNormal),
      after = orientation(to, i < 2 ? across : normal);
    if (before && after) return after.multiply(before.invert());
    return from.lengthSq() > 1e-10 && to.lengthSq() > 1e-10
      ? new THREE.Quaternion().setFromUnitVectors(from.normalize(), to.normalize())
      : new THREE.Quaternion();
  });
}
