import * as THREE from 'three';
import { ARM_IDS, orientation } from './arm-pose.js';
const v = (p) => new THREE.Vector3(p.x, p.y, p.z),
  degrees = (x) => (x * 180) / Math.PI;
const angle = (a, b) =>
  Math.acos(
    THREE.MathUtils.clamp(a.clone().normalize().dot(b.clone().normalize()), -1, 1),
  );

export function capturePoseSignature(body, hand, side) {
  const ids = ARM_IDS[side];
  if (
    !ids ||
    !ids.every(
      (i) => body?.[i] && [body[i].x, body[i].y, body[i].z].every(Number.isFinite),
    ) ||
    hand?.length !== 21 ||
    !hand.every((p) => [p.x, p.y, p.z].every(Number.isFinite))
  )
    return null;
  const [shoulder, elbow, wrist] = ids.map((i) => v(body[i])),
    upper = shoulder.clone().sub(elbow),
    forearm = wrist.clone().sub(elbow),
    palm = v(hand[9]).sub(v(hand[0])),
    across = v(hand[5]).sub(v(hand[17]));
  const frame = orientation(forearm, upper);
  if (!frame || forearm.length() < 0.06 || upper.length() < 0.06) return null;
  const axis = forearm.clone().normalize(),
    a = upper.clone().addScaledVector(axis, -upper.dot(axis)).normalize(),
    b = across.clone().addScaledVector(axis, -across.dot(axis)).normalize();
  if (a.lengthSq() < 0.5 || b.lengthSq() < 0.5) return null;
  const roll = Math.atan2(axis.dot(new THREE.Vector3().crossVectors(a, b)), a.dot(b));
  const view = new THREE.Vector3(0, 0, 1).applyQuaternion(frame.clone().invert());
  return {
    elbow: angle(upper, forearm),
    wrist: angle(palm, forearm),
    roll,
    view: view.toArray(),
  };
}

export function compareCapturePose(reference, current) {
  if (!current)
    return { ok: false, message: 'Arm pose is uncertain. Show the whole arm clearly.' };
  if (!reference) return { ok: true };
  if (degrees(Math.abs(current.elbow - reference.elbow)) > 15)
    return {
      ok: false,
      message:
        'Your elbow bend changed. Return to the starting bend and turn the whole arm as one piece.',
    };
  if (degrees(Math.abs(current.wrist - reference.wrist)) > 18)
    return {
      ok: false,
      message: 'Your wrist bend changed. Keep the wrist fixed relative to the forearm.',
    };
  const roll = Math.atan2(
    Math.sin(current.roll - reference.roll),
    Math.cos(current.roll - reference.roll),
  );
  if (Math.abs(degrees(roll)) > 25)
    return {
      ok: false,
      message:
        'Your wrist is twisting. Keep the fist fixed and rotate your torso or the entire rigid arm.',
    };
  return { ok: true };
}

export function armViewCoverage(signatures) {
  const views = signatures.map((s) => new THREE.Vector3(...s.view));
  let span = 0;
  for (let i = 0; i < views.length; i++)
    for (let j = 0; j < i; j++)
      span = Math.max(span, degrees(angle(views[i], views[j])));
  return span;
}
