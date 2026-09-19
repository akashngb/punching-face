import test from 'node:test';
import assert from 'node:assert/strict';
import { FaceImpactRig } from '../src/impact-rig.js';

const anchors = {
  13: [0, -0.04, 0.02],
  14: [0, -0.042, 0.02],
  152: [0, -0.1, 0],
  50: [-0.055, -0.004, 0],
  280: [0.055, -0.004, 0],
  61: [-0.03, -0.041, 0.015],
  291: [0.03, -0.041, 0.015],
  159: [-0.037, 0.04, 0.01],
  386: [0.037, 0.04, 0.01],
};
const point = { x: -0.05, y: -0.004, z: 0 },
  direction = { x: 0.7, y: 0, z: -0.7 };

function rig(points) {
  const rest = new Float32Array(points.flat());
  return { rest, rig: new FaceImpactRig(rest, anchors) };
}

const magnitude = (offset, index) =>
  Math.hypot(...offset.slice(index * 3, index * 3 + 3));

test('a cheek punch moves the lips and jaw, squeezes the eyelids, and leaves the back and scalp attached', () => {
  const points = [
    anchors[50],
    anchors[61],
    anchors[152],
    [-0.037, 0.044, 0.01],
    [-0.037, 0.03, 0.01],
    [0, 0.14, -0.04],
    [0, 0, -0.2],
    [0, -0.15, 0],
  ];
  const { rest, rig: r } = rig(points);
  r.trigger(rest, point, direction, 1, 0.6);
  r.step(0.12);
  assert.ok(magnitude(r.offset, 0) > 0.015, 'cheek must visibly compress and shear');
  assert.ok(magnitude(r.offset, 1) > 0.015, 'lips must follow the cheek');
  assert.ok(magnitude(r.offset, 2) > 0.01, 'the jaw silhouette must respond');
  assert.ok(
    r.offset[0] > 0 && r.offset[3] > 0 && r.offset[6] > 0,
    'transport follows the hook',
  );
  const eyeGap = rest[10] + r.offset[10] - (rest[13] + r.offset[13]);
  assert.ok(eyeGap > 0 && eyeGap < 0.014 * 0.7, 'eyelids squeeze without crossing');
  for (const i of [5, 6, 7])
    assert.equal(magnitude(r.offset, i), 0, 'unaffected attachments remain fixed');
});

test('left and right hooks mirror their facial deformation', () => {
  const points = [
    anchors[50],
    anchors[61],
    anchors[152],
    [-0.037, 0.035, 0.01],
    [-0.085, -0.055, -0.03],
  ];
  const left = rig(points),
    right = rig(points.map(([x, y, z]) => [-x, y, z]));
  left.rig.trigger(left.rest, point, direction, 1, 0.6);
  right.rig.trigger(
    right.rest,
    { ...point, x: -point.x },
    { ...direction, x: -direction.x },
    1,
    0.6,
  );
  left.rig.step(0.12);
  right.rig.step(0.12);
  left.rig.offset.forEach((value, i) =>
    assert.ok(Math.abs(value * (i % 3 === 0 ? -1 : 1) - right.rig.offset[i]) < 1e-7),
  );
});

test('repeated and alternating hooks stay bounded and recover without changing the rest pose', () => {
  const { rest, rig: r } = rig(Object.values(anchors)),
    original = rest.slice();
  let peak = 0;
  for (let frame = 0; frame < 240; frame++) {
    if (frame % 6 === 0) {
      const side = frame % 12 === 0 ? -1 : 1;
      r.trigger(rest, { ...point, x: side * 0.05 }, direction, 4, 1);
    }
    r.step(1 / 120);
    assert.ok(r.offset.every(Number.isFinite));
    for (let i = 0; i < rest.length / 3; i++)
      peak = Math.max(peak, magnitude(r.offset, i));
  }
  assert.ok(peak > 0.015 && peak < 0.033);
  for (let i = 0; i < 150; i++) r.step(1 / 120);
  assert.ok(r.offset.every((v) => v === 0));
  assert.deepEqual(rest, original);
  r.trigger(rest, point, direction, 1, 0.6);
  r.step(0.12);
  r.reset();
  assert.equal(r.events.length, 0);
  assert.ok(r.offset.every((v) => v === 0));
});

test('speed and softness scale the response and paused time preserves a held pose', () => {
  const points = Object.values(anchors),
    peaks = [];
  for (const [speed, softness] of [
    [0.4, 0],
    [0.8, 0],
    [0.8, 1],
  ]) {
    const { rest, rig: r } = rig(points);
    r.trigger(rest, point, direction, speed, softness);
    r.step(0.06);
    assert.equal(r.hasPeaked, false);
    r.step(0.06);
    assert.equal(r.hasPeaked, true);
    const held = r.offset.slice();
    r.step(0);
    assert.deepEqual(r.offset, held);
    peaks.push(Math.max(...points.map((_, i) => magnitude(r.offset, i))));
  }
  assert.ok(peaks[0] < peaks[1] && peaks[1] < peaks[2]);
});
