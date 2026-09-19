import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { FaceDynamics } from '../src/physics.js';
import {
  impactParameters,
  DEFAULT_IMPACT_MAGNITUDE,
  MAX_PERMANENT_DISPLACEMENT,
} from '../src/tissue-field.js';

function fixture() {
  const g = new THREE.SphereGeometry(1, 64, 48);
  g.scale(0.085, 0.135, 0.095);
  g.computeVertexNormals();
  return new FaceDynamics(g);
}
function settle(d, seconds = 2) {
  for (let i = 0; i < seconds * 120; i++) d.step(1 / 120);
}
function hit(d, location, direction, magnitude) {
  const t = d.impactRig.tissue,
    p = t.nearest(location, d.geometry.attributes.position.array);
  return d.applyImpact({
    location: Array.from(
      d.geometry.attributes.position.array.slice(p.index * 3, p.index * 3 + 3),
    ),
    direction,
    magnitude,
  });
}
const peak = (a) => {
  let max = 0;
  for (let i = 0; i < a.length; i += 3)
    max = Math.max(max, Math.hypot(...a.slice(i, i + 3)));
  return max;
};

test('CV contract rejects invalid values and normalizes direction without modifying magnitude', () => {
  assert.deepEqual(
    impactParameters({ location: [1, 2, 3], direction: [0, 0, -4], magnitude: 0.95 }),
    { location: [1, 2, 3], direction: [0, 0, -1], magnitude: 0.95 },
  );
  for (const input of [
    { magnitude: NaN },
    { magnitude: 1.01 },
    { magnitude: -0.01 },
    { direction: [0, 0, 0] },
    { location: [0, Infinity, 0] },
  ])
    assert.throws(
      () =>
        impactParameters({
          location: [0, 0, 0],
          direction: [0, 0, -1],
          magnitude: 0.5,
          ...input,
        }),
      RangeError,
    );
});
test('every surface region accepts inward and oblique impacts without affecting the opposite side', () => {
  for (const [point, dir] of [
    [
      [0, 0, 0.095],
      [0, 0, -1],
    ],
    [
      [0, 0.13, 0],
      [0, -1, 0],
    ],
    [
      [0.085, 0, 0],
      [-1, 0, 0.3],
    ],
    [
      [-0.085, 0, 0],
      [1, 0.4, 0],
    ],
    [
      [0, 0, -0.095],
      [0.2, 0, 1],
    ],
    [
      [0, -0.13, 0],
      [0, 1, -0.4],
    ],
  ]) {
    const d = fixture();
    assert.ok(hit(d, point, dir, 0.7) > 10);
    settle(d, 0.12);
    assert.ok(d.maxDisplacement > 0.001);
    assert.ok(d.geometry.attributes.position.array.every(Number.isFinite));
    const opposite = d.impactRig.tissue.nearest(point.map((v) => -v)).index;
    assert.ok(
      Math.hypot(...d.impactRig.offset.slice(opposite * 3, opposite * 3 + 3)) < 0.002,
      'local deformation must not tunnel through the head',
    );
  }
});
test('zero means no contact and response increases across normalized strengths', () => {
  const peaks = [];
  for (const magnitude of [0, 0.2, 0.5, 0.8]) {
    const d = fixture();
    const count = hit(d, [0, 0.08, 0.07], [0.3, 0, -1], magnitude);
    if (!magnitude) assert.equal(count, 0);
    settle(d, 0.12);
    peaks.push(d.maxDisplacement);
  }
  for (let i = 1; i < peaks.length; i++)
    assert.ok(peaks[i] > peaks[i - 1], JSON.stringify(peaks));
});
test('clay accumulates persistent dents, survives mode changes and restores only on reset', () => {
  const d = fixture();
  d.setHeadMode('clay');
  hit(d, [-0.07, 0, 0.05], [0.9, 0.1, -0.4], 0.7);
  settle(d, 1);
  const first = d.geometry.attributes.position.array.slice();
  assert.ok(d.maxDisplacement > 0.003);
  settle(d, 2);
  assert.deepEqual(d.geometry.attributes.position.array, first);
  hit(d, [0.06, 0, 0.06], [-0.7, 0.2, -0.7], 0.6);
  settle(d, 1);
  assert.notDeepEqual(d.geometry.attributes.position.array, first);
  const saved = d.impactRig.snapshot(),
    expected = d.geometry.attributes.position.array.slice();
  d.setHeadMode('live');
  settle(d, 1);
  assert.deepEqual(d.geometry.attributes.position.array, expected);
  const restored = fixture();
  restored.impactRig.restore(saved);
  restored.step(0);
  assert.deepEqual(restored.geometry.attributes.position.array, expected);
  d.reset();
  assert.deepEqual(d.geometry.attributes.position.array, d.original);
  assert.equal(peak(d.impactRig.permanent), 0);
});
test('successive clay hits deepen the same moving contact patch, including rapid hits', () => {
  for (const interval of [0.2, 0.025]) {
    const d = fixture();
    d.setHeadMode('clay');
    const tissue = d.impactRig.tissue,
      contact = tissue.nearest([-0.07, 0, 0.05]),
      direction = tissue.vertices[contact.node].n.map((v) => -v),
      depths = [];
    for (let hit = 0; hit < 4; hit++) {
      const positions = d.geometry.attributes.position.array;
      assert.ok(
        d.applyImpact({
          location: Array.from(
            positions.slice(contact.index * 3, contact.index * 3 + 3),
          ),
          direction,
          magnitude: DEFAULT_IMPACT_MAGNITUDE,
        }) > 0,
      );
      settle(d, interval);
      const reserved = d.impactRig.permanent.slice();
      for (const event of d.impactRig.events)
        for (let i = 0; i < reserved.length; i++)
          reserved[i] += event.plastic[i] * (1 - event.committed);
      depths.push(
        direction.reduce((sum, v, j) => sum + v * reserved[contact.index * 3 + j], 0),
      );
    }
    for (let i = 1; i < depths.length; i++)
      assert.ok(depths[i] > depths[i - 1] + 0.001, JSON.stringify(depths));
    settle(d, 2);
    assert.ok(peak(d.impactRig.permanent) > 0.05);
    const restored = fixture();
    restored.impactRig.restore(d.impactRig.snapshot());
    restored.step(0);
    assert.deepEqual(
      restored.geometry.attributes.position.array,
      d.geometry.attributes.position.array,
    );
    d.setHeadMode('live');
    settle(d, 2);
    assert.deepEqual(
      d.geometry.attributes.position.array,
      restored.geometry.attributes.position.array,
    );
    d.reset();
    assert.deepEqual(d.geometry.attributes.position.array, d.original);
  }
});
test('default and full-power strikes visibly deform both head modes, with elastic recovery', () => {
  for (const mode of ['clay', 'live']) {
    const peaks = [];
    for (const magnitude of [DEFAULT_IMPACT_MAGNITUDE, 1]) {
      const d = fixture();
      d.setHeadMode(mode);
      hit(d, [-0.07, 0, 0.05], [0.9, 0.1, -0.4], magnitude);
      settle(d, 0.125);
      peaks.push(peak(d.impactRig.offset));
      assert.ok(peaks.at(-1) > 0.022, `${mode}: ${peaks.at(-1)}`);
      settle(d, 2);
      if (mode === 'live' && magnitude === DEFAULT_IMPACT_MAGNITUDE)
        assert.ok(d.maxDisplacement < 0.0001);
    }
    assert.ok(peaks[1] > peaks[0] * 1.05, `${mode}: ${peaks}`);
  }
});
test('live damage uses strict >0.9 threshold and stays concentrated on firm support', () => {
  let damaged = 0;
  for (const magnitude of [0.9, 0.9001, 1]) {
    const d = fixture();
    hit(d, [0, 0.09, 0.065], [0, 0, -1], magnitude);
    settle(d, 2);
    const p = peak(d.impactRig.permanent);
    if (magnitude === 0.9) {
      assert.equal(p, 0);
      assert.ok(d.maxDisplacement < 0.0001);
    } else {
      assert.ok(p > 0);
      assert.ok(p > damaged);
      damaged = p;
    }
  }
  assert.ok(damaged > 0.003);
});
test('repeated clay strikes remain finite, welded across UV seams and bounded', () => {
  const d = fixture();
  d.setHeadMode('clay');
  for (let i = 0; i < 30; i++) {
    hit(d, [0, 0, -0.095], [0.2, 0.1, 1], 1);
    settle(d, 0.15);
  }
  settle(d, 1);
  assert.ok(d.geometry.attributes.position.array.every(Number.isFinite));
  assert.ok(d.maxDisplacement <= MAX_PERMANENT_DISPLACEMENT + 1e-6);
  const t = d.impactRig.tissue,
    p = d.geometry.attributes.position.array;
  for (const node of t.vertices)
    for (const v of node.copies)
      for (let j = 0; j < 3; j++)
        assert.ok(Math.abs(p[v * 3 + j] - p[node.copies[0] * 3 + j]) < 1e-6);
  const quality = t.measure(p);
  assert.equal(quality.reversedTriangles, 0, JSON.stringify(quality));
  assert.ok(quality.minAreaRatio > 0.05, JSON.stringify(quality));
  const adjacent = fixture();
  adjacent.impactRig.restore(d.impactRig.snapshot());
  adjacent.step(0);
  const nearby = adjacent.impactRig.tissue.nearest([0.06, 0, -0.065]),
    nearbyDirection = adjacent.impactRig.tissue.vertices[nearby.node].n.map((v) => -v),
    nearbyBefore = adjacent.impactRig.permanent.slice(
      nearby.index * 3,
      nearby.index * 3 + 3,
    );
  assert.ok(Math.hypot(...nearbyBefore) < MAX_PERMANENT_DISPLACEMENT - 0.005);
  adjacent.applyImpact({
    location: Array.from(
      adjacent.geometry.attributes.position.array.slice(
        nearby.index * 3,
        nearby.index * 3 + 3,
      ),
    ),
    direction: nearbyDirection,
    magnitude: DEFAULT_IMPACT_MAGNITUDE,
  });
  settle(adjacent, 0.2);
  const nearbyAdded = nearbyDirection.reduce(
    (sum, v, j) =>
      sum + v * (adjacent.impactRig.permanent[nearby.index * 3 + j] - nearbyBefore[j]),
    0,
  );
  assert.ok(
    nearbyAdded > 0.001,
    `A saturated vertex must not suppress its unsaturated neighbor: ${nearbyAdded}`,
  );
  const opposite = t.nearest([0, 0, 0.095]).index;
  const before = Array.from(
    d.impactRig.permanent.slice(opposite * 3, opposite * 3 + 3),
  );
  hit(d, [0, 0, 0.095], [0, 0, -1], DEFAULT_IMPACT_MAGNITUDE);
  settle(d, 0.2);
  const added = Math.hypot(
    ...before.map((v, j) => d.impactRig.permanent[opposite * 3 + j] - v),
  );
  assert.ok(added > 0.01, 'A saturated dent must not suppress impacts elsewhere');
});

test('invisible cage landmarks cannot intercept a punch intended for rendered skin', () => {
  const g = new THREE.BufferGeometry();
  g.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(
      [0, 0, 0.002, -0.01, -0.01, 0, 0.01, -0.01, 0, 0, 0.01, 0],
      3,
    ),
  );
  g.setIndex([1, 2, 3]);
  g.computeVertexNormals();
  const d = new FaceDynamics(g);
  const selected = d.impactRig.tissue.nearest([0, 0, 0.002]);
  assert.notEqual(selected.index, 0);
  d.setHeadMode('clay');
  assert.ok(
    d.applyImpact({ location: [0, 0, 0.002], direction: [0, 0, -1], magnitude: 0.7 }) >
      0,
  );
  settle(d, 0.2);
  assert.ok(d.impactRig.permanent.slice(3).some((v) => Math.abs(v) > 0.0001));
});

test('triangle-area guard protects sliver triangles throughout the impact interpolation', () => {
  const g = new THREE.BufferGeometry();
  g.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(
      [0, 0, 0, 0.05, 0, 0, 0.02, 0.00001, 0, 0, 0.02, 0],
      3,
    ),
  );
  g.setIndex([0, 1, 2, 0, 2, 3]);
  g.computeVertexNormals();
  const d = new FaceDynamics(g),
    field = new Float32Array(d.rest.length);
  field[7] = -0.004;
  const positions = d.rest.map((v, i) => v + field[i]);
  assert.ok(d.impactRig.tissue.measure(positions).reversedTriangles > 0);
  d.impactRig.tissue.validity.constrain(field);
  for (let step = 0; step <= 10; step++) {
    const p = d.rest.map((v, i) => v + (field[i] * step) / 10),
      quality = d.impactRig.tissue.measure(p);
    assert.equal(quality.reversedTriangles, 0);
    assert.ok(quality.minAreaRatio > 0.08);
  }
});
