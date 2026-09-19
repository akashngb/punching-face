import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from 'three';
import { FaceDynamics, sweptEllipsoid } from '../src/physics.js';

function fixture() {
  const data = JSON.parse(
    fs.readFileSync(new URL('../public/reference/face-poisson.json', import.meta.url)),
  );
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(data.positions, 3),
  );
  geometry.setIndex(data.indices);
  geometry.computeVertexNormals();
  return new FaceDynamics(geometry);
}

test('swept contact catches a crossing that skips the entire head in one frame', () => {
  assert.notEqual(
    sweptEllipsoid([-0.5, 0, 0], [0.5, 0, 0], [0, 0, 0], [0.08, 0.12, 0.09]),
    null,
  );
  assert.equal(
    sweptEllipsoid([-0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0], [0.08, 0.12, 0.09]),
    null,
  );
  assert.equal(
    sweptEllipsoid([0.5, 0, 0], [0.6, 0, 0], [0, 0, 0], [0.08, 0.12, 0.09]),
    null,
  );
});
test('real reconstructed surface compresses, stays bounded, and returns to rest', () => {
  const d = fixture();
  const index = Array.from(d.rest).findIndex((v, i) => i % 3 === 2 && v > 0.08);
  const point = new THREE.Vector3(...d.rest.slice(index - 2, index + 1));
  assert.ok(d.impulse(point, new THREE.Vector3(0.6, 0, -0.8), 2) > 50);
  let peak = 0;
  for (let frame = 0; frame < 360; frame++) {
    d.step(1 / 120);
    peak = Math.max(peak, d.maxDisplacement);
    assert.ok(d.geometry.attributes.position.array.every(Number.isFinite));
  }
  assert.ok(peak > 0.001 && peak < 0.044);
  assert.ok(d.maxDisplacement < 0.0001);
  assert.ok(d.recoil.length() < 0.001);
});
test('sculpt history restores exact vertex positions independently of animated compression', () => {
  const d = fixture(),
    before = d.rest.slice(),
    p = new THREE.Vector3(...before.slice(0, 3));
  d.remember();
  d.sculpt(p, 0.008);
  const edited = d.rest.slice();
  assert.notDeepEqual(edited, before);
  d.undo();
  assert.deepEqual(d.rest, before);
  d.redo();
  assert.deepEqual(d.rest, edited);
  d.reset();
  assert.deepEqual(d.rest, before);
});
test('export has four finite nonzero morph fields and does not mutate active expression', () => {
  const d = fixture();
  d.rig.smile = 0.4;
  const morphs = d.exportMorphs();
  assert.equal(morphs.length, 4);
  assert.equal(d.rig.smile, 0.4);
  for (const m of morphs) {
    assert.ok(m.array.every(Number.isFinite));
    assert.ok(m.array.some((x) => Math.abs(x) > 0.0001));
  }
});
test('online avatar impact produces local strain rather than rigid head motion', () => {
  const data = JSON.parse(
    fs.readFileSync(new URL('../public/online-face/mesh.json', import.meta.url)),
  );
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3));
  g.setIndex(data.indices);
  g.computeVertexNormals();
  const d = new FaceDynamics(g);
  const target = new THREE.Vector3(-0.045, 0.005, 0.07);
  let index = 0,
    best = Infinity;
  for (let i = 0; i < d.rest.length; i += 3) {
    const q = new THREE.Vector3(...d.rest.slice(i, i + 3));
    const dist = q.distanceToSquared(target);
    if (dist < best) {
      best = dist;
      index = i;
    }
  }
  d.impulse(
    new THREE.Vector3(...d.rest.slice(index, index + 3)),
    new THREE.Vector3(0.8, 0.1, -0.6).normalize(),
    1.3,
  );
  for (let i = 0; i < 10; i++) d.step(1 / 120);
  assert.ok(d.maxDisplacement > 0.004);
  assert.ok(d.regionPeaks.forehead < d.maxDisplacement * 0.15);
  const changed = [];
  for (let i = 0; i < d.offset.length; i += 3)
    changed.push(Math.hypot(...d.offset.slice(i, i + 3)));
  assert.ok(Math.max(...changed) - Math.min(...changed) > 0.004);
  assert.ok(g.attributes.position.array.every(Number.isFinite));
});
