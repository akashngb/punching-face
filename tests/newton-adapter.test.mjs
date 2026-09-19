import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as THREE from 'three';
import { NewtonFaceDynamics } from '../src/newton-dynamics.js';

function fixture() {
  const positions = Array.from({ length: 468 }, (_, i) => [
    ((i % 20) - 10) * 0.006,
    (Math.floor(i / 20) - 12) * 0.006,
    0,
  ]).flat();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const triangles = [];
  for (let y = 0; y < 22; y++)
    for (let x = 0; x < 19; x++) {
      const a = y * 20 + x;
      triangles.push(a, a + 1, a + 20, a + 1, a + 21, a + 20);
    }
  geometry.setIndex(triangles);
  geometry.computeVertexNormals();
  const anchors = {
    50: [-0.055, -0.004, 0],
    280: [0.055, -0.004, 0],
    13: [0, -0.04, 0.02],
    14: [0, -0.041, 0.02],
    152: [0, -0.1, 0],
    61: [-0.03, -0.04, 0.01],
    291: [0.03, -0.04, 0.01],
    70: [-0.04, 0.065, 0],
    300: [0.04, 0.065, 0],
    159: [-0.04, 0.04, 0],
    386: [0.04, 0.04, 0],
  };
  const cage = { positions, indices: triangles, rigAnchors: anchors };
  const binding = {
    indices: Array.from({ length: 468 }, (_, i) => [i, i, i]).flat(),
    weights: Array.from({ length: 468 }, () => [1, 0, 0]).flat(),
    active: Array(468).fill(1),
  };
  binding.active[1] = 0;
  return { geometry, cage, binding };
}

test('Newton displacement drives the bound skin while excluded head vertices stay fixed', async (t) => {
  const { geometry, cage, binding } = fixture();
  const offsets = Array(1404).fill(0);
  offsets[2] = -0.003;
  offsets[5] = -0.003;
  t.mock.method(globalThis, 'fetch', async (url) => ({
    ok: true,
    json: async () =>
      url.endsWith('open')
        ? { session: 'test', version: 'test', tetrahedra: 1 }
        : { offsets, peakMm: 3 },
  }));
  const d = new NewtonFaceDynamics(geometry, binding, cage);
  d.step(0);
  await d.connect('test');
  await d.advance(1 / 30);
  d.ready = false;
  d.step(1 / 30);
  assert.ok(geometry.attributes.position.array[2] < -0.002);
  assert.equal(geometry.attributes.position.array[5], d.rest[5]);
  assert.equal(d.lastMetrics.peakMm, 3);
});
test('an unavailable Newton service disables impact instead of falling back to spring animation', async (t) => {
  const { geometry, cage, binding } = fixture();
  t.mock.method(globalThis, 'fetch', async () => {
    throw new Error('Service unavailable');
  });
  const d = new NewtonFaceDynamics(geometry, binding, cage);
  await assert.rejects(d.connect('test'), /Service unavailable/);
  assert.equal(d.ready, false);
  assert.equal(d.impulse(new THREE.Vector3(), new THREE.Vector3(0, 0, -1), 2), 0);
  d.step(1 / 30);
  assert.equal(d.maxDisplacement, 0);
});
test('a restored physics connection retains its generation and verifies the complete cage', async (t) => {
  const { geometry, cage, binding } = fixture();
  let opened;
  t.mock.method(globalThis, 'fetch', async (_url, request) => {
    opened = JSON.parse(request.body);
    return {
      ok: true,
      json: async () => ({ session: 'saved', version: 'test', tetrahedra: 1 }),
    };
  });
  const saved = { id: 'capture', generation: 'a'.repeat(32), cage, binding };
  const dynamics = new NewtonFaceDynamics(geometry, saved.binding, saved.cage);
  await dynamics.connect(saved.id, saved.generation);
  assert.equal(opened.generation, saved.generation);
  assert.equal(dynamics.generation, saved.generation);
  assert.equal(
    opened.cageSha256,
    createHash('sha256')
      .update(Buffer.from(new Float32Array(cage.positions).buffer))
      .update(Buffer.from(new Uint32Array(cage.indices).buffer))
      .digest('hex'),
  );
});
test('the landmark jaw control separates lower and upper lips and moves the chin', () => {
  const { geometry, cage, binding } = fixture();
  const d = new NewtonFaceDynamics(geometry, binding, cage);
  d.rig.jaw = 0.5;
  const upper = d.rigDelta(...cage.rigAnchors['13']),
    lower = d.rigDelta(...cage.rigAnchors['14']),
    chin = d.rigDelta(...cage.rigAnchors['152']);
  assert.ok(lower[1] < -0.006);
  assert.ok(Math.abs(upper[1]) < 0.002);
  assert.ok(chin[1] < -0.008);
  assert.ok(lower.every(Number.isFinite));
});
test('captured-face impacts add a broad rig pose without changing Newton offsets or the manual expression', () => {
  const { geometry, cage, binding } = fixture();
  geometry.attributes.position.setXYZ(467, ...cage.rigAnchors['152']);
  const d = new NewtonFaceDynamics(geometry, binding, cage);
  d.rig.smile = 0.35;
  d.step(0);
  d.ready = true;
  assert.ok(
    d.impulse(new THREE.Vector3(-0.05, 0, 0), new THREE.Vector3(0.7, 0, -0.7), 1) > 0,
  );
  d.ready = false;
  for (let i = 0; i < 15; i++) d.step(1 / 120);
  assert.ok(d.maxDisplacement > 0.015);
  assert.ok(d.regionPeaks.lips > 0.008);
  assert.ok(d.regionPeaks.jaw > 0.006);
  assert.ok(d.offset.every((v) => v === 0));
  assert.equal(d.rig.smile, 0.35);
  assert.ok(geometry.attributes.position.array.every(Number.isFinite));
  d.resetMotion();
  d.step(0);
  assert.equal(d.maxDisplacement, 0);
  assert.equal(d.pending.length, 0);
  assert.equal(d.rig.smile, 0.35);
});
