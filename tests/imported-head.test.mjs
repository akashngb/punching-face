import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { importedHeadGeometry } from '../src/imported-head.js';
import { FaceDynamics } from '../src/physics.js';

test('an interleaved head can deform without overwriting its texture coordinates', () => {
  const source = new THREE.SphereGeometry(0.1, 16, 12);
  const position = source.getAttribute('position');
  const uv = source.getAttribute('uv');
  const packed = new Float32Array(position.count * 5);
  for (let i = 0; i < position.count; i++)
    packed.set(
      [position.getX(i), position.getY(i), position.getZ(i), uv.getX(i), uv.getY(i)],
      i * 5,
    );
  const buffer = new THREE.InterleavedBuffer(packed, 5);
  source.setAttribute('position', new THREE.InterleavedBufferAttribute(buffer, 3, 0));
  source.setAttribute('uv', new THREE.InterleavedBufferAttribute(buffer, 2, 3));
  const mesh = new THREE.Mesh(source);
  mesh.position.y = 0.01;
  mesh.updateMatrixWorld(true);

  const geometry = importedHeadGeometry(mesh);
  const rest = geometry.getAttribute('position').array.slice();
  const dynamics = new FaceDynamics(geometry);
  dynamics.rig.jaw = 1;
  dynamics.step(1 / 60);

  assert.equal(dynamics.original.length, position.count * 3);
  assert.ok(geometry.getAttribute('position').array.every(Number.isFinite));
  assert.notDeepEqual(geometry.getAttribute('position').array, rest);
  assert.deepEqual(geometry.getAttribute('uv').array, uv.array);
  assert.equal(source.getAttribute('position').data, buffer);
  assert.equal(source.getAttribute('position').getY(0), position.getY(0));
  assert.ok(Math.abs(rest[1] - position.getY(0) - 0.01) < 1e-7);
});

test('quantized attributes retain decoded values and accept head-space transforms', () => {
  const source = new THREE.BufferGeometry();
  source.setAttribute(
    'position',
    new THREE.Int16BufferAttribute(
      [-32767, 0, 32767, 32767, 0, 0, 0, 32767, 0],
      3,
      true,
    ),
  );
  source.setAttribute(
    'uv',
    new THREE.Uint16BufferAttribute([0, 65535, 65535, 0, 32768, 32768], 2, true),
  );
  const mesh = new THREE.Mesh(source);
  mesh.scale.setScalar(0.1);
  mesh.position.z = 0.05;
  mesh.updateMatrixWorld(true);
  const geometry = importedHeadGeometry(mesh);

  assert.ok(geometry.getAttribute('position').array instanceof Float32Array);
  assert.equal(geometry.getAttribute('position').normalized, false);
  assert.ok(Math.abs(geometry.getAttribute('position').getZ(0) - 0.15) < 1e-7);
  assert.ok(Math.abs(geometry.getAttribute('position').getX(0) + 0.1) < 1e-7);
  assert.equal(geometry.getAttribute('uv').getY(0), 1);
  assert.ok(Math.abs(geometry.getAttribute('uv').getX(2) - 32768 / 65535) < 1e-7);
  assert.equal(source.getAttribute('position').array[2], 32767);
});

test('an unindexed head above the subdivision threshold has editable triangles', () => {
  const source = new THREE.SphereGeometry(0.1, 40, 30).toNonIndexed();
  assert.ok(source.getAttribute('position').count > 4000);
  const mesh = new THREE.Mesh(source);
  mesh.updateMatrixWorld(true);
  const geometry = importedHeadGeometry(mesh);
  const dynamics = new FaceDynamics(geometry);
  dynamics.rig.smile = 0.5;
  dynamics.step(1 / 60);

  assert.equal(geometry.index.count, source.getAttribute('position').count);
  assert.equal(geometry.index.getX(geometry.index.count - 1), geometry.index.count - 1);
  assert.ok(geometry.getAttribute('position').array.every(Number.isFinite));
  assert.equal(source.index, null);
});
