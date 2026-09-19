import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { TissueField } from '../src/tissue-field.js';
import {
  DeformationGradientRig,
  boundedTangents,
} from '../src/deformation-gradient.js';

function fixture(geometry = new THREE.PlaneGeometry(0.16, 0.2, 20, 24)) {
  const tissue = new TissueField(
    geometry.attributes.position.array,
    geometry.index.array,
    geometry.attributes.normal.array,
  );
  return { tissue, rig: new DeformationGradientRig(tissue), rest: tissue.rest };
}
const maximumDifference = (a, b) =>
  a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

test('tangent singular-value projection retains rotation and clamps directional stretch', () => {
  const angle = 1.1,
    u = [Math.cos(angle), 0, Math.sin(angle)],
    v = [0, 1, 0];
  const rigid = boundedTangents(u, v);
  assert.ok(maximumDifference(rigid.u, u) < 1e-12);
  assert.ok(maximumDifference(rigid.v, v) < 1e-12);
  const strained = boundedTangents(
    u.map((x) => x * 3),
    v.map((x) => x * 0.2),
  );
  assert.ok(Math.abs(Math.hypot(...strained.u) - 1.45) < 1e-10);
  assert.ok(Math.abs(Math.hypot(...strained.v) - 0.6) < 1e-10);
  assert.ok(strained.u.every((x, j) => Math.abs(x / 1.45 - u[j]) < 1e-10));
});

test('neutral and rigid poses preserve identity, detail, rotation, and translation', () => {
  const { rest, rig } = fixture(new THREE.SphereGeometry(0.1, 24, 20));
  const neutral = new Float32Array(rest.length);
  rig.refine(neutral);
  assert.equal(maximumDifference(neutral, new Float32Array(rest.length)), 0);
  const field = rest.map((v, i) => {
    const j = i % 3,
      k = i - j;
    return j === 0
      ? Math.cos(0.9) * v - Math.sin(0.9) * rest[k + 1] - v + 0.02
      : j === 1
        ? Math.sin(0.9) * rest[k] + Math.cos(0.9) * v - v - 0.01
        : 0.03;
  });
  const expected = field.slice();
  rig.refine(field);
  assert.ok(maximumDifference(field, expected) < 1e-7);
  assert.equal(rig.lastSolve.limitedTriangles, 0);
});

test('a sharp local pull is redistributed into a connected patch without drifting the boundary', () => {
  const { rest, rig, tissue } = fixture();
  const field = new Float32Array(rest.length);
  const seed = tissue.nearest([0, 0, 0]).index;
  field[seed * 3 + 2] = 0.04;
  rig.refine(field);
  assert.ok(rig.lastSolve.accepted, JSON.stringify(rig.lastSolve));
  assert.ok(rig.lastSolve.relativeResidual < 0.002);
  assert.ok(rig.lastSolve.limitedTriangles > 0);
  assert.ok(field[seed * 3 + 2] < 0.032);
  assert.ok(field[seed * 3 + 2] > 0.005);
  const neighbors = tissue.vertices[tissue.map[seed]].links.map(
    ([i]) => tissue.vertices[i].copies[0],
  );
  assert.ok(neighbors.some((i) => field[i * 3 + 2] > 0.0005));
  assert.ok(Math.abs(field[2]) < 0.0001);
});

test('seam copies remain welded and a nearby disconnected layer receives no impact', () => {
  const g = new THREE.PlaneGeometry(0.12, 0.12, 12, 12).toNonIndexed();
  const rest = g.attributes.position.array;
  const both = new Float32Array(rest.length * 2 + 3);
  both.set(rest);
  both.set(rest, rest.length);
  for (let i = rest.length + 2; i < both.length - 3; i += 3) both[i] += 0.001;
  // An unrendered cage node is intentionally not included in the triangles.
  both.set([0, 0, 0.01], both.length - 3);
  const tissue = new TissueField(
    both,
    Array.from({ length: (rest.length * 2) / 3 }, (_, i) => i),
  );
  const rig = new DeformationGradientRig(tissue),
    field = new Float32Array(both.length);
  for (let i = 0; i < rest.length; i += 3)
    field[i + 2] = 0.035 * Math.exp(-(rest[i] ** 2 + rest[i + 1] ** 2) / 0.000025);
  rig.refine(field);
  assert.ok(rig.lastSolve.accepted);
  assert.ok(field.every(Number.isFinite));
  assert.ok(field.slice(rest.length).every((v) => v === 0));
  for (const v of tissue.vertices)
    for (const copy of v.copies)
      assert.deepEqual(
        field.slice(copy * 3, copy * 3 + 3),
        field.slice(v.copies[0] * 3, v.copies[0] * 3 + 3),
      );
});
