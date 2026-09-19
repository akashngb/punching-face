import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { FaceImpactRig } from '../src/impact-rig.js';
import { PainExpression, painEnvelope } from '../src/pain-expression.js';
import { TissueField } from '../src/tissue-field.js';

const anchors = {
  13: [0, -0.04, 0.07],
  14: [0, -0.042, 0.07],
  152: [0, -0.1, 0.06],
  50: [-0.05, -0.005, 0.07],
  280: [0.05, -0.005, 0.07],
  61: [-0.03, -0.04, 0.07],
  291: [0.03, -0.04, 0.07],
  159: [-0.035, 0.038, 0.07],
  145: [-0.035, 0.029, 0.07],
  386: [0.035, 0.038, 0.07],
  374: [0.035, 0.029, 0.07],
};
function plane() {
  const g = new THREE.PlaneGeometry(0.19, 0.26, 38, 52);
  g.translate(0, 0, 0.07);
  return g;
}
function rig(g = plane(), a = anchors) {
  g.computeVertexNormals();
  return new FaceImpactRig(g.attributes.position.array, a, {
    indices: g.index.array,
    normals: g.attributes.normal.array,
  });
}
function strike(r, id = 50, direction = [0.75, -0.05, -0.6], magnitude = 0.85) {
  const hit = r.tissue.nearest(r.anchors[id]);
  r.impact(
    r.tissue.rest,
    { location: r.tissue.vertices[hit.node].p, direction, magnitude },
    0.75,
  );
}
const peak = (a) => {
  let max = 0;
  for (let i = 0; i < a.length; i += 3)
    max = Math.max(max, Math.hypot(a[i], a[i + 1], a[i + 2]));
  return max;
};
test('pain begins after contact, remains after the dent, and recovers completely', () => {
  const r = rig();
  strike(r);
  assert.equal(painEnvelope(0), 0);
  r.step(0.12);
  assert.equal(r.hasPeaked, false);
  r.step(0.05);
  assert.equal(r.hasPeaked, true);
  const held = r.offset.slice();
  r.step(0);
  assert.deepEqual(r.offset, held);
  r.step(0.48);
  assert.ok(r.envelope(0.65) < 0.003);
  assert.ok(
    peak(r.offset) > 0.01,
    'a clearly visible grimace must remain after compression',
  );
  r.step(1.1);
  assert.ok(r.offset.every((v) => v === 0));
  assert.equal(peak(r.permanent), 0);
  strike(r);
  r.step(0.2);
  r.reset();
  assert.equal(r.events.length, 0);
  assert.equal(peak(r.offset), 0);
});
test('clay does not wince and pain strength grows with the incoming magnitude', () => {
  const clay = rig();
  clay.setMode('clay');
  strike(clay);
  assert.equal(clay.events[0].reaction, null);
  const values = [];
  for (const magnitude of [0.2, 0.5, 0.85]) {
    const r = rig();
    strike(r, 50, [0.75, -0.05, -0.6], magnitude);
    r.step(0.5);
    values.push(peak(r.offset));
  }
  assert.ok(values[0] < values[1] && values[1] < values[2], JSON.stringify(values));
});
test('eyelid skin reacts while detached eyeballs and invisible landmarks remain fixed', () => {
  const skin = plane(),
    eye = new THREE.SphereGeometry(0.01, 12, 10);
  eye.translate(-0.035, 0.033, 0.064);
  const rest = new Float32Array(
    skin.attributes.position.array.length + eye.attributes.position.array.length + 3,
  );
  rest.set(skin.attributes.position.array);
  rest.set(eye.attributes.position.array, skin.attributes.position.array.length);
  const start = skin.attributes.position.count;
  const indices = [
    ...skin.index.array,
    ...Array.from(eye.index.array, (i) => i + start),
  ];
  const tissue = new TissueField(rest, indices),
    r = new PainExpression(tissue);
  const field = r.build(anchors, anchors[50], 0.85);
  assert.ok(peak(field) > 0.01);
  assert.ok(field.slice(start * 3).every((v) => v === 0));
});

const capture = process.env.FACE_IMPACT_CAPTURE;
test(
  'captured-head pain is visible without lid crossing or reversed triangles during recovery',
  { skip: !capture && 'Set FACE_IMPACT_CAPTURE to a local reconstructed head' },
  () => {
    const data = JSON.parse(fs.readFileSync(path.join(capture, 'mesh.json')));
    const cage = JSON.parse(fs.readFileSync(path.join(capture, 'physics-cage.json')));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3));
    g.setIndex(data.indices);
    for (const [anchor, direction] of [
      [50, [0.75, -0.05, -0.6]],
      [280, [-0.75, -0.05, -0.6]],
      [152, [0, 0.85, -0.5]],
    ]) {
      const r = rig(g, cage.rigAnchors);
      strike(r, anchor, direction, 1);
      const nearestSkin = (id) => {
        const point = cage.positions.slice(id * 3, id * 3 + 3);
        let best = Infinity,
          index;
        r.tissue.vertices.forEach((v, i) => {
          if (!r.painExpression.skin[i]) return;
          const d = Math.hypot(...v.p.map((x, j) => x - point[j]));
          if (d < best) {
            best = d;
            index = v.copies[0];
          }
        });
        return index;
      };
      const lids = [
        [159, 145],
        [386, 374],
      ].map((pair) => pair.map(nearestSkin));
      const brows = [107, 336].map(nearestSkin),
        corners = [61, 291].map(nearestSkin);
      const rest = r.tissue.rest;
      for (let frame = 0; frame < 90; frame++) {
        r.step(0.02);
        const p = rest.map((x, i) => x + r.offset[i]);
        const q = r.tissue.measure(p);
        assert.ok(q.finite);
        assert.equal(
          q.reversedTriangles,
          0,
          `${anchor}, frame ${frame}: ${JSON.stringify(q)}`,
        );
        for (const [up, lo] of lids)
          assert.ok(
            p[up * 3 + 1] > p[lo * 3 + 1],
            `lids crossed: ${anchor}, frame ${frame}`,
          );
        if (frame === 17) {
          for (const [up, lo] of lids)
            assert.ok(
              (p[up * 3 + 1] - p[lo * 3 + 1]) / (rest[up * 3 + 1] - rest[lo * 3 + 1]) <
                0.55,
              'both eyes visibly tighten',
            );
          for (const i of brows)
            assert.ok(r.offset[i * 3 + 1] < -0.004, 'brows lower visibly');
          for (const i of corners)
            assert.ok(r.offset[i * 3 + 1] < -0.005, 'mouth corners grimace');
        }
      }
    }
  },
);
