import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { FaceDynamics } from '../src/physics.js';
import {
  DEFAULT_IMPACT_MAGNITUDE,
  MAX_PERMANENT_DISPLACEMENT,
} from '../src/tissue-field.js';

// Local capture assets stay private and out of the repository. Run against a
// reconstructed head with FACE_IMPACT_CAPTURE=/path/to/capture node --test ... .
const capture = process.env.FACE_IMPACT_CAPTURE;
test(
  'captured head keeps deepening at one cheek without reversing skin triangles',
  {
    skip:
      !capture && 'Set FACE_IMPACT_CAPTURE to a local reconstructed capture directory',
  },
  () => {
    const data = JSON.parse(fs.readFileSync(path.join(capture, 'mesh.json')));
    const cage = JSON.parse(fs.readFileSync(path.join(capture, 'physics-cage.json')));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(data.positions, 3),
    );
    geometry.setIndex(data.indices);
    geometry.computeVertexNormals();
    const dynamics = new FaceDynamics(geometry);
    dynamics.impactRig.setAnchors(cage.rigAnchors);
    dynamics.setHeadMode('clay');
    const tissue = dynamics.impactRig.tissue;
    const contact = tissue.nearest(cage.rigAnchors[50]);
    const direction = tissue.vertices[contact.node].n.map((v) => -v);
    let previousDepth = 0,
      previousPeak = 0;
    for (let hit = 0; hit < 4; hit++) {
      const positions = geometry.attributes.position.array;
      assert.ok(
        dynamics.applyImpact({
          location: Array.from(
            positions.slice(contact.index * 3, contact.index * 3 + 3),
          ),
          direction,
          magnitude: DEFAULT_IMPACT_MAGNITUDE,
        }) > 0,
      );
      for (let frame = 0; frame < 24; frame++) dynamics.step(1 / 120);
      const depth = direction.reduce(
        (sum, v, j) => sum + v * dynamics.impactRig.permanent[contact.index * 3 + j],
        0,
      );
      assert.ok(
        depth > previousDepth + 0.001,
        `Hit ${hit + 1} must deepen the same spot: ${previousDepth} -> ${depth}`,
      );
      assert.ok(
        dynamics.maxDisplacement > previousPeak + 0.001,
        `Hit ${hit + 1} must not undo the existing dent: ${previousPeak} -> ${dynamics.maxDisplacement}`,
      );
      assert.ok(dynamics.maxDisplacement <= MAX_PERMANENT_DISPLACEMENT + 1e-5);
      const quality = tissue.measure(positions);
      assert.equal(quality.finite, true);
      assert.equal(quality.reversedTriangles, 0, JSON.stringify(quality));
      previousDepth = depth;
      previousPeak = dynamics.maxDisplacement;
    }
    const deformed = geometry.attributes.position.array.slice();
    dynamics.setHeadMode('live');
    dynamics.step(0.2);
    assert.deepEqual(geometry.attributes.position.array, deformed);
    dynamics.reset();
    assert.deepEqual(geometry.attributes.position.array, dynamics.original);
  },
);
