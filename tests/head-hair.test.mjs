import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { HeadHair, remapHairRoots } from '../src/head-hair.js';

function fixture(type = 'wavy') {
  const g = new THREE.BufferGeometry();
  g.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(
      [-0.02, 0.13, -0.01, 0.02, 0.13, -0.01, 0, 0.13, -0.04],
      3,
    ),
  );
  g.setIndex([0, 1, 2]);
  g.computeVertexNormals();
  const spec = {
    seed: 12,
    hairlineY: 0.1,
    rootTriangles: Array.from({ length: 12 }, () => [0, 1, 2]).flat(),
    rootWeights: Array.from({ length: 12 }, () => [0.3, 0.3, 0.4]).flat(),
    parameters: {
      type,
      colorSrgb: [0.08, 0.06, 0.04],
      lengthMm: 50,
      sideLengthMm: 10,
      density: 1,
      curlTightness: 0.7,
      curlRadiusMm: 4,
      frizz: 0.3,
      waveLengthMm: 14,
      rootLiftMm: 3,
    },
  };
  return { g, spec };
}

test('hair types change finite 3D geometry and saved controls reproduce it exactly', () => {
  const shapes = [];
  for (const type of ['straight', 'wavy', 'curly', 'coily', 'braided', 'locs']) {
    const { g, spec } = fixture(type),
      hair = new HeadHair(g, spec),
      copy = new HeadHair(g, JSON.parse(JSON.stringify(hair.spec)));
    const points = hair.geometry.attributes.position.array;
    assert.ok(points.every(Number.isFinite));
    assert.deepEqual(points, copy.geometry.attributes.position.array);
    shapes.push(points.slice());
    hair.rebuild(g, { lengthMm: 95, rootLiftMm: 8 });
    assert.notDeepEqual(hair.geometry.attributes.position.array, points);
    hair.dispose();
    copy.dispose();
    g.dispose();
  }
  assert.notDeepEqual(shapes[0], shapes[1]);
  assert.notDeepEqual(shapes[1], shapes[2]);
});
test('scalp-bound hair translates and rotates with its surface, then returns to rest', () => {
  const { g, spec } = fixture(),
    hair = new HeadHair(g, spec),
    before = hair.geometry.attributes.position.array.slice();
  g.translate(0.015, -0.004, 0.002);
  hair.updateSurface(g);
  const after = hair.geometry.attributes.position.array;
  for (let i = 0; i < after.length; i++)
    assert.ok(Math.abs(after[i] - before[i] - [0.015, -0.004, 0.002][i % 3]) < 1e-7);
  g.rotateZ(0.15);
  g.computeVertexNormals();
  hair.updateSurface(g);
  assert.ok(after.every(Number.isFinite));
  hair.dispose();
  g.dispose();
});
test('GLB vertex reordering rebinds hair without attaching to the wrong face vertices', () => {
  const { g, spec } = fixture(),
    old = g.attributes.position.array,
    next = new Float32Array([...old.slice(6), ...old.slice(0, 6)]);
  const mapped = remapHairRoots(spec, old, next);
  assert.deepEqual(mapped.rootTriangles.slice(0, 3), [1, 2, 0]);
  assert.deepEqual(spec.rootTriangles.slice(0, 3), [0, 1, 2]);
  assert.throws(() => remapHairRoots(spec, old, new Float32Array(9)), /do not match/);
});
test('malformed imported root weights cannot corrupt the renderer', () => {
  const { g, spec } = fixture();
  spec.rootWeights[0] = NaN;
  assert.throws(() => new HeadHair(g, spec), /Invalid/);
});
test('binary GLB retains independent strand geometry, material and editable specification', async () => {
  const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js');
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const previous = globalThis.FileReader;
  globalThis.FileReader = class {
    readAsArrayBuffer(blob) {
      blob.arrayBuffer().then((result) => {
        this.result = result;
        this.onloadend?.();
      });
    }
  };
  try {
    const { g, spec } = fixture('curly'),
      hair = new HeadHair(g, spec);
    hair.userData.groom = hair.spec;
    const binary = await new GLTFExporter().parseAsync(hair, { binary: true });
    const parsed = await new GLTFLoader().parseAsync(binary, '');
    const restored = parsed.scene.children[0];
    assert.equal(restored.userData.accessory, 'hair');
    assert.equal(restored.userData.groom.parameters.type, 'curly');
    assert.deepEqual(
      restored.geometry.attributes.position.array,
      hair.geometry.attributes.position.array,
    );
    assert.equal(restored.material.isMeshPhysicalMaterial, true);
    hair.dispose();
    g.dispose();
  } finally {
    globalThis.FileReader = previous;
  }
});
test('photo-guided detail stays close to captured locks and skips unseen roots without losing bindings', () => {
  const { g, spec } = fixture();
  spec.mode = 'photo-detail';
  spec.photoGuides = {
    directions: Array.from({ length: 12 }, () => [1, 0, 0]).flat(),
    colors: Array.from({ length: 12 }, () => [0.12, 0.08, 0.05]).flat(),
    confidence: Array.from({ length: 12 }, (_, i) => (i % 2 ? 0.8 : 0)),
  };
  const hair = new HeadHair(g, spec);
  assert.equal(hair.strandCount, 6);
  assert.equal(hair.material.isMeshBasicMaterial, true);
  const root = new THREE.Vector3(
    -0.02 * 0.3 + 0.02 * 0.3,
    0.13,
    -0.01 * 0.6 - 0.04 * 0.4,
  );
  const points = hair.geometry.attributes.position.array;
  for (let i = 0; i < points.length; i += 3)
    assert.ok(new THREE.Vector3().fromArray(points, i).distanceTo(root) < 0.004);
  const before = points.slice();
  g.translate(0.01, 0, 0);
  hair.updateSurface(g);
  for (let i = 0; i < points.length; i++)
    assert.ok(Math.abs(points[i] - before[i] - (i % 3 === 0 ? 0.01 : 0)) < 1e-7);
  const saved = new HeadHair(g, JSON.parse(JSON.stringify(hair.spec)));
  assert.equal(saved.material.isMeshBasicMaterial, true);
  hair.dispose();
  saved.dispose();
  g.dispose();
});

function tracedFixture() {
  const { g, spec } = fixture();
  spec.version = 3;
  spec.mode = 'photo-strands';
  const segments = 18,
    offsets = [],
    normals = [];
  for (let i = 0; i < 12; i++)
    for (let j = 0; j <= segments; j++) {
      const t = j / segments;
      offsets.push((t - 0.5) * 0.036, 0, Math.sin(t * Math.PI * 2) * 0.0018);
      normals.push(0, 1, 0);
    }
  spec.photoGuides = {
    directions: Array.from({ length: 12 }, () => [1, 0, 0]).flat(),
    colors: Array.from({ length: 12 }, () => [0.14, 0.1, 0.07]).flat(),
    confidence: Array(12).fill(0.6),
    curveOffsets: offsets,
    curveNormals: normals,
    segments,
    referenceLengthMm: 50,
  };
  return { g, spec };
}

test('captured locks have continuous centimeter-scale waves, physical shading and relief', () => {
  const { g, spec } = tracedFixture(),
    hair = new HeadHair(g, spec),
    position = hair.geometry.attributes.position.array;
  assert.equal(hair.strandCount, 36);
  assert.ok(hair.material.isMeshPhysicalMaterial);
  assert.ok(hair.material.anisotropy > 0.5);
  hair.geometry.computeBoundingBox();
  const size = hair.geometry.boundingBox.getSize(new THREE.Vector3());
  assert.ok(size.x > 0.034);
  assert.ok(size.y > 0.0004);
  assert.ok(size.y < 0.006);
  assert.ok(size.z > 0.003);
  assert.ok(position.every(Number.isFinite));
  assert.ok(hair.geometry.attributes.normal.array.every(Number.isFinite));
  const copy = new HeadHair(g, JSON.parse(JSON.stringify(hair.spec)));
  assert.deepEqual(copy.geometry.attributes.position.array, position);
  const saved = position.slice();
  g.translate(0.01, 0.003, -0.002);
  hair.updateSurface(g);
  for (let i = 0; i < position.length; i++)
    assert.ok(Math.abs(position[i] - saved[i] - [0.01, 0.003, -0.002][i % 3]) < 1e-7);
  hair.rebuild(g, { frizz: 0.9, rootLiftMm: 10 });
  assert.notDeepEqual(hair.geometry.attributes.position.array, position);
  hair.dispose();
  copy.dispose();
  g.dispose();
});
test('traced locks reject damaged session curves before allocating geometry', () => {
  const { g, spec } = tracedFixture();
  spec.photoGuides.curveOffsets[1] = Infinity;
  assert.throws(() => new HeadHair(g, spec), /Invalid photographic hair curves/);
  spec.photoGuides.curveOffsets[1] = 0;
  spec.photoGuides.segments = 100000;
  assert.throws(() => new HeadHair(g, spec), /segments/);
  g.dispose();
});
test('photographic curve stations follow local edits and retain sampled colors', () => {
  const { g, spec } = tracedFixture();
  const p = [...g.attributes.position.array, 0.04, 0.13, -0.04];
  g.setAttribute('position', new THREE.Float32BufferAttribute(p, 3));
  g.setIndex([0, 1, 2, 1, 3, 2]);
  g.deleteAttribute('normal');
  g.computeVertexNormals();
  const guides = spec.photoGuides;
  guides.observedOnly = true;
  spec.rootCount = 12;
  guides.curveColors = guides.curveOffsets.map((_, i) => [0.09, 0.06, 0.04][i % 3]);
  guides.curveBindings = { triangles: [], weights: [] };
  for (let i = 0; i < 12; i++)
    for (let j = 0; j <= guides.segments; j++) {
      guides.curveBindings.triangles.push(...(j < 9 ? [0, 1, 2] : [3, 1, 2]));
      guides.curveBindings.weights.push(1, 0, 0);
    }
  const hair = new HeadHair(g, spec);
  assert.ok(hair.material.isMeshBasicMaterial);
  const before = hair.geometry.attributes.position.array.slice();
  g.attributes.position.array[10] += 0.004;
  hair.updateSurface(g);
  const after = hair.geometry.attributes.position.array;
  assert.equal(after[1], before[1]);
  assert.ok(Math.abs(after[18 * 9 + 1] - before[18 * 9 + 1] - 0.004) < 1e-6);
  const copy = new HeadHair(g, JSON.parse(JSON.stringify(hair.spec)));
  assert.ok(copy.curveBinding);
  copy.dispose();
  hair.dispose();
  g.dispose();
});
test('exported traced fibers keep anisotropy and editable source paths through GLB', async () => {
  const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js');
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const old = globalThis.FileReader;
  globalThis.FileReader = class {
    readAsArrayBuffer(blob) {
      blob.arrayBuffer().then((result) => {
        this.result = result;
        this.onloadend?.();
      });
    }
  };
  try {
    const { g, spec } = tracedFixture(),
      hair = new HeadHair(g, spec);
    hair.userData.groom = hair.spec;
    const binary = await new GLTFExporter().parseAsync(hair, { binary: true }),
      parsed = await new GLTFLoader().parseAsync(binary, '');
    const restored = parsed.scene.children[0];
    assert.deepEqual(
      restored.userData.groom.photoGuides.curveOffsets,
      spec.photoGuides.curveOffsets,
    );
    assert.equal(restored.material.anisotropy, 0.8);
    assert.deepEqual(
      restored.geometry.attributes.position.array,
      hair.geometry.attributes.position.array,
    );
    hair.dispose();
    g.dispose();
  } finally {
    globalThis.FileReader = old;
  }
});
