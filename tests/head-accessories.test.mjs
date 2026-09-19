import test from 'node:test';
import assert from 'node:assert/strict';
import { HeadGlasses } from '../src/head-accessories.js';
import * as THREE from 'three';

function fixture() {
  const ring = (x) =>
    Array.from({ length: 16 }, (_, i) => [
      x + 0.024 * Math.cos((i * Math.PI) / 8),
      0.04 + 0.016 * Math.sin((i * Math.PI) / 8),
      0.024,
    ]);
  return {
    rims: [ring(-0.037), ring(0.037)],
    bridge: [
      [-0.013, 0.043, 0.025],
      [0, 0.048, 0.03],
      [0.013, 0.043, 0.025],
    ],
    temples: [
      [
        [0.06, 0.046, 0.023],
        [0.08, 0.046, -0.05],
        [0.08, 0.03, -0.1],
      ],
      [
        [-0.06, 0.046, 0.023],
        [-0.08, 0.046, -0.05],
        [-0.08, 0.03, -0.1],
      ],
    ],
    frameColor: [0.05, 0.05, 0.05],
    radius: 0.0015,
    lensTint: 0,
  };
}

test('glasses have independent three-dimensional geometry that round-trips through a saved specification', () => {
  const spec = fixture(),
    g = new HeadGlasses(spec);
  const restored = new HeadGlasses(JSON.parse(JSON.stringify(g.spec)));
  const bounds = new THREE.Box3().setFromObject(g);
  assert.ok(bounds.max.z - bounds.min.z > 0.1);
  assert.equal(g.children.length, 9);
  g.children.forEach((m, i) => {
    assert.ok(m.geometry.attributes.position.array.every(Number.isFinite));
    assert.deepEqual(
      m.geometry.attributes.position.array,
      restored.children[i].geometry.attributes.position.array,
    );
    assert.equal(m.userData.accessory, 'eyeglasses');
    assert.equal(m.geometry.morphAttributes.position, undefined);
  });
  spec.rims[0][0][0] = 0.8;
  assert.notEqual(g.spec.rims[0][0][0], 0.8);
  g.dispose();
  restored.dispose();
});
test('acetate arms are flatter than their height and taper toward the ear; clear lenses preserve the photographed eyes', () => {
  const spec = fixture();
  spec.templeWidth = 0.006;
  spec.temples = [
    [
      [0.07, 0.04, 0],
      [0.07, 0.04, -0.06],
      [0.07, 0.04, -0.12],
    ],
  ];
  const glasses = new HeadGlasses(spec),
    arm = glasses.getObjectByName('Eyeglass temple 1');
  const positions = arm.geometry.attributes.position;
  const section = (i) => {
    const box = new THREE.Box3();
    for (let j = 0; j < 16; j++)
      box.expandByPoint(new THREE.Vector3().fromBufferAttribute(positions, i * 16 + j));
    return box.getSize(new THREE.Vector3());
  };
  const start = section(0),
    end = section(96);
  assert.ok(start.y > start.x * 2);
  assert.ok(end.y < start.y * 0.6);
  const lens = glasses.getObjectByName('Eyeglass lens 1');
  assert.ok(lens.material.transparent);
  assert.ok(lens.material.opacity < 0.12);
  assert.equal(lens.material.depthWrite, false);
  assert.equal(lens.material.ior, 1.5);
  assert.ok(lens.geometry.attributes.position.count > 500);
  assert.ok(glasses.getObjectByName('Eyeglass hinge 1'));
  glasses.dispose();
});
test('invalid eyewear cannot install non-finite vertices into the renderer', () => {
  const spec = fixture();
  spec.rims[0][3][2] = Infinity;
  assert.throws(() => new HeadGlasses(spec), /Invalid/);
});
test('GLB round-trip retains acetate frames, clear lenses and separate hinges', async () => {
  const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js'),
    { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const previous = globalThis.FileReader;
  globalThis.FileReader = class {
    readAsArrayBuffer(blob) {
      blob.arrayBuffer().then((result) => {
        this.result = result;
        this.onloadend?.();
      });
    }
  };
  const glasses = new HeadGlasses(fixture());
  try {
    const binary = await new GLTFExporter().parseAsync(glasses, { binary: true }),
      restored = (await new GLTFLoader().parseAsync(binary, '')).scene.children[0];
    assert.equal(restored.children.length, 9);
    for (let i = 0; i < 9; i++) {
      assert.deepEqual(
        restored.children[i].geometry.attributes.position.array,
        glasses.children[i].geometry.attributes.position.array,
      );
      assert.equal(restored.children[i].userData.accessory, 'eyeglasses');
    }
    const lens = restored.children.find((m) => m.name.startsWith('Eyeglass_lens'));
    assert.ok(lens.material.transparent);
    assert.ok(lens.material.opacity < 0.12);
  } finally {
    glasses.dispose();
    globalThis.FileReader = previous;
  }
});
