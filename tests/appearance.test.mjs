import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from 'three';
import { SurfaceAppearance, weldTexturedSurface } from '../src/surface-appearance.js';
import { FaceDynamics } from '../src/physics.js';

function fixture() {
  const data = JSON.parse(
    fs.readFileSync(new URL('../public/online-face/mesh.json', import.meta.url)),
  );
  const atlas = JSON.parse(
    fs.readFileSync(
      new URL('../public/online-face/texture-atlas.json', import.meta.url),
    ),
  );
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3));
  g.setIndex(data.indices);
  g.computeVertexNormals();
  return { g, atlas };
}

test('UV seam copies remain coincident during impact and in exported morph targets', () => {
  const { g, atlas } = fixture(),
    d = new FaceDynamics(g),
    surface = new SurfaceAppearance(g, atlas, new THREE.Texture());
  d.impulse(
    new THREE.Vector3(-0.045, 0.005, 0.07),
    new THREE.Vector3(0.8, 0, -0.6),
    1.3,
  );
  for (let i = 0; i < 10; i++) d.step(1 / 120);
  assert.ok(d.maxDisplacement > 0.004);
  surface.updateSurface(g.attributes.position.array, g.attributes.normal.array);
  const live = surface.geometry.attributes.position.array.slice(),
    seen = new Map();
  let copies = 0;
  surface.mapping.forEach((source, i) => {
    const xyz = live.slice(i * 3, i * 3 + 3);
    if (seen.has(source)) {
      assert.deepEqual(xyz, seen.get(source));
      copies++;
    } else seen.set(source, xyz);
  });
  assert.ok(copies > 1000);
  const out = surface.exportGeometry(d.rest, d.exportMorphs());
  assert.deepEqual(surface.geometry.attributes.position.array, live);
  assert.equal(out.morphAttributes.position.length, 4);
  for (const attribute of [out.attributes.position, ...out.morphAttributes.position]) {
    seen.clear();
    surface.mapping.forEach((source, i) => {
      const xyz = attribute.array.slice(i * 3, i * 3 + 3);
      assert.ok(xyz.every(Number.isFinite));
      if (seen.has(source)) assert.deepEqual(xyz, seen.get(source));
      else seen.set(source, xyz);
    });
  }
  out.dispose();
  surface.dispose();
  g.dispose();
});
test('textured export re-import keeps triangle positions and UVs while welding simulation seams', () => {
  const { g, atlas } = fixture(),
    surface = new SurfaceAppearance(g, atlas, new THREE.Texture());
  const restored = weldTexturedSurface(surface.geometry),
    other = new SurfaceAppearance(
      restored.geometry,
      restored.atlas,
      new THREE.Texture(),
    );
  assert.ok(restored.geometry.attributes.position.count < surface.mapping.length);
  const before = surface.geometry.attributes.position.array,
    after = other.geometry.attributes.position.array;
  assert.equal(before.length, after.length);
  for (let i = 0; i < before.length; i++)
    assert.ok(Math.abs(before[i] - after[i]) <= 1e-7);
  assert.deepEqual(
    other.geometry.attributes.uv.array,
    surface.geometry.attributes.uv.array,
  );
  assert.deepEqual(other.geometry.index.array, surface.geometry.index.array);
  surface.dispose();
  other.dispose();
  g.dispose();
  restored.geometry.dispose();
});
test('an invalid atlas fails before reading beyond the source surface', () => {
  const { g, atlas } = fixture();
  atlas.mapping[0] = -1;
  assert.throws(
    () => new SurfaceAppearance(g, atlas, new THREE.Texture()),
    /does not match/,
  );
  g.dispose();
});
test('eye roughness uses a linear material map and is released with the surface', () => {
  const { g, atlas } = fixture();
  atlas.stats = { material: 'lit' };
  const roughness = new THREE.Texture();
  roughness.colorSpace = THREE.SRGBColorSpace;
  let disposed = false;
  roughness.addEventListener('dispose', () => (disposed = true));
  const surface = new SurfaceAppearance(g, atlas, new THREE.Texture(), roughness);
  assert.equal(surface.material.roughnessMap, roughness);
  assert.equal(surface.material.roughness, 1);
  assert.equal(roughness.colorSpace, THREE.NoColorSpace);
  surface.dispose();
  assert.ok(disposed);
  g.dispose();
});
