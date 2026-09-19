import test from 'node:test';
import assert from 'node:assert/strict';
import { enhanceSurface, featureProtection } from '../src/realism-surface.js';

function fixture(disconnected = false) {
  // Two UV islands meet along a welded edge, with a deliberately different
  // exposure on either side. A disconnected variant occupies the same space.
  const positions = [0, 0, 0, 0.01, 0, 0, 0, 1, 0, 0.01, 1, 0, 0.02, 0, 0, 0.02, 1, 0];
  const mapping = [0, 1, 2, 3, 1, 4, 3, 5];
  if (disconnected) {
    positions.push(0.01, 0, 0, 0.01, 1, 0);
    mapping[4] = 6;
    mapping[6] = 7;
  }
  const uv = [
    0.05, 0.1, 0.4, 0.1, 0.05, 0.9, 0.4, 0.9, 0.6, 0.1, 0.95, 0.1, 0.6, 0.9, 0.95, 0.9,
  ];
  const indices = [0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6];
  const width = 100,
    height = 100,
    pixels = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = x < width / 2 ? 120 : 165;
      pixels.set([v, v - 15, v - 30, 255], (y * width + x) * 4);
    }
  return { positions, mapping, uv, indices, width, height, pixels, flipY: false };
}
const sample = (data, x, y) => data[(y * 100 + x) * 4];

test('balances an exposure discontinuity across UV islands without changing geometry', () => {
  const data = fixture(),
    before = structuredClone(data);
  const { pixels, report } = enhanceSurface(data);
  const gapBefore = sample(data.pixels, 61, 50) - sample(data.pixels, 38, 50);
  const gapAfter = sample(pixels, 61, 50) - sample(pixels, 38, 50);
  assert.ok(Math.abs(gapAfter) < gapBefore * 0.3, `seam ${gapBefore} -> ${gapAfter}`);
  assert.deepEqual(data, before);
  assert.equal(report.geometryUnchanged, true);
  assert.ok(report.adjustedTexels > 0);
});

test('spatially coincident disconnected surfaces cannot donate colour', () => {
  const data = fixture(true),
    { pixels } = enhanceSurface(data);
  assert.deepEqual(pixels, data.pixels);
});

test('protected triangles and alpha are retained exactly', () => {
  const data = fixture();
  data.protectedVertices = [0, 1, 2, 3, 4, 5];
  assert.deepEqual(enhanceSurface(data).pixels, data.pixels);
  data.protectedVertices = [];
  data.pixels[50 * 100 * 4 + 20 * 4 + 3] = 0;
  const { pixels } = enhanceSurface(data);
  for (let i = 3; i < pixels.length; i += 4) assert.equal(pixels[i], data.pixels[i]);
  assert.equal(sample(pixels, 20, 50), sample(data.pixels, 20, 50));
});

test('texture orientation follows flipY for imported models', () => {
  const a = fixture();
  for (let y = 0; y < 100; y++)
    for (let x = 0; x < 100; x++) a.pixels[(y * 100 + x) * 4] += y % 7;
  const b = structuredClone(a);
  b.flipY = true;
  for (let y = 0; y < 100; y++)
    b.pixels.set(a.pixels.slice(y * 400, (y + 1) * 400), (99 - y) * 400);
  const outA = enhanceSurface(a).pixels,
    outB = enhanceSurface(b).pixels;
  for (let y = 10; y < 90; y++)
    for (const x of [10, 25, 38, 61, 75, 90])
      assert.ok(Math.abs(sample(outA, x, y) - sample(outB, x, 99 - y)) <= 1);
});

test('invalid UVs, positions and overlapping islands fail without mutating input', () => {
  const a = fixture();
  a.uv[0] = -1;
  assert.throws(() => enhanceSurface(a), /valid/);
  const b = fixture();
  b.positions[0] = NaN;
  assert.throws(() => enhanceSurface(b), /finite/);
  const c = fixture();
  c.uv.splice(8, 8, ...c.uv.slice(0, 8));
  assert.throws(() => enhanceSurface(c), /overlapping/);
});

test('facial feature protection preserves landmarks and fades continuously', () => {
  const landmarks = new Array(468 * 3).fill(0);
  landmarks[33 * 3] = -0.05;
  landmarks[263 * 3] = 0.05;
  landmarks[10 * 3 + 1] = 0.1;
  const positions = [0, 0, 0, 0.03, 0.025, 0, 0.09, -0.05, 0, 0, 0.11, 0];
  const weights = featureProtection(positions, landmarks);
  assert.equal(weights[0], 0);
  assert.equal(weights[2], 1);
  assert.equal(weights[3], 0);
  assert.ok(weights.every((v) => v >= 0 && v <= 1));
});
