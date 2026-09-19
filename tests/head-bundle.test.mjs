import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { loadHeadBundle } from '../src/head-bundle.js';

const hash = (value) => createHash('sha256').update(value).digest('hex');
function bundle(generation, marker) {
  const positions = Array(468 * 3).fill(marker);
  const texture = Buffer.from('color-' + marker);
  const files = {
    'mesh.json': JSON.stringify({ positions }),
    'physics-cage.json': JSON.stringify({ positions }),
    'physics-binding.json': JSON.stringify({ marker }),
    'texture-atlas.json': JSON.stringify({
      textureSha256: hash(texture),
      positionsSha256: hash(Buffer.from(new Float32Array(positions).buffer)),
      roughnessTexture: 'present',
    }),
    'appearance.png': texture,
    'appearance-roughness.png': Buffer.from('rough-' + marker),
  };
  return {
    files,
    release: {
      generation,
      files: Object.fromEntries(
        Object.entries(files).map(([name, value]) => [name, hash(value)]),
      ),
    },
  };
}

test('a load pins all files to A while a newer B becomes current', async () => {
  const a = bundle('a'.repeat(32), 1),
    b = bundle('b'.repeat(32), 2);
  let current = a;
  const seen = [];
  const fetcher = async (url) => {
    const params = new URL(url, 'http://localhost').searchParams;
    if (params.get('asset') === 'model-release.json') {
      const response = Response.json(current.release);
      current = b;
      return response;
    }
    seen.push(params.get('generation'));
    const pinned = params.get('generation') === a.release.generation ? a : current;
    return new Response(pinned.files[params.get('asset')]);
  };
  const loaded = await loadHeadBundle('scan', fetcher);
  assert.equal(loaded.generation, a.release.generation);
  assert.equal(loaded.data.positions[0], 1);
  assert.equal(loaded.binding.marker, 1);
  assert.equal(Buffer.from(loaded.roughnessBytes).toString(), 'rough-1');
  assert.deepEqual(new Set(seen), new Set([a.release.generation]));
});

test('corrupted roughness is rejected before the viewer receives a bundle', async () => {
  const a = bundle('a'.repeat(32), 1);
  const fetcher = async (url) => {
    const name = new URL(url, 'http://localhost').searchParams.get('asset');
    return name === 'model-release.json'
      ? Response.json(a.release)
      : new Response(name === 'appearance-roughness.png' ? 'broken' : a.files[name]);
  };
  await assert.rejects(
    loadHeadBundle('scan', fetcher),
    /failed verification.*roughness/,
  );
});

test('a legacy cage from a different head cannot be paired with a valid texture', async () => {
  const a = bundle('legacy', 1);
  a.files['physics-cage.json'] = JSON.stringify({ positions: Array(468 * 3).fill(2) });
  const fetcher = async (url) => {
    const name = new URL(url, 'http://localhost').searchParams.get('asset');
    return name === 'model-release.json'
      ? Response.json(a.release)
      : new Response(a.files[name]);
  };
  await assert.rejects(loadHeadBundle('scan', fetcher), /physics cage does not match/);
});
