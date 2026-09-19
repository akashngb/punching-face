import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// The Meshy engine lives in its own files (meshy_backend.py, src/meshy-engine.js) and reaches the app through a few
// one-line hooks in files that several people and coding agents edit at once. If this fails, a hook was dropped
// during an edit: restore it, do not delete this test. What the engine is for: README.md, "Reconstruction engines".
const read = (path) => readFileSync(new URL('../' + path, import.meta.url), 'utf8');

test('the head-scan dialog still defers build, poll and load to the engine choice', () => {
  const capture = read('src/face-capture.js');
  assert.match(capture, /import \{ EngineChoice \} from '\.\/meshy-engine\.js'/);
  assert.match(capture, /this\.engines = new EngineChoice\(this\)/);
  assert.match(capture, /this\.engines\.refresh\(\)/);
  assert.match(capture, /this\.count < this\.engines\.minimumViews/);
  for (const method of ['build', 'load'])
    assert.match(
      capture,
      new RegExp(
        `async ${method}\\(\\) \\{\\s*if \\(this\\.engines\\.meshy\\) return this\\.engines\\.${method}\\(\\);`,
      ),
      `${method}() no longer hands over to Meshy`,
    );
  assert.match(
    capture,
    /if \(this\.engines\.meshy\) return this\.engines\.poll\(id\);/,
  );
});

test('the scene still offers the GLB import path and status fields the Meshy loader relies on', () => {
  const main = read('src/main.js');
  // src/meshy-engine.js hands the Meshy GLB to the same input a user upload goes through, then waits for the busy
  // overlay and checks the model name. Renaming any of these silently breaks loading a Meshy head.
  assert.match(main, /\$\('face-file'\)\.onchange\s*=/);
  for (const id of [
    'face-file',
    'busy',
    'model-name',
    'scene-name',
    'model-kind',
    'physics-engine',
    'photo-count',
  ])
    assert.match(main, new RegExp(`id="${id}"`), `#${id} is gone from main.js`);
  assert.match(main, /window\.__labReady = true/);
});

test('server.py still routes the Meshy endpoints and shares one key lookup', () => {
  const server = read('server.py');
  assert.match(server, /^import meshy_backend$/m);
  assert.match(server, /MESHY = meshy_backend\.MeshyEngine\(FACE_STORE\)/);
  assert.equal(
    server.match(
      /if url\.path in meshy_backend\.ROUTES:\s+return MESHY\.handle\(self, url\)/g,
    )?.length,
    2,
    'both do_GET and do_POST must route meshy_backend.ROUTES',
  );
  assert.match(server, /key = meshy_backend\.api_key\(\)/);
  // The POST hook must sit behind the local-origin check, like every other state-changing route.
  assert.ok(
    server.indexOf('Only the local lab may request reconstruction.') <
      server.lastIndexOf('if url.path in meshy_backend.ROUTES:'),
  );
});
