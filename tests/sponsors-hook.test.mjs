import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Two tiny hooks connect the sponsor features to the app. Both live in files that are edited often,
// by more than one person and more than one coding agent, so their presence is asserted here.
// If this fails, a hook was dropped during an edit: restore it, do not delete this test.
// What they are for: SPONSOR_SETUP.md.
test('index.html still loads the sponsor dock after main.js', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.match(
    html,
    /src="\/src\/main\.js"\s*>\s*<\/script>\s*<script\s+type="module"\s+src="\/src\/sponsors\/boot\.js"\s*>/,
  );
});

test('main.js still exposes remotePunch, the only way a LiveKit guest can land a hit', () => {
  const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  const remotePunch = /window\.__punchingFace\.remotePunch\s*=/;
  assert.match(
    main,
    remotePunch,
    'the arena hook at the end of src/main.js is missing',
  );
  // It must go through the real contact() path and the real mesh, not a shortcut around the physics.
  const hook = main.slice(main.search(remotePunch));
  assert.match(hook, /raycaster\.intersectObject\(mesh/);
  assert.match(hook, /contact\(/);
  // Inputs are made finite and bounded inside the hook: a NaN speed once took the Newton session down.
  assert.match(hook, /Number\.isFinite/);
  assert.match(hook, /number\(\s*punch\.speed,\s*1\.2,\s*0,\s*4,?\s*\)/);
  const initialization = main.search(/window\.__punchingFace\s*=\s*\{/);
  assert.ok(
    initialization >= 0 && initialization < main.search(remotePunch),
    'the hook must come after __punchingFace is created',
  );
});

test('the Python services keep their optional Sentry hooks, so one click still reads as one trace', () => {
  const read = (path) => readFileSync(new URL('../' + path, import.meta.url), 'utf8');
  for (const [file, service] of [
    ['server.py', 'punching-face-api'],
    ['physics_server.py', 'physics'],
  ]) {
    const source = read(file);
    assert.match(
      source,
      new RegExp(
        `sponsor_obs\\.init\\('${service}'\\)[;\\s]+sponsor_obs\\.instrument_http\\(Handler\\)`,
      ),
      file + ' lost its tracing hook',
    );
    assert.match(
      source,
      /except ImportError:\s*pass/,
      file + ' must keep working if sponsor_obs.py is absent',
    );
  }
  const pipeline = read('face_pipeline.py');
  assert.match(pipeline, /from sponsor_obs import child_env as trace_env/);
  assert.match(
    pipeline,
    /subprocess\.Popen\([\s\S]*?\benv\s*=\s*trace_env\(\),?\s*\)/,
    'the pipeline subprocess no longer inherits the trace',
  );
  const build = read('scripts/build_photo_face.py');
  assert.match(build, /sponsor_obs\.patch_pipeline_timer\(\)/);
  assert.match(build, /with trace:\s*run\(/);
});
