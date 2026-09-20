import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker as ThreadWorker } from 'node:worker_threads';
import * as THREE from 'three';
import { ImpactPreparation } from '../src/impact-preparation.js';
import { FaceImpactRig } from '../src/impact-rig.js';
import { FaceDynamics } from '../src/physics.js';
import { impactCacheKey, restContact } from '../src/impact-cache.js';
const emptyContact = { location: [0, 0, 0], direction: [0, 0, -1], magnitude: 0.85 };

function makeWorker() {
  const source = `import {parentPort} from 'node:worker_threads';
    globalThis.self={postMessage:(data,transfer)=>parentPort.postMessage(data,transfer)};
    await import(${JSON.stringify(new URL('../src/impact-worker.js', import.meta.url).href)});
    parentPort.on('message',data=>self.onmessage({data}));`;
  const worker = new ThreadWorker(
    new URL('data:text/javascript,' + encodeURIComponent(source)),
  );
  const adapter = {
    postMessage: (data) => worker.postMessage(data),
    terminate: () => worker.terminate(),
  };
  worker.on('message', (data) => adapter.onmessage?.({ data }));
  worker.on('error', (error) => adapter.onerror?.(error));
  return adapter;
}

function fixture() {
  const g = new THREE.SphereGeometry(0.1, 24, 20);
  const rig = new FaceImpactRig(g.attributes.position.array, undefined, {
    indices: g.index.array,
    normals: g.attributes.normal.array,
  });
  const hit = rig.tissue.nearest(rig.anchors[50]);
  const input = {
    location: rig.tissue.vertices[hit.node].p,
    direction: [0.75, 0, -0.6],
    magnitude: 0.85,
  };
  return { rig, input };
}

test('real worker returns the same live endpoints as synchronous preparation', async (t) => {
  const { rig, input } = fixture();
  const broker = new ImpactPreparation(rig.preparationModel(), { makeWorker });
  t.after(() => broker.dispose());
  await broker.ready;
  assert.equal(broker.failed, undefined);
  const result = await new Promise((resolve) =>
    broker.request(input, 0.75, true, (result) => {
      if (result.stage !== 'preview') resolve(result);
    }),
  );
  assert.equal(result.error, undefined);
  rig.impact(rig.tissue.rest, input, 0.75);
  for (const key of [
    'field',
    'plastic',
    'target',
    'reaction',
    'reactionTarget',
    'combinedTarget',
  ])
    assert.deepEqual(result.event[key], rig.events[0][key], key);
  assert.equal(result.event.age, 0);
  assert.equal(result.affected, rig.lastImpact.affected);
  // A different job proves transferred buffers did not corrupt worker state.
  const nextInput = { ...input, magnitude: 0.8 };
  const second = await new Promise((resolve) =>
    broker.request(nextInput, 0.75, true, (result) => {
      if (result.stage !== 'preview') resolve(result);
    }),
  );
  rig.reset();
  rig.impact(rig.tissue.rest, nextInput, 0.75);
  assert.deepEqual(second.event.target, rig.events[0].target);
});

test('prewarmed camera contacts return full-quality poses immediately and isolate event ages', async (t) => {
  const { rig } = fixture();
  const broker = new ImpactPreparation(rig.preparationModel(), { makeWorker });
  t.after(() => broker.dispose());
  await broker.ready;
  const direction = [0.75, -0.05, -0.6];
  const length = Math.hypot(...direction);
  const input = {
    location: restContact(
      rig.tissue,
      rig.anchors[50],
      direction.map((v) => v / length),
    ),
    direction: direction.map((v) => v / length),
    magnitude: 0.9,
  };
  let first;
  broker.request(input, 0.75, true, (result) => (first = result));
  assert.ok(first?.cached, 'ready includes the exact camera contact');
  rig.impact(rig.tissue.rest, input, 0.75);
  assert.deepEqual(first.event.target, rig.events[0].target);
  first.event.age = 0.5;
  first.event.committed = 1;
  broker.invalidate();
  let second;
  broker.request(input, 0.75, true, (result) => (second = result));
  assert.ok(second?.cached, 'reset retains valid geometry cache');
  assert.notEqual(first.event, second.event);
  assert.equal(second.event.age, 0);
  assert.equal(second.event.committed, 0);
  assert.notEqual(
    impactCacheKey(input, 0.75, true),
    impactCacheKey({ ...input, magnitude: 0.89 }, 0.75, true),
  );
});

test('prewarmed contact selection matches the live Three.js front-face raycast', () => {
  const { rig } = fixture();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(rig.tissue.rest, 3));
  geometry.setIndex(rig.tissue.indices);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  for (const [anchor, incoming] of [
    [50, [0.75, -0.05, -0.6]],
    [280, [-0.75, -0.05, -0.6]],
    [152, [0, 0.85, -0.5]],
  ]) {
    const point = new THREE.Vector3(...rig.anchors[anchor]);
    const direction = new THREE.Vector3(...incoming).normalize();
    const ray = new THREE.Raycaster(
      point.clone().addScaledVector(direction, -0.12),
      direction,
      0,
      0.3,
    );
    const intersection = ray.intersectObject(mesh)[0]?.point ?? point;
    const expected =
      rig.tissue.vertices[rig.tissue.nearest(intersection.toArray()).node].p;
    assert.deepEqual(
      restContact(rig.tissue, point.toArray(), direction.toArray()),
      expected,
    );
  }
});

test('progressive poses retain their age, finish at the exact endpoint, and cannot revive after expiry', () => {
  const { rig, input } = fixture();
  rig.impact(rig.tissue.rest, input, 0.75, rig.tissue.rest, { refine: false });
  const preview = rig.events[0];
  rig.reset();
  rig.impact(rig.tissue.rest, input, 0.75);
  const complete = rig.events[0];
  rig.reset();
  let reply;
  rig.preparer = {
    request(_input, _softness, _reaction, callback) {
      reply = callback;
      return true;
    },
    invalidate() {
      reply = null;
    },
  };
  rig.impact(rig.tissue.rest, input, 0.75);
  reply({ stage: 'preview', event: { ...preview }, affected: 1 });
  rig.step(0.06);
  reply({ stage: 'complete', event: { ...complete }, affected: 1 });
  assert.equal(rig.events[0].age, 0.06);
  assert.ok(rig.events[0].refinement);
  rig.step(0.04);
  assert.equal(rig.events[0].refinement, null);
  assert.deepEqual(rig.events[0].target, complete.target);
  rig.events[0].refinement = { from: preview, elapsed: 0 };
  rig.step(0);
  assert.equal(rig.events[0].refinement, null, 'held peak adopts the finished pose');
  rig.reset();
  rig.impact(rig.tissue.rest, input, 0.75);
  reply({ stage: 'preview', event: { ...preview }, affected: 1 });
  rig.step(2);
  reply({ stage: 'complete', event: { ...complete }, affected: 1 });
  assert.equal(rig.events.length, 0);
});

test('reset, reconfiguration, and disposal ignore outdated worker replies and bound the queue', async () => {
  const sent = [],
    replies = [],
    fake = { postMessage: (m) => sent.push(m), terminate() {} };
  const broker = new ImpactPreparation({}, { makeWorker: () => fake });
  for (let i = 0; i < 4; i++)
    assert.equal(
      broker.request(emptyContact, 0.75, true, (r) => replies.push(r)),
      true,
    );
  assert.equal(
    broker.request(emptyContact, 0.75, true, () => {}),
    false,
  );
  const old = sent.at(-1);
  broker.invalidate();
  fake.onmessage({ data: { id: old.id, epoch: old.epoch, event: {} } });
  assert.equal(replies.length, 0);
  broker.configure({ rest: [] });
  broker.request(emptyContact, 0.75, true, (r) => replies.push(r));
  const current = sent.at(-1);
  fake.onmessage({ data: { id: current.id, epoch: current.epoch, event: {} } });
  assert.equal(replies.length, 1);
  broker.dispose();
  fake.onmessage({ data: { id: current.id, epoch: current.epoch, event: {} } });
  assert.equal(replies.length, 1);
});

test('an unavailable worker fails queued jobs once and permits synchronous fallback', async () => {
  const fake = { postMessage() {}, terminate() {} };
  const broker = new ImpactPreparation({}, { makeWorker: () => fake });
  let failures = 0;
  broker.request(emptyContact, 0.75, true, (result) => {
    assert.ok(result.error);
    failures++;
  });
  fake.onerror();
  fake.onerror();
  await broker.ready;
  assert.equal(failures, 1);
  assert.equal(broker.failed, true);
  assert.equal(
    broker.request(emptyContact, 0.75, true, () => {}),
    false,
  );
  broker.dispose();
});

test('cached manual poses match direct evaluation and invalidate on edits, undo, and reset', () => {
  const d = new FaceDynamics(new THREE.SphereGeometry(0.1, 12, 10));
  d.rig.smile = 0.37;
  const check = () => {
    const pose = d.poseOffsets();
    for (let i = 0; i < d.rest.length; i += 3)
      assert.deepEqual(
        Array.from(pose.slice(i, i + 3), (v) => v || 0),
        d.rigDelta(d.rest[i], d.rest[i + 1], d.rest[i + 2]).map((v) => v || 0),
      );
    assert.equal(d.poseOffsets(), pose);
  };
  check();
  d.rig.jaw = 0.61;
  check();
  d.remember();
  d.sculpt(new THREE.Vector3(0, 0, 0.1), 0.004);
  assert.equal(d.impactRig.asyncRestEdited, true);
  check();
  d.undo();
  check();
  d.redo();
  check();
  d.reset();
  check();
});
