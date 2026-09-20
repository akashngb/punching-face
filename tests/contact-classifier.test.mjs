import test from 'node:test';
import assert from 'node:assert/strict';
import {ContactClassifier, arenaRegions, forceFromSpeed} from '../src/contact/classifier.js';

test('arenaRegions maps points to zones', () => {
  assert.equal(arenaRegions({x: -0.05, y: 0, z: 0}), 'cheek-left');
  assert.equal(arenaRegions({x: 0.05, y: 0, z: 0}), 'cheek-right');
  assert.equal(arenaRegions({x: 0, y: -0.08, z: 0}), 'jaw');
  assert.equal(arenaRegions({x: 0, y: 0, z: 0}), 'nose');
  assert.equal(arenaRegions({x: 0, y: 0, z: 0.2}), null); // out of face
});

test('strike is emitted with region and force, and debounced', () => {
  const events = [];
  const clf = new ContactClassifier({regionFromPoint: arenaRegions, onEvent: e => events.push(e)});
  clf.strike({point: {x: -0.05, y: 0, z: 0}, speed: 2.0});
  clf.strike({point: {x: -0.05, y: 0, z: 0}, speed: 2.5});  // same region, within 80ms → dropped
  assert.equal(events.length, 1);
  assert.equal(events[0].type, 'strike');
  assert.equal(events[0].region, 'cheek-left');
  assert.equal(events[0].force, forceFromSpeed(2));
  clf.dispose();
});

test('press emits after dwell, release follows with rebound flag', async () => {
  // Sustained-contact path is scenario-agnostic; exercise it against arenaRegions.
  const events = [];
  const clf = new ContactClassifier({regionFromPoint: arenaRegions, onEvent: e => events.push(e)});
  const point = {x: -0.05, y: 0, z: 0};  // cheek-left
  for (let i = 0; i < 8; i++) {
    clf.observe({point, velocity: 0.05, pressure: 0.4});
    await new Promise(r => setTimeout(r, 30));
  }
  clf.releaseAt({speed: 0.8});
  const kinds = events.map(e => e.type);
  assert.ok(kinds.includes('press'), 'press event emitted');
  assert.ok(kinds.includes('release'), 'release event emitted');
  const release = events.find(e => e.type === 'release');
  assert.equal(release.region, 'cheek-left');
  assert.equal(release.rebound, true);
  clf.dispose();
});
