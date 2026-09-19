import test from 'node:test';
import assert from 'node:assert/strict';
import {EventBus, formatEvent} from '../src/omni/events.js';

test('EventBus adds t stamp and preserves order', () => {
  const bus = new EventBus();
  const seen = [];
  bus.addEventListener('event', e => seen.push(e.detail));
  bus.emit({type: 'strike', region: 'cheek-left', force: 55});
  bus.emit({type: 'press', region: 'cheek-right', pressure: 0.5});
  assert.equal(seen.length, 2);
  assert.equal(seen[0].type, 'strike');
  assert.ok(seen[0].t >= 0);
  assert.equal(seen[1].type, 'press');
});

test('formatEvent produces one-line context for each type', () => {
  assert.equal(formatEvent({type: 'press', region: 'cheek-right', pressure: 0.42}), '[EVENT] press cheek-right pressure=0.42');
  assert.equal(formatEvent({type: 'release', region: 'cheek-right', speed: 0.71, rebound: true}),
    '[EVENT] release cheek-right speed=0.71 rebound=true');
  assert.equal(formatEvent({type: 'strike', region: 'cheek-left', force: 55}), '[EVENT] strike cheek-left force=55');
  assert.equal(formatEvent({type: 'guard', state: true, region: 'head'}), '[EVENT] guard on region=head');
  assert.equal(formatEvent({type: 'consent', stage: 'scan', granted: true}), '[EVENT] consent scan granted');
  assert.equal(formatEvent({type: 'user.text', text: 'How is my guard?'}), '[USER] How is my guard?');
});

test('EventBus filters history since a timestamp', () => {
  const bus = new EventBus();
  bus.emit({type: 'strike', region: 'nose'});
  const marker = performance.now();
  bus.emit({type: 'press', region: 'cheek-right'});
  const recent = bus.history({since: marker});
  assert.equal(recent.length, 1);
  assert.equal(recent[0].type, 'press');
});
