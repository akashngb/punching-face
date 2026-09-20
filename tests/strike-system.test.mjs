import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { HeadCollider, StrikeTracker } from '../src/strike-system.js';

function collider() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(
      [
        0,
        0,
        0.06, // face
        0,
        0.13,
        0, // crown
        -0.13,
        -0.15,
        0.01, // shoulder
        0.13,
        -0.15,
        0.01, // shoulder
        0,
        -0.18,
        0.01, // chest
      ],
      3,
    ),
  );
  const root = new THREE.Group(),
    mesh = new THREE.Mesh(geometry);
  root.add(mesh);
  root.updateMatrixWorld(true);
  return new HeadCollider(mesh, root, () => null);
}

test('face is the preferred target while shoulder and chest remain valid contacts', () => {
  const target = collider();
  assert.equal(
    target.sweep({ from: [0, 0, 0.2], to: [0, 0, 0], radius: 0.045 }).region,
    'face',
  );
  assert.equal(
    target.sweep({ from: [-0.13, -0.15, 0.2], to: [-0.13, -0.15, 0], radius: 0.045 })
      .region,
    'shoulder',
  );
  assert.equal(
    target.sweep({ from: [0, -0.18, 0.2], to: [0, -0.18, 0], radius: 0.045 }).region,
    'chest',
  );
});

test('each hand owns one strike lifecycle and stale reacquisition cannot punch', () => {
  const strikes = new StrikeTracker({ startSpeed: 0.4, minInward: 0.1 });
  const target = [0, 0, 0],
    hit = ({ to }) =>
      to[2] < 0.15
        ? { point: [0, 0, 0.06], direction: [0, 0, -1], region: 'face' }
        : null;
  const sample = (hand, z, time) =>
    strikes.update(
      {
        hand,
        position: [hand === 'left' ? -0.03 : 0.03, 0, z],
        target,
        timestamp: time,
        closed: 1,
      },
      hit,
    );
  assert.equal(sample('left', 0.22, 0), null);
  assert.ok(sample('left', 0.13, 50));
  assert.equal(sample('left', 0.05, 100), null, 'one extension emits one impact');
  assert.equal(sample('right', 0.22, 0), null);
  assert.ok(sample('right', 0.13, 50), 'hands do not share history or cooldown');

  const stale = new StrikeTracker();
  assert.equal(
    stale.update(
      { hand: 'left', position: [0, 0, 0.3], target, timestamp: 0, closed: 1 },
      hit,
    ),
    null,
  );
  assert.equal(
    stale.update(
      { hand: 'left', position: [0, 0, 0], target, timestamp: 500, closed: 1 },
      hit,
    ),
    null,
  );
});

test('a fist lost mid-extension sweeps its blind window once, and a guarded hand releases to nothing', () => {
  const strikes = new StrikeTracker({ startSpeed: 0.4, minInward: 0.1 });
  const target = [0, 0, 0],
    wall = ({ to }) =>
      to[2] < 0.15
        ? { point: [0, 0, 0.06], direction: [0, 0, -1], region: 'face' }
        : null;
  strikes.update(
    { hand: 'left', position: [0, 0, 0.4], target, timestamp: 0, closed: 1 },
    () => null,
  );
  strikes.update(
    { hand: 'left', position: [0, 0, 0.3], target, timestamp: 50, closed: 1 },
    () => null,
  );
  // Occlusion at full punch: the hand vanishes while EXTENDING; the carried velocity crosses the face.
  const event = strikes.release('left', 170, wall);
  assert.equal(event?.region, 'face');
  assert.equal(event.blind, true);
  assert.ok(event.speed > 0.4, 'impact keeps the flight speed');
  assert.equal(
    strikes.release('left', 200, wall),
    null,
    'a released lifecycle cannot fire twice',
  );

  const idle = new StrikeTracker();
  idle.update(
    { hand: 'right', position: [0, 0, 0.4], target, timestamp: 0, closed: 1 },
    () => null,
  );
  idle.update(
    { hand: 'right', position: [0, 0, 0.398], target, timestamp: 50, closed: 1 },
    () => null,
  );
  assert.equal(
    idle.release('right', 120, wall),
    null,
    'a hand lost in guard is not a punch',
  );
});
