import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { FaceDynamics } from '../src/physics.js';
import { createDemoPunch, takeDemoContact } from '../src/demo-punch.js';
import { DEFAULT_IMPACT_MAGNITUDE } from '../src/tissue-field.js';

function fixture() {
  const geometry = new THREE.SphereGeometry(1, 32, 24);
  geometry.scale(0.085, 0.135, 0.095);
  geometry.computeVertexNormals();
  const dynamics = new FaceDynamics(geometry);
  dynamics.setHeadMode('clay');
  return dynamics;
}

test('each button punch lands once, even when a slow frame skips contact', () => {
  for (const [side, type] of [
    [-1, 'hook'],
    [1, 'hook'],
    [1, 'uppercut'],
  ]) {
    const dynamics = fixture();
    const punch = createDemoPunch(dynamics, {
      side,
      type,
      magnitude: DEFAULT_IMPACT_MAGNITUDE,
    });
    assert.equal(takeDemoContact(dynamics, punch, 0.3), null);
    const contact = takeDemoContact(dynamics, punch, 1.4);
    assert.equal(contact.magnitude, DEFAULT_IMPACT_MAGNITUDE);
    assert.ok(dynamics.applyImpact(contact) > 0);
    for (let frame = 0; frame < 20; frame++) dynamics.step(1 / 120);
    assert.ok(dynamics.maxDisplacement > 0.01, `${side} ${type}`);
    assert.equal(takeDemoContact(dynamics, punch, 1.5), null);
    assert.equal(takeDemoContact(dynamics, punch, 0.52), null);
    if (type === 'uppercut') assert.ok(contact.direction[1] > 0.8);
    else assert.ok(contact.direction[0] * side < -0.7);
  }
});

test('repeated button punches track the same deforming skin instead of the original anchor', () => {
  const dynamics = fixture();
  const input = { side: -1, type: 'hook', magnitude: DEFAULT_IMPACT_MAGNITUDE };
  const first = createDemoPunch(dynamics, input);
  dynamics.applyImpact(takeDemoContact(dynamics, first, 0.52));
  for (let frame = 0; frame < 24; frame++) dynamics.step(1 / 120);
  const depth = dynamics.maxDisplacement;
  const second = createDemoPunch(dynamics, input);
  assert.equal(second.vertex, first.vertex);
  const contact = takeDemoContact(dynamics, second, 0.52);
  assert.deepEqual(
    contact.location,
    Array.from(
      dynamics.geometry.attributes.position.array.slice(
        second.vertex * 3,
        second.vertex * 3 + 3,
      ),
    ),
  );
  assert.notDeepEqual(
    contact.location,
    Array.from(dynamics.rest.slice(second.vertex * 3, second.vertex * 3 + 3)),
  );
  dynamics.applyImpact(contact);
  for (let frame = 0; frame < 24; frame++) dynamics.step(1 / 120);
  assert.ok(dynamics.maxDisplacement > depth + 0.001);
});
