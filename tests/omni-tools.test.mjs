import test from 'node:test';
import assert from 'node:assert/strict';
import { ToolRegistry, validate } from '../src/omni/tools.js';

test('validate rejects missing required args', () => {
  const schema = {
    type: 'object',
    properties: { region: { type: 'string' }, pain: { type: 'number' } },
    required: ['region'],
  };
  assert.equal(validate(schema, { pain: 3 })._error !== undefined, true);
});

test('validate rejects unknown enum values', () => {
  const schema = {
    type: 'object',
    properties: { phase: { type: 'string', enum: ['press', 'release'] } },
  };
  assert.equal(validate(schema, { phase: 'wiggle' })._error !== undefined, true);
});

test('validate coerces numeric strings to numbers', () => {
  const schema = { type: 'object', properties: { n: { type: 'number' } } };
  const out = validate(schema, { n: '3.14' });
  assert.strictEqual(out.n, 3.14);
});

test('ToolRegistry dispatches to a handler and emits tool.applied', async () => {
  const applied = [];
  const bus = { emit: (e) => applied.push(e), addEventListener: () => {} };
  const registry = new ToolRegistry({ bus });
  registry.register('do_thing', {
    description: 'no-op',
    parameters: {
      type: 'object',
      properties: { region: { type: 'string' } },
      required: ['region'],
    },
    handler: (args) => ({ echoed: args.region }),
  });
  const result = await registry.dispatch({
    name: 'do_thing',
    arguments: { region: 'RLQ' },
  });
  assert.deepEqual(result, { echoed: 'RLQ' });
  const applyEvent = applied.find((e) => e.type === 'tool.applied');
  assert.ok(applyEvent);
  assert.equal(applyEvent.tool, 'do_thing');
  assert.deepEqual(applyEvent.args, { region: 'RLQ' });
});

test('ToolRegistry surfaces unknown tools via onError', async () => {
  const errors = [];
  const registry = new ToolRegistry({ onError: (e) => errors.push(e.message) });
  await registry.dispatch({ name: 'nope', arguments: {} });
  assert.ok(errors[0].includes('unknown tool'));
});

test('schemas() returns entries in registered order', () => {
  const registry = new ToolRegistry();
  registry.register('a', { description: 'a', parameters: {}, handler: () => 1 });
  registry.register('b', { description: 'b', parameters: {}, handler: () => 2 });
  const names = registry.schemas().map((s) => s.name);
  assert.deepEqual(names, ['a', 'b']);
});
