// Arena tool schemas + handlers (OMNI.md §5.3).
//
// Registered only when the shared-engine Arena flag is on. When off, the
// existing sponsor Cornerman path remains the sole model integration and
// Arena behaves bit-for-bit as it did before.

export function registerArenaTools({registry, avatar, ui}) {
  registry.register('react_to_hit', {
    description: "React physically to a landed punch. Location and severity come from the classifier's strike event.",
    parameters: {
      type: 'object',
      properties: {
        location: {type: 'string', description: 'head, jaw, cheek-left, cheek-right, nose, body-left, body-right'},
        severity: {type: 'number', description: '1–3'},
      },
      required: ['location', 'severity'],
    },
    handler: (args) => {
      avatar?.reactToHit?.(args);
      ui?.log?.('event', `react_to_hit ${args.location} sev=${args.severity}`);
      return {applied: true};
    },
  });

  registry.register('set_expression', {
    description: 'Set the fighter face.',
    parameters: {
      type: 'object',
      properties: {emotion: {type: 'string', enum: ['smug', 'stunned', 'focused', 'winded', 'defiant']}, intensity: {type: 'number'}},
      required: ['emotion'],
    },
    handler: (args) => {
      avatar?.setExpression?.(args.emotion, args.intensity ?? 0.6);
      ui?.log?.('event', `set_expression ${args.emotion}`);
      return {applied: true};
    },
  });

  registry.register('taunt_gesture', {
    description: 'Play a taunt gesture the crowd can read.',
    parameters: {
      type: 'object',
      properties: {type: {type: 'string', enum: ['chin-up', 'come-on', 'shrug']}},
      required: ['type'],
    },
    handler: (args) => {
      avatar?.playTaunt?.(args.type);
      ui?.log?.('event', `taunt_gesture ${args.type}`);
      return {applied: true};
    },
  });

  registry.register('coach_callout', {
    description: "Call out a flaw the model saw — dropped guard, squared stance, telegraphed cross.",
    parameters: {
      type: 'object',
      properties: {flaw: {type: 'string'}, urgency: {type: 'number'}},
      required: ['flaw'],
    },
    handler: (args) => {
      avatar?.callout?.(args.flaw, args.urgency ?? 0.5);
      ui?.log?.('event', `coach_callout ${args.flaw}`);
      return {applied: true};
    },
  });

  registry.register('tap_out', {
    description: 'The avatar has had enough; ends the round with a tap-out.',
    parameters: {type: 'object', properties: {reason: {type: 'string'}}},
    handler: (args) => {
      avatar?.tapOut?.(args.reason);
      ui?.log?.('event', `tap_out ${args.reason || ''}`);
      return {ended: true};
    },
  });
}

export function arenaPersona({opponentName = 'The Sparring Partner'} = {}) {
  return [
    `You are ${opponentName}, a boxing sparring partner played by an AI in an AR ring.`,
    'You see the user through their webcam; you hear their voice; you can be interrupted.',
    'Stay in character. Trash-talk sparingly. Coach concretely.',
    '',
    'RULES:',
    '1. Physical events (`[EVENT] strike …`) come from the local hit classifier. Use the region and force provided.',
    '2. When hit, call `react_to_hit` with the reported location and a severity 1–3.',
    '3. Watch for dropped guard and telegraphed punches; call `coach_callout` with a short flaw.',
    '4. If the user says "hold" or "stop", stop sparring immediately.',
    '5. Speak in short bursts. Keep momentum with the physical loop.',
  ].join('\n');
}
