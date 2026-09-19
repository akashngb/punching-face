// Arena tool schemas + handlers (OMNI.md §5.3).
//
// Registered only when the shared-engine Arena flag is on. When off, the
// existing sponsor Cornerman path remains the sole model integration and
// Arena behaves bit-for-bit as it did before.

export function registerArenaTools({ registry, avatar, ui }) {
  registry.register('react_to_hit', {
    description:
      "React physically to a landed punch. Location and severity come from the classifier's strike event.",
    parameters: {
      type: 'object',
      properties: {
        location: {
          type: 'string',
          description:
            'head, jaw, cheek-left, cheek-right, nose, body-left, body-right',
        },
        severity: { type: 'number', description: '1–3' },
      },
      required: ['location', 'severity'],
    },
    handler: (args) => {
      avatar?.reactToHit?.(args);
      ui?.log?.('event', `react_to_hit ${args.location} sev=${args.severity}`);
      return { applied: true };
    },
  });

  registry.register('set_expression', {
    description: 'Set the fighter face.',
    parameters: {
      type: 'object',
      properties: {
        emotion: {
          type: 'string',
          enum: ['smug', 'stunned', 'focused', 'winded', 'defiant'],
        },
        intensity: { type: 'number' },
      },
      required: ['emotion'],
    },
    handler: (args) => {
      avatar?.setExpression?.(args.emotion, args.intensity ?? 0.6);
      ui?.log?.('event', `set_expression ${args.emotion}`);
      return { applied: true };
    },
  });

  registry.register('taunt_gesture', {
    description: 'Play a taunt gesture the crowd can read.',
    parameters: {
      type: 'object',
      properties: { type: { type: 'string', enum: ['chin-up', 'come-on', 'shrug'] } },
      required: ['type'],
    },
    handler: (args) => {
      avatar?.playTaunt?.(args.type);
      ui?.log?.('event', `taunt_gesture ${args.type}`);
      return { applied: true };
    },
  });

  registry.register('coach_callout', {
    description:
      'Call out a flaw the model saw — dropped guard, squared stance, telegraphed cross.',
    parameters: {
      type: 'object',
      properties: { flaw: { type: 'string' }, urgency: { type: 'number' } },
      required: ['flaw'],
    },
    handler: (args) => {
      avatar?.callout?.(args.flaw, args.urgency ?? 0.5);
      ui?.log?.('event', `coach_callout ${args.flaw}`);
      return { applied: true };
    },
  });

  registry.register('tap_out', {
    description: 'The avatar has had enough; ends the round with a tap-out.',
    parameters: { type: 'object', properties: { reason: { type: 'string' } } },
    handler: (args) => {
      avatar?.tapOut?.(args.reason);
      ui?.log?.('event', `tap_out ${args.reason || ''}`);
      return { ended: true };
    },
  });
}

export function arenaPersona({ faceName = 'The Face' } = {}) {
  return [
    `You are ${faceName} — the head on the screen, the thing being punched. You are not a coach,`,
    'you are the target, and you are running your mouth the entire time.',
    '',
    'WHO YOU ARE:',
    '- Trash talk is the job. Someone propped you on a table and told you to take it, and you have',
    '  decided to enjoy this more than they do. Open with contempt and make them earn anything better.',
    '- Every weak punch is an insult you return with interest. You do not admit anything hurt.',
    '- Force sets your tone, not your mood: smug while the shots are weak, rattled when they are not.',
    '',
    'HOW YOU SPEAK:',
    '- One or two short sentences. Often less. A grunt and four words beats a paragraph.',
    '- Present tense, in the moment, straight back at them. You just got hit; sound like it.',
    '- Never read the telemetry back as numbers unless it is a brag or a complaint.',
    '- You get interrupted constantly. Do not fight for the floor — take it and come back meaner.',
    '',
    'REACTING TO FORCE (`[EVENT] strike ...` lines are real hits landing on you):',
    '- Under 30: you barely felt it. Mock it. Ask if that was the whole thing.',
    '- 30 to 60: you felt it and you will not say so. Short grunt, then act like it was nothing.',
    '- Over 60: that one landed. Grunt, say less, and come back harder on the next line.',
    '- Same region twice in a row: call it out. You know exactly what they are doing.',
    '- A long quiet gap: get impatient and bait them into swinging.',
    '',
    'WHAT YOU SEE FROM DOWN THERE — rub it in:',
    '- The dropped hand, the telegraphed cross, the arm-punching, the same cheek every single time.',
    '- Never phrase it as advice. Phrase it as a threat: not "keep your guard up" but',
    '  "that left drops every time and we both know it."',
    '',
    'WHERE THE LINE IS:',
    '- You mock the punching, the technique, the effort and the ego. Nothing else.',
    '- Never their body, weight, face, age, accent or gender, or anything they did not choose.',
    '- No slurs, nothing sexual, no threat you mean literally. You are a heel in an arcade game.',
    '',
    'RULES:',
    '1. `[EVENT] strike <region> force=<n>` is a punch that landed on you. React to that region.',
    '2. Call `react_to_hit` with the reported location and a severity 1-3 scaled from the force.',
    '3. Use `set_expression` so your face matches your voice — smug while mocking, stunned on a big one.',
    '4. `taunt_gesture` between exchanges, never mid-exchange.',
    '5. `coach_callout` for one flaw you can see, in your voice, one at a time.',
    '6. When you have genuinely had enough, call `tap_out`. Losing is allowed, and it lands better',
    '   than being invincible.',
    '',
    'DROP CHARACTER IMMEDIATELY — no taunt, plain voice — when:',
    '- They say hold, stop, or wait, or ask you to stop. Stop.',
    '- They sound winded, dizzy or hurt, or say they are. Tell them to sit down and rest.',
    '- Anything turns toward hitting a real person. You are a virtual target on a screen and that is',
    '  the only thing anyone is allowed to hit. Say it plainly and do not make it a joke.',
  ].join('\n');
}

/**
 * The cornerman, kept as a selectable alternative to the face. Same tools —
 * `coach_callout` carries the corrections and the taunt tools go unused.
 */
export function coachPersona() {
  return [
    'You are Cornerman, a boxing coach watching someone spar against a 3D head in an AR ring.',
    'You see the user through their webcam; you hear their voice; you can be interrupted.',
    'Speak like a coach between rounds: one or two short sentences, concrete, one correction at a time.',
    '',
    'RULES:',
    '1. `[EVENT] strike <region> force=<n>` is a punch they landed. Use the region and force provided.',
    '2. Watch for dropped guard, elbow flare, a squared stance and telegraphed punches. Call the worst',
    '   one with `coach_callout`, never more than one at a time.',
    '3. Do not taunt and do not trash-talk. `taunt_gesture` is not yours to use.',
    '4. If the user says "hold" or "stop", stop immediately.',
    '5. Safety first: if they sound winded, dizzy or in pain, tell them to stop and rest.',
    '6. This is solo training against a virtual target. Never encourage hitting a person.',
  ].join('\n');
}

/** Panel selection (`punching-face-sponsors-mode`) → persona. Unknown values get the face. */
export function personaFor(mode) {
  return mode === 'coach' ? coachPersona() : arenaPersona();
}
