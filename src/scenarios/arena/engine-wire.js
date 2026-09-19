// Arena on the shared engine (OMNI.md §5.4, cut-order: safe by default).
//
// This module is imported from `src/sponsors/boot.js` behind a flag. When
// `ARENA_OMNI` is off (default), nothing loads and Arena remains the
// Cornerman-only experience it is today. When on, the shared OmniSession +
// tool dispatcher replace the SSE turn call; the classifier converts landed
// contacts into `strike` events; the model reacts through Arena's tools.
//
// Every interaction still passes through `window.__punchingFace.remotePunch`
// and the existing physics rig — this file adds tools around it, never
// replaces the physical hit path.

import {OmniSession} from '../../omni/session.js';
import {FallbackSession} from '../../omni/fallback.js';
import {EventBus, formatEvent} from '../../omni/events.js';
import {ToolRegistry} from '../../omni/tools.js';
import {ContactClassifier, arenaRegions} from '../../contact/classifier.js';
import {play as playReaction, keyForEvent, warmUp, ARENA_KEYS} from '../../omni/reactions.js';
import {registerArenaTools, personaFor} from './tools.js';

const FLAG_KEY = 'contact-sponsors-arena-omni';

function readFlag() {
  try {
    const raw = localStorage.getItem(FLAG_KEY);
    if (raw === '1' || raw === 'true') return true;
  } catch { /* no localStorage */ }
  // A URL override wins so the flag can be flipped without touching the DOM.
  try { return new URL(window.location.href).searchParams.get('arena_omni') === '1'; } catch { return false; }
}

export function enableArenaOmniIfFlagged({stats}) {
  if (!readFlag()) return null;
  return startArenaOmni({stats});
}

/**
 * Standalone entry, exported so tests / QA can construct the wire with
 * synthetic sources and no `window` global. The default path uses `window.__punchingFace`.
 */
export function startArenaOmni({stats, punchingFace = null, uiLog = null} = {}) {
  const lab = punchingFace || (typeof window !== 'undefined' ? window.__punchingFace : null);
  const bus = new EventBus();
  let session;
  try { session = new OmniSession(); }
  catch { session = new FallbackSession(); }

  const avatar = {
    reactToHit: ({location, severity}) => {
      // Nudge the local head with a mild recoil — the physical hit already
      // landed via the classifier, this is just a visible ack.
      uiLog?.('event', `avatar recoiled ${location}/${severity}`);
    },
    setExpression: (emotion, intensity) => uiLog?.('event', `expression ${emotion} ${intensity}`),
    playTaunt: type => uiLog?.('event', `taunt ${type}`),
    callout: (flaw, urgency) => uiLog?.('event', `callout ${flaw} u=${urgency}`),
    tapOut: reason => uiLog?.('event', `tap_out ${reason || ''}`),
  };
  const registry = new ToolRegistry({bus, session, onError: err => uiLog?.('warn', err.message)});
  registerArenaTools({registry, avatar, ui: {log: uiLog}});

  const classifier = new ContactClassifier({
    regionFromPoint: arenaRegions,
    onEvent: event => {
      bus.emit(event);
      session.sendEngineEvent(event, formatEvent(event));
      const key = keyForEvent(event);
      if (key) playReaction(key);
      if (event.type === 'strike') session.requestResponse(mode === 'coach'
        ? 'They landed that one. One short corrective cue, and coach_callout if you saw a flaw.'
        : 'You just got hit. Call react_to_hit, then one short line in character.');
    },
  });

  // Every landed contact main.js broadcasts on window.__punchingFace.remotePunch
  // → we tap the same signal by polling window.__lastContact (same source the
  // sponsor stats reader uses; consistent with the existing hook contract).
  let lastSeenAt = 0;
  const poll = setInterval(() => {
    const contact = typeof window !== 'undefined' ? window.__lastContact : null;
    if (!contact || contact.time === lastSeenAt) return;
    lastSeenAt = contact.time;
    classifier.strike({
      point: {x: contact.point?.[0] ?? 0, y: contact.point?.[1] ?? 0, z: contact.point?.[2] ?? 0},
      speed: contact.speed,
      mode: contact.mode || 'hook',
    });
  }, 33);

  // Same selector the sponsor panel writes, so one choice drives both paths.
  let mode = 'face';
  try { mode = localStorage.getItem('punching-face-sponsors-mode') || 'face'; } catch { /* no localStorage */ }
  session.configure({persona: personaFor(mode), tools: registry.schemas()});
  session.addEventListener('tool.call', event => registry.dispatch(event.detail));
  session.connect().catch(err => uiLog?.('warn', 'arena omni: ' + err.message));

  warmUp(ARENA_KEYS);

  return {
    dispose() { clearInterval(poll); session.disconnect(); classifier.dispose(); },
    get session() { return session; },
    get bus() { return bus; },
    get classifier() { return classifier; },
    setFlag(on) { try { localStorage.setItem(FLAG_KEY, on ? '1' : '0'); } catch { /* no-op */ } },
  };
}
