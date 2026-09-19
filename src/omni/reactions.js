// Cached reaction audio (OMNI.md §3.2, §6.1).
//
// Reactions are pre-generated short clips (100–400 ms) keyed by
// `<scenario>.<intent>.<intensity>`, e.g. `arena.grunt.high`. Contact fires
// one of them within 50 ms via WebAudio decode-and-play; the model's own line
// follows at ~1 s. The user sees an instant physical reaction.
//
// Two sources:
//   1. Baked-in synthetic tones (small, always available). These are enough
//      for dev/CI and act as a graceful default.
//   2. Static files under `/omni-reactions/<key>.wav`, if present. Ship
//      real ElevenLabs-generated voice clips for the demo.
//
// A single AudioContext is shared. play() returns quickly; decoding runs
// once per key and is cached.

// The face is the one taking the punch, so these are all reactions to being hit
// — a ladder of force, plus a few things it does between exchanges.
const SYNTH = {
  'arena.grunt.low': { tone: 190, ms: 120, fadeMs: 40, shape: 'grunt' },
  'arena.grunt.mid': { tone: 220, ms: 180, fadeMs: 50, shape: 'grunt' },
  'arena.grunt.high': { tone: 260, ms: 260, fadeMs: 60, shape: 'exhale' },
  'arena.tap': { tone: 340, ms: 90, fadeMs: 30, shape: 'grunt' },
  'arena.scoff': { tone: 300, ms: 110, fadeMs: 35, shape: 'exhale' },
  'arena.laugh': { tone: 210, ms: 150, fadeMs: 45, shape: 'moan' },
  'arena.wheeze': { tone: 170, ms: 380, fadeMs: 90, shape: 'exhale' },
};

/** Every key the arena scenario can fire; warm these up at scene load. */
export const ARENA_KEYS = Object.keys(SYNTH);

const cache = new Map();
let ctx = null;

function _getCtx() {
  if (ctx) return ctx;
  if (typeof window === 'undefined') return null;
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  return ctx;
}

async function _load(key) {
  const audioCtx = _getCtx();
  if (!audioCtx) return null;
  if (cache.has(key)) return cache.get(key);
  // Try a real file first (production case).
  try {
    const response = await fetch(`/omni-reactions/${key}.wav`, {
      cache: 'force-cache',
    });
    if (response.ok) {
      const arrayBuffer = await response.arrayBuffer();
      const buffer = await audioCtx.decodeAudioData(arrayBuffer);
      cache.set(key, buffer);
      return buffer;
    }
  } catch {
    /* fall through to synth */
  }
  const spec = SYNTH[key];
  if (!spec) return null;
  const buffer = _synthesize(audioCtx, spec);
  cache.set(key, buffer);
  return buffer;
}

function _synthesize(audioCtx, { tone, ms, fadeMs, shape }) {
  const rate = audioCtx.sampleRate;
  const samples = Math.round((rate * ms) / 1000);
  const buffer = audioCtx.createBuffer(1, samples, rate);
  const data = buffer.getChannelData(0);
  const fadeSamples = Math.round((rate * fadeMs) / 1000);
  for (let i = 0; i < samples; i++) {
    const t = i / rate;
    // A little vibrato + a soft attack. Not a real voice, but reads as a
    // human-ish grunt for a demo without shipping audio assets.
    const wobble = 1 + 0.03 * Math.sin(2 * Math.PI * 5.5 * t);
    let s = Math.sin(2 * Math.PI * tone * wobble * t);
    if (shape === 'grunt') s = Math.sign(s) * Math.pow(Math.abs(s), 0.7);
    else if (shape === 'moan')
      s =
        Math.sin(2 * Math.PI * tone * t) * 0.6 +
        Math.sin(2 * Math.PI * tone * 1.5 * t) * 0.4;
    else if (shape === 'cry')
      s = Math.sin(2 * Math.PI * (tone + (40 * t) / (ms / 1000)) * t);
    else if (shape === 'exhale') s = (Math.random() * 2 - 1) * 0.7 + s * 0.3;
    // envelope
    let env = 1;
    if (i < fadeSamples) env = i / fadeSamples;
    else if (i > samples - fadeSamples) env = (samples - i) / fadeSamples;
    data[i] = 0.4 * env * s;
  }
  return buffer;
}

/** Play the reaction. Returns the AudioBufferSourceNode so callers can cancel. */
export async function play(key, { volume = 1.0 } = {}) {
  const audioCtx = _getCtx();
  if (!audioCtx) return null;
  if (audioCtx.state === 'suspended') {
    try {
      await audioCtx.resume();
    } catch {
      /* ignore */
    }
  }
  const buffer = await _load(key);
  if (!buffer) return null;
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  const gain = audioCtx.createGain();
  gain.gain.value = volume;
  source.connect(gain).connect(audioCtx.destination);
  source.start();
  return source;
}

/**
 * Convenience: pick a reaction key from an engine event. Scenarios may pass
 * their own key builder; this is the default.
 */
export function keyForEvent(event) {
  if (!event) return null;
  if (event.type === 'strike') {
    const bucket = event.force > 60 ? 'high' : event.force > 30 ? 'mid' : 'low';
    return `arena.grunt.${bucket}`;
  }
  // A press is contact without a punch behind it — the face registers it and
  // nothing more. A big hit ends with the face getting its breath back.
  if (event.type === 'press') return 'arena.tap';
  if (event.type === 'release' && event.rebound) return 'arena.wheeze';
  return null;
}

/** Warm up decoding at scene load so the very first hit doesn't pay the cost. */
export async function warmUp(keys) {
  await Promise.all(keys.map((k) => _load(k)));
}
