import test from 'node:test';
import assert from 'node:assert/strict';
import {
  downsample,
  encodeWav,
  bytesToBase64,
  base64ToBytes,
  pcm16ToFloat32,
  VoiceGate,
} from '../src/sponsors/audio.js';
import { zoneOf, RoundStats } from '../src/sponsors/telemetry.js';
import { PunchDetector } from '../src/sponsors/punch-detect.js';
import {
  EventStream,
  sanitizePunch,
  encode,
  decode,
  RateLimit,
} from '../src/sponsors/sse.js';

test('microphone audio becomes a valid 16 kHz mono PCM16 WAV and survives the round trip', () => {
  const tone = Float32Array.from(
    { length: 4800 },
    (_, i) => Math.sin((i / 48000) * 2 * Math.PI * 440) * 0.5,
  );
  const low = downsample(tone, 48000, 16000);
  assert.equal(low.length, 1600);
  const wav = encodeWav(low, 16000),
    view = new DataView(wav.buffer);
  assert.equal(String.fromCharCode(...wav.subarray(0, 4)), 'RIFF');
  assert.equal(String.fromCharCode(...wav.subarray(8, 12)), 'WAVE');
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 3200);
  assert.equal(wav.length, 44 + 3200);
  // A header on a streamed chunk is skipped; raw PCM is read from byte zero.
  const back = pcm16ToFloat32(base64ToBytes(bytesToBase64(wav)));
  assert.equal(back.length, 1600);
  assert.ok(Math.max(...low.map((v, i) => Math.abs(v - back[i]))) < 1 / 16000);
  assert.equal(pcm16ToFloat32(wav.subarray(44)).length, 1600);
  assert.deepEqual(
    [...pcm16ToFloat32(encodeWav(Float32Array.of(2, -2), 16000))].map(
      (v) => +v.toFixed(3),
    ),
    [1, -1],
  );
});

test('voice gate opens on speech, ignores a punch-length thud, and closes after a pause', () => {
  const gate = new VoiceGate(),
    events = [];
  const run = (level, ms) => {
    for (let t = 0; t < ms; t += 20) {
      const e = gate.push(level, 20);
      if (e) events.push(e);
    }
  };
  run(0.003, 600);
  assert.deepEqual(events, []);
  run(0.2, 80);
  run(0.003, 900);
  assert.deepEqual(events, [], 'an 80 ms impact thud must not start a turn');
  run(0.08, 1200);
  run(0.003, 800);
  assert.deepEqual(events, ['start', 'end']);
  // A hall that is loud from the first frame is learned as the floor; it never opens the gate.
  const hall = new VoiceGate();
  let opened = 0;
  for (let t = 0; t < 4000; t += 20)
    if (hall.push(0.03 + 0.004 * Math.sin(t), 20) === 'start') opened++;
  assert.equal(opened, 0);
  // ...yet someone speaking over that hall still gets a turn.
  const over = [];
  for (let t = 0; t < 900; t += 20) {
    const e = hall.push(0.18, 20);
    if (e) over.push(e);
  }
  for (let t = 0; t < 900; t += 20) {
    const e = hall.push(0.03, 20);
    if (e) over.push(e);
  }
  assert.deepEqual(over, ['start', 'end']);
  // Syllable-modulated speech longer than the cap is sent once, and the floor is left alone.
  const long = new VoiceGate({ maxMs: 1000 }),
    seen = [];
  for (let t = 0; t < 500; t += 20) long.push(0.003, 20);
  for (let t = 0; t < 1300; t += 20) {
    const e = long.push(Math.floor(t / 100) % 2 ? 0.02 : 0.12, 20);
    if (e) seen.push(e);
  }
  assert.deepEqual(
    seen.slice(0, 2),
    ['start', 'end'],
    'a turn is capped so one request never grows without bound',
  );
  assert.ok(long.noise < 0.03);
  // Music that starts mid-session costs one false turn, then becomes the floor: no endless loop.
  const drone = new VoiceGate({ maxMs: 2000 });
  let starts = 0;
  for (let t = 0; t < 500; t += 20) drone.push(0.003, 20);
  for (let t = 0; t < 30000; t += 20) if (drone.push(0.05, 20) === 'start') starts++;
  assert.equal(starts, 1);
});

test("contacts are zoned from the puncher's side and a combo or personal best is detected once", () => {
  assert.equal(zoneOf([-0.05, 0, 0.04]), 'cheek-L');
  assert.equal(zoneOf([0.05, 0, 0.04]), 'cheek-R');
  assert.equal(zoneOf([0, 0, 0.07]), 'nose');
  assert.equal(zoneOf([0, 0.07, 0.05]), 'forehead');
  assert.equal(zoneOf([0.01, -0.045, 0.05]), 'mouth');
  assert.equal(zoneOf([0, -0.09, 0.03]), 'chin');
  assert.equal(zoneOf([0.05, -0.07, 0.02]), 'jaw-R');
  const stats = new RoundStats();
  const hit = (time, speed, x = -0.05, id = 'a', name = 'Ana') =>
    stats.add({ id, name, speed, point: [x, 0, 0.04], time });
  assert.deepEqual(hit(0, 1), []);
  assert.deepEqual(hit(400, 1.1), []);
  assert.deepEqual(hit(800, 1.2), ['combo']);
  assert.deepEqual(
    hit(1100, 1.2),
    [],
    'a fourth punch does not re-announce the same combo',
  );
  assert.deepEqual(hit(9000, 2.4), ['personal-best']);
  hit(9500, 1.5, 0.05, 'b', 'Ben');
  const snap = stats.snapshot(10000, 'combo');
  assert.equal(snap.trigger, 'combo');
  assert.deepEqual(
    snap.participants.map((p) => p.name),
    ['Ana', 'Ben'],
  );
  const ana = snap.participants[0];
  assert.equal(ana.count, 5);
  assert.equal(ana.max, 2.4);
  assert.equal(ana.left, 5);
  assert.equal(ana.zones['cheek-L'], 5);
  assert.equal(snap.last.name, 'Ben');
  // The coach only hears about the last 30 s; the scoreboard keeps the whole session.
  assert.deepEqual(stats.snapshot(60000).participants, []);
  assert.deepEqual(
    stats.scoreboard().map((s) => [s.name, s.count]),
    [
      ['Ana', 5],
      ['Ben', 1],
    ],
  );
  assert.ok(
    !JSON.stringify(snap).includes('point'),
    'no geometry or media leaves in telemetry',
  );
});

// A closed fist: fingertips curled back beside the knuckles, as fistScore expects.
function fist(cx, cy, size) {
  const p = (x, y) => ({ x: cx + (x * size) / (16 / 9), y: cy + y * size, z: 0 }),
    lm = Array.from({ length: 21 }, () => p(0, 0));
  lm[0] = p(0, 0.5);
  [5, 9, 13, 17].forEach((k, i) => {
    const x = -0.3 + i * 0.2;
    lm[k] = p(x, -0.5);
    lm[k + 1] = p(x, -0.75);
    lm[k + 2] = p(x, -0.55);
    lm[k + 3] = p(x, -0.3);
  });
  lm[9] = p(0, -0.5);
  [1, 2, 3, 4].forEach((k, i) => (lm[k] = p(-0.45, 0.3 - i * 0.15)));
  return lm;
}

const open = (cx, cy, size) =>
  fist(cx, cy, size).map((q, i) =>
    [8, 12, 16, 20].includes(i) ? { ...q, y: q.y - size * 0.9 } : q,
  );

test('a straight punch is a closed fist rushing the camera; waving, open hands and repeats are ignored', () => {
  const detector = new PunchDetector();
  let event = null,
    time = 0;
  for (const size of [0.1, 0.1, 0.1, 0.125, 0.16, 0.2]) {
    event = detector.update('h', fist(0.3, 0.45, size), (time += 33)) || event;
  }
  assert.ok(event, 'a fist growing 2x in 100 ms is a punch');
  assert.equal(event.kind, 'straight');
  assert.equal(event.side, 'right');
  assert.ok(
    event.u > 0,
    "the thrower's right hand lands on the viewer-right of a head facing them",
  );
  assert.ok(event.speed >= 0.9 && event.speed <= 4);
  assert.equal(
    detector.update('h', fist(0.3, 0.45, 0.26), (time += 33)),
    null,
    'cooldown: one punch, one hit',
  );
  const slow = new PunchDetector();
  let drift = null;
  for (let i = 0; i < 30; i++)
    drift =
      slow.update('h', fist(0.3 + i * 0.002, 0.45, 0.1 + i * 0.0005), i * 33) || drift;
  assert.equal(drift, null, 'slow drift toward the camera is not a punch');
  const palm = new PunchDetector();
  let wave = null;
  [0.1, 0.1, 0.13, 0.17, 0.21].forEach(
    (s, i) => (wave = palm.update('h', open(0.5, 0.5, s), i * 33) || wave),
  );
  assert.equal(wave, null, 'an open hand never punches');
  const gap = new PunchDetector();
  gap.update('h', fist(0.5, 0.5, 0.1), 0);
  assert.equal(
    gap.update('h', fist(0.5, 0.5, 0.3), 900),
    null,
    'tracking dropouts do not become teleport punches',
  );
});

test('a hook is fast sideways travel and lands on the side it came from, moving across', () => {
  const detector = new PunchDetector();
  let event = null;
  [0.8, 0.8, 0.72, 0.62, 0.52].forEach(
    (x, i) => (event = detector.update('h', fist(x, 0.45, 0.12), i * 33) || event),
  );
  assert.ok(event);
  assert.equal(event.kind, 'hook');
  assert.equal(event.side, 'left');
  assert.ok(
    event.u < 0 && event.lateral > 0,
    'a left hook lands viewer-left and travels right',
  );
});

test('guest messages are clamped, rate limited and tolerant of garbage before touching physics', () => {
  assert.deepEqual(
    sanitizePunch({
      type: 'punch',
      u: 99,
      v: -99,
      lateral: 'x',
      speed: 1e9,
      side: '<img>',
      kind: 1,
    }),
    { u: 1, v: -1, lateral: 0, speed: 4, side: 'right', kind: 'straight' },
  );
  assert.equal(sanitizePunch({ type: 'chat' }), null);
  assert.equal(sanitizePunch(null), null);
  assert.equal(sanitizePunch({ type: 'punch', u: NaN, speed: -5 }).speed, 0.4);
  assert.deepEqual(decode(encode({ type: 'punch', u: 0.5 })), {
    type: 'punch',
    u: 0.5,
  });
  assert.equal(decode(new Uint8Array([255, 0, 1])), null);
  assert.equal(decode(encode('text')), null);
  const limit = new RateLimit(6);
  let passed = 0;
  for (let t = 0; t < 1000; t += 10) if (limit.allow('g', t)) passed++;
  assert.equal(passed, 6);
  assert.ok(limit.allow('other', 500), 'one spammer does not block another guest');
  assert.ok(limit.allow('g', 2100));
});

test('the coach stream parses across arbitrary chunk boundaries and survives a bad event', () => {
  const stream = new EventStream(),
    wire =
      'event: meta\ndata: {"mock":true}\n\nevent: text\ndata: {"delta":"Hands "}\n\nevent: text\ndata: {oops\n\nevent: done\ndata: {"ms":12}\n\n';
  const events = [];
  for (let i = 0; i < wire.length; i += 7)
    events.push(...stream.feed(wire.slice(i, i + 7)));
  assert.deepEqual(
    events.map((e) => e.event),
    ['meta', 'text', 'done'],
  );
  assert.equal(events[1].data.delta, 'Hands ');
  assert.equal(stream.buffer, '');
});
