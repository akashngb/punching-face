// Entry for the sponsor features. Loaded by its own <script> in index.html, after main.js, and
// talks to the app only through window.__punchingFace / window.__lastContact, so main.js stays as is.
// If the sponsor service is not running, the app is untouched apart from one small offline pill.
import './sponsors.css';
import { RoundStats } from './telemetry.js';
import { obs, initSentry, reportPhysics } from './sentry.js';
import { createCornerman } from './cornerman.js';
import { createArenaHost } from './arena-host.js';
// The shared-engine Arena is a NO-OP unless the `arena_omni` flag is set.
// Off = today's behaviour bit-for-bit (OMNI.md §5.3 cut order).
import { enableArenaOmniIfFlagged } from '../scenarios/arena/engine-wire.js';

const API = 'http://127.0.0.1:5176';
let config = null,
  seen = 0,
  demoFace = null,
  demoRequested = !!window.__punchingFaceDemoStarted;
const stats = new RoundStats();
const remember = (key, value) => {
  try {
    if (value === undefined)
      return localStorage.getItem('punching-face-sponsors-' + key);
    localStorage.setItem('punching-face-sponsors-' + key, value);
  } catch {
    return null;
  }
};

async function loadConfig() {
  const response = await fetch(API + '/sponsors/config');
  if (!response.ok) throw new Error('Sponsor service error.');
  config = await response.json();
  return config;
}

const dock = document.createElement('div');
dock.id = 'sponsor-dock';
dock.className = remember('open') === '1' ? '' : 'collapsed';
dock.innerHTML = `<button class="sd-pill"><i class="sd-dot" data-d="coach" title="OMNI face"></i><i class="sd-dot" data-d="arena" title="LiveKit arena"></i><i class="sd-dot" data-d="obs" title="Sentry"></i><span>The Face · Arena</span></button>
  <div class="sd-body"><div class="sd-tabs"><button data-tab="coach">The Face</button><button data-tab="arena">Arena</button></div>
  <section data-panel="coach"></section><section data-panel="arena"></section></div>`;
document.body.append(dock);
const pill = dock.querySelector('.sd-pill'),
  dots = Object.fromEntries(
    [...dock.querySelectorAll('[data-d]')].map((n) => [n.dataset.d, n]),
  );
pill.onclick = () => {
  dock.classList.toggle('collapsed');
  remember('open', dock.classList.contains('collapsed') ? '0' : '1');
};

function show(tab) {
  for (const b of dock.querySelectorAll('[data-tab]'))
    b.classList.toggle('active', b.dataset.tab === tab);
  for (const s of dock.querySelectorAll('[data-panel]'))
    s.classList.toggle('active', s.dataset.panel === tab);
  remember('tab', tab);
}

for (const b of dock.querySelectorAll('[data-tab]'))
  b.onclick = () => show(b.dataset.tab);

function startDemoFace() {
  demoRequested = true;
  dock.classList.remove('collapsed');
  show('coach');
  // Keep this call synchronous with Begin punching so WebAudio can use its gesture.
  void demoFace?.startFace();
}

window.addEventListener('punching-face-demo-start', startDemoFace);
if (demoRequested) startDemoFace();

function offline() {
  dock.querySelector('[data-panel=coach]').innerHTML =
    `<div class="sd-status">Sponsor services are not running, so the face and the arena are off. The rest of PUNCHING FACE is unaffected.</div>
    <div class="sd-status">Start them with <code>npm run sponsors</code>, then:</div><div class="sd-row"><button class="primary" data-k="retry" style="flex:1">Retry</button></div>`;
  dock.querySelector('[data-k=retry]').onclick = start;
  dots.coach.className = dots.arena.className = 'sd-dot warn';
  show('coach');
}

async function start() {
  try {
    await loadConfig();
  } catch {
    return offline();
  }
  if (config.sentry.dsn) {
    try {
      await initSentry(config.sentry);
      reportPhysics(() => window.__punchingFace?.state);
      obs.tag('app', 'punching-face-host');
    } catch (error) {
      console.warn('Sentry did not start:', error);
    }
  }
  dots.obs.className = 'sd-dot' + (obs.enabled ? ' on' : '');
  dots.obs.title = obs.enabled
    ? 'Sentry: tracing, logs and replay are on (webcam and 3D canvas are never recorded)'
    : 'Sentry: no DSN configured';
  const shared = { api: API, config: () => config, stats, refreshConfig: loadConfig };
  const tabButton = dock.querySelector('[data-tab=coach]'),
    pillLabel = pill.querySelector('span');
  const coach = createCornerman({
    ...shared,
    panel: dock.querySelector('[data-panel=coach]'),
    // The panel owns which voice is talking; the dock just mirrors the name.
    onMode: (name) => {
      tabButton.textContent = name;
      pillLabel.textContent = name + ' · Arena';
    },
  });
  demoFace = coach;
  // One place records a landed punch, whoever threw it: stats first, then the face and the room hear about it.
  const record = (contact, who) => {
    seen = contact.time;
    const triggers = stats.add({
      id: who.id,
      name: who.name,
      speed: contact.speed,
      point: contact.point,
      side: who.side,
      time: performance.now(),
    });
    const event = { ...stats.last };
    coach.onPunch(triggers);
    arena.onPunch(event);
    return event;
  };
  const arena = createArenaHost({
    ...shared,
    panel: dock.querySelector('[data-panel=arena]'),
    coach,
    record,
  });
  // main.js publishes every contact on window.__lastContact. Remote hits are recorded synchronously by the
  // arena (it knows who threw them); this poll picks up the host's own tracked and demo punches.
  setInterval(() => {
    const contact = window.__lastContact;
    if (!contact || contact.time === seen) return;
    record(
      contact,
      contact.source === 'remote'
        ? { id: 'guest', name: 'Guest' }
        : { id: 'host', name: arena.hostName },
    );
  }, 33);
  setInterval(() => {
    dots.arena.className =
      'sd-dot' + (arena.live ? ' live' : config.livekit.configured ? ' on' : '');
    dots.coach.className = 'sd-dot' + (config.omni.configured ? ' on' : ' warn');
  }, 1000);
  show(remember('tab') || 'coach');
  // Config may arrive after the user has already entered the demo.
  if (demoRequested || window.__punchingFaceDemoStarted) startDemoFace();
  // Rejoin a session that a dev-server reload interrupted, once the head is loaded and the canvas has a size.
  const ready = setInterval(() => {
    const canvas = document.querySelector('#stage canvas');
    if (window.__labReady && canvas?.width > 0) {
      clearInterval(ready);
      if (arena.resume()) show('arena');
    }
  }, 300);
  // Opt-in shared-engine Arena. Users flip it via ?arena_omni=1 or by setting
  // localStorage.'contact-sponsors-arena-omni'='1'. Sponsor Arena is unaffected.
  enableArenaOmniIfFlagged({ stats });
}

start();
