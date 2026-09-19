#!/usr/bin/env node
// Measure the running dev app over the DevTools Protocol and append a filled row to docs/perf.md.
// Node 22 ships a global WebSocket, so this needs no dependencies.
//
// Usage:
//   npm run dev                                   # in one terminal
//   node scripts/perf_snapshot.mjs --label baseline
//   node scripts/perf_snapshot.mjs --label omni-on --url "http://127.0.0.1:5173/?arena_omni=1"
//
// Connects to Chrome on --port (default 9222). When nothing is listening there it launches its own
// Chrome against a throwaway profile, so a measurement never disturbs your real browser session.
// Page.bringToFront is mandatory: requestAnimationFrame is throttled in a background tab, and a
// throttled sample reads as ~0 fps rather than as a failure.

import { readFile, writeFile, mkdtemp } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import path from 'node:path';

const args = Object.fromEntries(
  process.argv
    .slice(2)
    .map((a, i, arr) =>
      a.startsWith('--')
        ? [
            a.slice(2),
            arr[i + 1]?.startsWith('--') || arr[i + 1] === undefined
              ? true
              : arr[i + 1],
          ]
        : null,
    )
    .filter(Boolean),
);

const label = args.label || 'unlabelled';
const url = args.url || 'http://127.0.0.1:5173/';
const seconds = Number(args.seconds || 10);
const port = Number(args.port || 9222);
const ROOT = fileURLToPath(new URL('..', import.meta.url));

const CHROME =
  process.env.CHROME_PATH ||
  [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
  ].find((p) => existsSync(p));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function version() {
  try {
    return await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  } catch {
    return null;
  }
}

async function ensureChrome() {
  if (await version()) return null;
  if (!CHROME)
    throw new Error(
      'No Chrome found. Set CHROME_PATH, or start Chrome with --remote-debugging-port=' +
        port,
    );
  const profile = await mkdtemp(path.join(tmpdir(), 'perf-snap-'));
  const child = spawn(
    CHROME,
    [
      `--remote-debugging-port=${port}`,
      `--user-data-dir=${profile}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--new-window',
      url,
    ],
    { stdio: 'ignore', detached: false },
  );
  for (let i = 0; i < 60; i++) {
    if (await version()) return child;
    await sleep(500);
  }
  throw new Error('Chrome did not expose the DevTools port in 30s');
}

async function pageTarget() {
  for (let i = 0; i < 40; i++) {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const hit = list.find(
      (t) => t.type === 'page' && t.url.startsWith(url.split('?')[0]),
    );
    if (hit) return hit;
    if (i === 0)
      await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`, {
        method: 'PUT',
      }).catch(() => {});
    await sleep(500);
  }
  throw new Error(`No page target for ${url}. Is \`npm run dev\` running?`);
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  const pending = new Map();
  let id = 0;
  ws.addEventListener('message', (ev) => {
    const msg = JSON.parse(ev.data);
    const slot = pending.get(msg.id);
    if (!slot) return;
    pending.delete(msg.id);
    msg.error ? slot.reject(new Error(msg.error.message)) : slot.resolve(msg.result);
  });
  const ready = new Promise((res, rej) => {
    ws.addEventListener('open', res);
    ws.addEventListener('error', rej);
  });
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      pending.set(++id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
  return { ws, ready, send };
}

async function evaluate(send, expression) {
  const r = await send('Runtime.evaluate', {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (r.exceptionDetails)
    throw new Error(r.exceptionDetails.exception?.description || 'evaluate failed');
  return r.result.value;
}

// Sampled in the page: count rAF callbacks over `ms`, optionally throwing a hook every 700 ms.
const fpsProbe = (ms, punch) => `(async () => {
  ${punch ? `const puncher = setInterval(() => window.dispatchEvent(new KeyboardEvent('keydown', {code: Math.random() < .5 ? 'KeyQ' : 'KeyE'})), 700);` : ''}
  const fps = await new Promise(res => { let n = 0; const t0 = performance.now();
    (function loop(){ n++; const dt = performance.now() - t0; dt < ${ms} ? requestAnimationFrame(loop) : res(n / (dt / 1000)); })(); });
  ${punch ? 'clearInterval(puncher);' : ''}
  const s = window.__punchingFace.state;
  return {fps, impacts: s.impacts, lastSpeed: s.lastSpeed, peak: s.peak,
          physics: s.physics ?? null, metrics: s.physicsMetrics ?? null,
          hitLatency: document.querySelector('#hit-latency')?.textContent ?? null};
})()`;

const chrome = await ensureChrome();
const target = await pageTarget();
const { ws, ready, send } = connect(target.webSocketDebuggerUrl);
await ready;
try {
  await send('Page.enable');
  await send('Page.bringToFront');
  await sleep(500);
  const visible = await evaluate(send, 'document.visibilityState');
  if (visible !== 'visible')
    throw new Error(
      `Tab is "${visible}" — rAF is throttled and the sample would be meaningless. Un-minimise the Chrome window and retry.`,
    );
  await evaluate(
    send,
    `(async()=>{for(let i=0;i<120;i++){if(window.__punchingFace?.state?.ready)return true;await new Promise(r=>setTimeout(r,250));}throw new Error('app never reported ready')})()`,
  );

  console.log(`Sampling ${url} for ${seconds}s idle, then ${seconds}s under punches…`);
  const idle = await evaluate(send, fpsProbe(seconds * 1000, false));
  const busy = await evaluate(send, fpsProbe(seconds * 1000, true));

  const n = (v, d = 1) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');
  const omni = label.includes('omni');
  // Each table has its own column layout; a row goes under that table's marker.
  const day = new Date().toISOString().slice(0, 10);
  const row = omni
    ? `| ${day} | ${label} | — | — | ${n(idle.fps)} | ${n(busy.fps)} | — | — | — |`
    : `| ${day} | ${label} | — | — | ${n(idle.fps)} | ${n(busy.fps)} | — |`;
  const marker = omni ? '<!-- ROWS:omni -->' : '<!-- ROWS:baseline -->';
  const perfPath = path.join(ROOT, 'docs/perf.md');
  const before = await readFile(perfPath, 'utf8');
  if (!before.includes(marker)) throw new Error(`docs/perf.md is missing ${marker}`);
  await writeFile(
    perfPath,
    before.replace(
      marker,
      `${row}
${marker}`,
    ),
  );
  console.log(row);
  console.log(
    `Appended to docs/perf.md. Physics metrics: ${JSON.stringify(busy.metrics)}`,
  );
} finally {
  ws.close();
  if (chrome) chrome.kill();
}
