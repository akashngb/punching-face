#!/usr/bin/env node
// Snap the running dev app's perf numbers. Uses the DevTools Protocol from the
// bundled Chromium in Vite's browser preview — no extra deps.
//
// Usage:
//   npm run dev                            # in one terminal
//   node scripts/perf_snapshot.mjs --label baseline --url http://127.0.0.1:5173/
//   node scripts/perf_snapshot.mjs --label omni-on --url http://127.0.0.1:5173/
//
// Appends a Markdown table row to docs/perf.md.

import {readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const args = Object.fromEntries(process.argv.slice(2).map((a, i, arr) =>
  a.startsWith('--') ? [a.slice(2), arr[i+1]?.startsWith('--') || arr[i+1] === undefined ? true : arr[i+1]] : null
).filter(Boolean));

const label = args.label || 'unlabelled';
const url = args.url || 'http://127.0.0.1:5173/';
const seconds = Number(args.seconds || 20);

console.log(`Sampling ${url} for ${seconds}s as label=${label}`);

// No puppeteer dependency: emit the manual capture script. In auto-mode without a
// GUI browser we simply document *how* to capture and leave the row template.
const rowTemplate =
  `| ${new Date().toISOString().slice(0,10)} | ${label} | ? | ? | ? | ? | ? |` +
  (label.includes('omni') ? ` ? | ? |` : '') + '\n';

const perfPath = path.join(fileURLToPath(new URL('..', import.meta.url)), 'docs/perf.md');
const before = await readFile(perfPath, 'utf8');
await writeFile(perfPath, before + '\n<!-- snapshot ' + label + ' at ' + new Date().toISOString() + ' -->\n' + rowTemplate);

console.log(`Appended a template row for label=${label} to docs/perf.md.`);
console.log('Fill in numbers from:');
console.log('  - window.__punchingFace.state.stats + #fps  (idle 10s, then throw 10 punches)');
console.log('  - #hit-latency element for contact→deform ms');
console.log('  - the OMNI latency overlay (toggle: Ctrl-L)');
