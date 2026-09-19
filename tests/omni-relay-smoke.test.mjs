// Smoke test for the OMNI relay: boot omni_relay.py in a subprocess, open a
// WebSocket, exchange session.update + engine.event, expect the mock
// upstream to fire back `omni.session.ready` and a `tool.call` for the
// react_to_hit on a cheek strike.
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import net from 'node:net';

const ROOT = path.dirname(fileURLToPath(new URL('..', import.meta.url + '/')));

function pickPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1', () => {
      const {port} = server.address();
      server.close(() => resolve(port));
    });
    server.on('error', reject);
  });
}

async function waitForListen(port, timeoutMs = 4000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const ok = await new Promise(resolve => {
      const socket = net.connect(port, '127.0.0.1');
      socket.once('connect', () => { socket.end(); resolve(true); });
      socket.once('error', () => resolve(false));
    });
    if (ok) return true;
    await new Promise(r => setTimeout(r, 100));
  }
  return false;
}

// Minimal WS client — enough to exchange text frames with the relay.
function connectWs(port) {
  return new Promise((resolve, reject) => {
    const socket = net.connect(port, '127.0.0.1');
    const key = Buffer.from(Math.random().toString(36)).toString('base64');
    let buf = Buffer.alloc(0);
    let handshaked = false;
    const listeners = [];
    socket.on('data', chunk => {
      buf = Buffer.concat([buf, chunk]);
      if (!handshaked) {
        const idx = buf.indexOf('\r\n\r\n');
        if (idx === -1) return;
        const head = buf.slice(0, idx).toString('utf8');
        if (!head.startsWith('HTTP/1.1 101')) return reject(new Error('bad handshake: ' + head));
        buf = buf.slice(idx + 4);
        handshaked = true;
        resolve({
          send(text) {
            const payload = Buffer.from(text);
            const header = [];
            header.push(0x81);
            const mask = Buffer.from([0, 0, 0, 0]);
            if (payload.length < 126) header.push(0x80 | payload.length);
            else if (payload.length < 65536) {
              header.push(0x80 | 126, (payload.length >> 8) & 0xff, payload.length & 0xff);
            } else {
              header.push(0x80 | 127, 0,0,0,0, (payload.length >>> 24) & 0xff, (payload.length >>> 16) & 0xff, (payload.length >>> 8) & 0xff, payload.length & 0xff);
            }
            socket.write(Buffer.concat([Buffer.from(header), mask, payload]));
          },
          onMessage(fn) { listeners.push(fn); },
          close() { socket.end(); },
        });
        return;
      }
      // Parse frames.
      while (buf.length >= 2) {
        const b1 = buf[0], b2 = buf[1];
        let length = b2 & 0x7F;
        let offset = 2;
        if (length === 126) { if (buf.length < 4) return; length = buf.readUInt16BE(2); offset = 4; }
        else if (length === 127) { if (buf.length < 10) return; length = Number(buf.readBigUInt64BE(2)); offset = 10; }
        if (buf.length < offset + length) return;
        const payload = buf.slice(offset, offset + length);
        buf = buf.slice(offset + length);
        const opcode = b1 & 0x0F;
        if (opcode === 0x1) for (const fn of listeners) fn(payload.toString('utf8'));
      }
    });
    socket.on('error', reject);
    socket.write(`GET /omni/realtime HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\nOrigin: http://127.0.0.1:5173\r\n\r\n`);
  });
}

test('relay boots, upgrades a client, and echoes mock events', async () => {
  const port = 5177;
  const child = spawn('.venv/bin/python', ['omni_relay.py'], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    // OMNI_ENABLED=false forces the relay into mock mode regardless of .env.
    env: {...process.env, OMNI_ENABLED: 'false', OMNI_API_KEY: '', OMNI_REALTIME_URL: ''},
  });
  try {
    const listening = await waitForListen(port, 5000);
    assert.ok(listening, 'relay never came up');
    const client = await connectWs(port);
    const messages = [];
    client.onMessage(text => { try { messages.push(JSON.parse(text)); } catch { messages.push({_raw: text}); } });
    // Configure the session and push a strike event so the mock issues a tool call.
    client.send(JSON.stringify({type: 'session.update', session: {instructions: 'test', tools: [{name: 'react_to_hit'}]}}));
    client.send(JSON.stringify({type: 'engine.event', event: {type: 'strike', region: 'cheek-left', force: 55}}));
    // Wait up to 2 s for the tool call.
    const deadline = Date.now() + 2000;
    while (Date.now() < deadline) {
      if (messages.some(m => m.type === 'response.function_call_arguments.done' && m.name === 'react_to_hit')) break;
      await new Promise(r => setTimeout(r, 30));
    }
    client.close();
    assert.ok(messages.some(m => m.type === 'omni.session.ready'), 'no ready event');
    assert.ok(messages.some(m => m.type === 'session.updated'), 'no session.updated');
    const call = messages.find(m => m.type === 'response.function_call_arguments.done');
    assert.ok(call, 'no function call from mock');
    assert.equal(call.name, 'react_to_hit');
    const args = JSON.parse(call.arguments);
    assert.equal(args.location, 'cheek-left');
    assert.ok(args.severity >= 1 && args.severity <= 3);
  } finally {
    child.kill('SIGTERM');
    await new Promise(r => setTimeout(r, 200));
  }
});
