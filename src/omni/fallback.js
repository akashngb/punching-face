// Plan C fallback: turn-based chat completions → text → ElevenLabs voice.
//
// Same public surface as OmniSession so scenarios never branch on transport.
// Under Plan C we lose:
//   - streaming audio in (we send a WAV blob per turn)
//   - semantic interruption (we handle barge-in ourselves)
//   - cloned voice from OMNI (we use ElevenLabs; document it honestly)
//
// The relay is *not* the transport for Plan C — we reuse the existing
// sponsor_server.py `/sponsors/coach/turn` SSE endpoint, since it already
// handles the key, streaming and Sentry AI-monitoring spans. That means Plan
// C works even if the OMNI relay (:5177) is down. See OMNI.md §6.5.

import {latencyOverlay} from './latency.js';

const DEFAULT_TURN_URL = 'http://127.0.0.1:5176/sponsors/coach/turn';

export class FallbackSession extends EventTarget {
  constructor({turnUrl = DEFAULT_TURN_URL, persona, voiceUrl} = {}) {
    super();
    this.turnUrl = turnUrl;
    this.voiceUrl = voiceUrl || null;   // optional ElevenLabs POST endpoint
    this.persona = persona;
    this.plan = 'C';
    this.sessionId = 'fallback-' + Math.random().toString(36).slice(2, 10);
    this.mock = false;
    this._latency = latencyOverlay();
    this._pending = null;
    this._audioBuffer = [];
    this._frames = [];
    this._history = [];
    this._toolPrompt = '';
  }

  async prime() { return {plan: 'C'}; }
  async connect() { this._emit('ready', {plan: 'C', mock: false, sessionId: this.sessionId}); }
  disconnect() { this._pending?.abort?.(); }
  configure({persona, tools}) {
    if (persona !== undefined) this.persona = persona;
    if (tools) {
      // In turn-based mode we tell the model about the tools via a system-line;
      // it can't call them directly. It answers with JSON that we parse.
      this._toolPrompt = 'When you would take an action, output ONLY a JSON object '
        + '`{"tool":"NAME","arguments":{...}}` on its own line. Available tools: '
        + tools.map(t => t.name).join(', ') + '. '
        + 'Otherwise reply naturally.';
    }
  }

  appendAudio(base64) { this._audioBuffer.push(base64); }
  commitAudio() {
    // Concatenate PCM16 chunks into a WAV and post a turn. The sponsor endpoint
    // already accepts `audioWav` as base64.
    if (!this._audioBuffer.length) return;
    const wav = _pcm16ChunksToWav(this._audioBuffer, 16000);
    this._audioBuffer = [];
    this._turn({audioWav: _bytesToBase64(wav)});
  }
  sendFrame(base64) {
    this._frames.push(base64);
    if (this._frames.length > 4) this._frames.shift();
    this._latency.mark('frame.sent');
  }
  sendEngineEvent(event, contextText) {
    // Turn-based: emit a text turn immediately when the event is high-value
    // (release with rebound, guard change, consent flip). Otherwise buffer it
    // in history for the next turn.
    this._history.push({role: 'system', content: contextText});
    const urgent = event?.type === 'release' && event?.rebound
      || event?.type === 'consent';
    if (urgent) this._turn({text: contextText});
  }
  sendUserText(text) { this._turn({text}); }
  requestResponse(instructions) { this._turn({text: instructions}); }
  cancelResponse() { this._pending?.abort?.(); }

  async _turn(body) {
    this._pending?.abort?.();
    const controller = new AbortController();
    this._pending = controller;
    const start = performance.now();
    const payload = {
      frames: this._frames.slice(),
      telemetry: {},
      history: this._history.slice(-6),
      voice: true,
      ...body,
      // A system-prompt-shaped nudge; the real prompt is the sponsor server's COACH.
      persona: this.persona,
      toolInstruction: this._toolPrompt,
    };
    let said = '';
    let firstDelta = null;
    try {
      const response = await fetch(this.turnUrl, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`turn HTTP ${response.status}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const {value, done} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        // Parse SSE frames.
        const parts = buffer.split('\n\n');
        buffer = parts.pop();
        for (const part of parts) {
          const eventLine = /^event: (.+)$/m.exec(part);
          const dataLine = /^data: (.+)$/m.exec(part);
          if (!eventLine || !dataLine) continue;
          const evt = eventLine[1];
          let data; try { data = JSON.parse(dataLine[1]); } catch { data = {}; }
          if (evt === 'text') {
            firstDelta ??= performance.now();
            said += data.delta || '';
            this._emit('text.delta', {delta: data.delta || ''});
          } else if (evt === 'audio') {
            firstDelta ??= performance.now();
            this._emit('audio.delta', {base64: data.pcm16, rate: data.rate || 24000});
          } else if (evt === 'error') {
            this._emit('error', {reason: data.message || 'fallback error'});
          } else if (evt === 'done') {
            this._latency.mark('response.done', performance.now());
            this._emit('response.done', data);
          }
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') this._emit('error', {reason: error.message});
    } finally {
      this._pending = null;
      // Extract any tool-call from the JSON line the model emitted.
      const toolCall = _parseInlineToolCall(said);
      if (toolCall) this._emit('tool.call', toolCall);
      if (said) this._history.push({role: 'user', content: body.text || '[turn]'},
                                    {role: 'assistant', content: said});
      // Latency book-keeping so the overlay row is complete.
      if (firstDelta) this._latency.mark('response.first_delta', firstDelta);
    }
  }

  _emit(name, detail) { this.dispatchEvent(new CustomEvent(name, {detail})); }
}

function _parseInlineToolCall(text) {
  if (!text) return null;
  const match = text.match(/\{"tool"\s*:\s*"([^"]+)"[^{}]*?("arguments"\s*:\s*(\{[^{}]*\}))?\s*\}/);
  if (!match) return null;
  const name = match[1];
  let args = {};
  try { args = JSON.parse(match[3] || '{}'); } catch { /* ignore */ }
  return {name, arguments: args, call_id: null};
}

function _pcm16ChunksToWav(base64Chunks, sampleRate) {
  const pcm = _concatFromBase64(base64Chunks);
  const view = new DataView(new ArrayBuffer(44 + pcm.byteLength));
  const writeString = (offset, s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + pcm.byteLength, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, pcm.byteLength, true);
  new Uint8Array(view.buffer).set(pcm, 44);
  return new Uint8Array(view.buffer);
}

function _concatFromBase64(chunks) {
  const buffers = chunks.map(c => _base64ToBytes(c));
  const total = buffers.reduce((n, b) => n + b.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const buffer of buffers) { out.set(buffer, offset); offset += buffer.byteLength; }
  return out;
}

function _base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function _bytesToBase64(bytes) {
  let s = '';
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}
