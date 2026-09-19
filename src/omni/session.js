// Omni Session: connect to the relay's Realtime WebSocket, keep it warm, reconnect
// with backoff, and expose typed events. Scenarios never talk to the socket directly;
// they use this facade so the transport (Plan A/B/C) can change without them noticing.
//
// The relay lives on 127.0.0.1:5177 and always speaks WebSocket, whether it's
// bridging a real OMNI upstream or its in-process mock. See omni_relay.py.

import {latencyOverlay} from './latency.js';

const RELAY_HTTP = 'http://127.0.0.1:5177';
const RELAY_WS = 'ws://127.0.0.1:5177/omni/realtime';

const NOOP = () => {};

export class OmniSession extends EventTarget {
  constructor({relayHttp = RELAY_HTTP, relayWs = RELAY_WS, keepWarm = true} = {}) {
    super();
    this.relayHttp = relayHttp;
    this.relayWs = relayWs;
    this.socket = null;
    this.plan = null;
    this.mock = null;
    this.sessionId = null;
    this.attempt = 0;
    this.keepWarm = keepWarm;
    this.persona = null;
    this.tools = [];
    this.enabled = true;
    this._pendingPersona = null;
    this._pendingTools = null;
    this._pingTimer = null;
    this._closedByUser = false;
    this._latency = latencyOverlay();
    // yibuapi Realtime rejects `input_image_buffer.append` unless at least
    // one `input_audio_buffer.commit` has flowed first. We buffer frames until
    // that happens, then stream them freely. See scripts/omni_smoke_test.py.
    this._audioCommittedOnce = false;
    this._frameBuffer = [];
  }

  /**
   * Load /health once so the badge can tell the user what plan is active before
   * we've opened the socket. Also unblocks the scenario UI early.
   */
  async prime() {
    try {
      const response = await fetch(this.relayHttp + '/health');
      if (!response.ok) throw new Error(`relay ${response.status}`);
      const health = await response.json();
      this.enabled = health.enabled !== false;
      this.plan = health.plan;
      // The relay tells us what voice the current key actually accepts.
      // We echo this into session.update so the model doesn't 400 on us.
      if (health.voice) this.voice = health.voice;
      this._emit('health', health);
      return health;
    } catch (error) {
      this.enabled = false;
      this._emit('offline', {reason: error.message});
      return null;
    }
  }

  configure({persona, tools}) {
    if (persona !== undefined) this.persona = persona;
    if (tools !== undefined) this.tools = tools;
    // If we already have a session, resend on the fly.
    if (this.socket && this.socket.readyState === 1) this._sendSessionUpdate();
    else {
      this._pendingPersona = this.persona;
      this._pendingTools = this.tools;
    }
  }

  async connect() {
    if (this.socket && this.socket.readyState <= 1) return;
    this._closedByUser = false;
    await this.prime();
    if (!this.enabled) return;
    this.attempt++;
    return new Promise((resolve, reject) => {
      let done = false;
      const socket = new WebSocket(this.relayWs);
      socket.binaryType = 'arraybuffer';
      this.socket = socket;
      this._latency.mark('session.open.start');

      socket.onopen = () => {
        this._latency.mark('session.open.done');
        this._sendSessionUpdate();
        this._pingTimer = setInterval(() => {
          if (socket.readyState === 1) socket.send(JSON.stringify({type: 'engine.ping', t: Date.now()}));
        }, 20_000);
      };
      socket.onmessage = event => {
        if (typeof event.data === 'string') this._onText(event.data);
        else this._onBinary(event.data);
      };
      socket.onerror = () => {
        if (!done) { done = true; reject(new Error('relay socket error')); }
      };
      socket.onclose = () => {
        clearInterval(this._pingTimer);
        this._pingTimer = null;
        this._emit('closed', {clean: this._closedByUser});
        if (!this._closedByUser && this.keepWarm) this._reconnect();
        if (!done) { done = true; resolve(); }
      };
      // Resolve as soon as the socket is open — callers get an early handle.
      const readyPoll = setInterval(() => {
        if (socket.readyState === 1 && !done) { done = true; clearInterval(readyPoll); resolve(); }
      }, 40);
      // Fail fast if handshake takes too long.
      setTimeout(() => { if (!done) { done = true; clearInterval(readyPoll); reject(new Error('relay timeout')); } }, 8000);
    });
  }

  _sendSessionUpdate() {
    if (!this.socket || this.socket.readyState !== 1) return;
    const session = {
      modalities: ['text', 'audio'],
      instructions: this._pendingPersona ?? this.persona,
      tools: this._pendingTools ?? this.tools,
      input_audio_format: 'pcm16',
      output_audio_format: 'pcm16',
      turn_detection: null,
    };
    if (this.voice) session.voice = this.voice;
    this.socket.send(JSON.stringify({type: 'session.update', session}));
    this._pendingPersona = null;
    this._pendingTools = null;
  }

  _reconnect() {
    const wait = Math.min(30_000, 500 * 2 ** Math.min(this.attempt, 6));
    this._emit('reconnecting', {waitMs: wait, attempt: this.attempt});
    setTimeout(() => this.connect().catch(NOOP), wait);
  }

  disconnect() {
    this._closedByUser = true;
    if (this.socket) { try { this.socket.close(1000); } catch { /* already closing */ } }
  }

  /**
   * Send an audio chunk (PCM16 mono, base64) to the model. Called from
   * `omni/capture` — audio flows in continuously; commit() ends the turn.
   */
  appendAudio(base64) {
    if (!this.socket || this.socket.readyState !== 1) return;
    this.socket.send(JSON.stringify({type: 'input_audio_buffer.append', audio: base64}));
  }

  commitAudio() {
    if (!this.socket || this.socket.readyState !== 1) return;
    this.socket.send(JSON.stringify({type: 'input_audio_buffer.commit'}));
    this._audioCommittedOnce = true;
    this._latency.mark('audio.commit');
    // After the first commit, any frames we buffered are safe to send.
    if (this._frameBuffer.length) {
      for (const base64 of this._frameBuffer) {
        this.socket.send(JSON.stringify({type: 'input_image_buffer.append', image: base64}));
      }
      this._frameBuffer.length = 0;
      this._latency.mark('frame.sent');
    }
  }

  /** Send one image frame (JPEG base64, no `data:` prefix). Held until the
   * first `input_audio_buffer.commit` flows through the session (yibuapi
   * requirement), then streamed freely. */
  sendFrame(base64) {
    if (!this.socket || this.socket.readyState !== 1) return;
    if (!this._audioCommittedOnce) {
      // Only keep the freshest few frames while we wait.
      this._frameBuffer.push(base64);
      if (this._frameBuffer.length > 4) this._frameBuffer.shift();
      return;
    }
    this.socket.send(JSON.stringify({type: 'input_image_buffer.append', image: base64}));
    this._latency.mark('frame.sent');
  }

  /** Push a synthetic engine event into the model's context (e.g. `strike cheek-left`).
   * The relay uses these to drive the mock and forwards them to the upstream in
   * a way that keeps them ordered relative to audio and frames.
   *
   * If no audio has been committed yet, prime the audio buffer with a tiny
   * silent PCM16 chunk so any buffered frames can flow before the model is
   * asked to respond. */
  sendEngineEvent(event, contextText) {
    if (!this.socket || this.socket.readyState !== 1) return;
    this._maybePrimeAudio();
    this.socket.send(JSON.stringify({type: 'engine.event', event, contextText}));
    this._latency.mark('event.sent');
  }

  _maybePrimeAudio() {
    if (this._audioCommittedOnce) return;
    // 40 ms of silence, PCM16 mono 16 kHz — 640 samples, 1280 bytes.
    const silent = new Uint8Array(1280);
    let s = '';
    for (let i = 0; i < silent.length; i += 8192) {
      s += String.fromCharCode.apply(null, silent.subarray(i, i + 8192));
    }
    this.socket.send(JSON.stringify({type: 'input_audio_buffer.append', audio: btoa(s)}));
    this.commitAudio();
  }

  /** Type a text turn without speaking. */
  sendUserText(text) {
    if (!this.socket || this.socket.readyState !== 1) return;
    this._maybePrimeAudio();
    this.socket.send(JSON.stringify({
      type: 'conversation.item.create',
      item: {type: 'message', role: 'user', content: [{type: 'input_text', text}]}
    }));
    this.socket.send(JSON.stringify({type: 'response.create', response: {modalities: ['text', 'audio']}}));
  }

  /** Prompt the model to respond in a specific way (used for proactive turns). */
  requestResponse(instructions, {modalities = ['text', 'audio']} = {}) {
    if (!this.socket || this.socket.readyState !== 1) return;
    this._maybePrimeAudio();
    this.socket.send(JSON.stringify({type: 'response.create',
      response: {modalities, instructions}}));
    this._latency.mark('response.requested');
  }

  /** Cancel an in-flight response (for barge-in). */
  cancelResponse() {
    if (!this.socket || this.socket.readyState !== 1) return;
    this.socket.send(JSON.stringify({type: 'response.cancel'}));
  }

  // ---------------------- private: dispatch typed events ---------------------

  _onText(data) {
    let msg;
    try { msg = JSON.parse(data); } catch { return; }
    const type = msg.type || msg.event;
    switch (type) {
      case 'omni.session.ready':
        this.plan = msg.plan;
        this.mock = !!msg.mock;
        this.sessionId = msg.sessionId;
        this.attempt = 0;
        this._emit('ready', msg);
        return;
      case 'omni.session.error':
        this._emit('error', {reason: msg.reason});
        return;
      case 'session.updated':
        this._emit('session.updated', msg.session || {});
        return;
      case 'response.output_text.delta':
      case 'response.text.delta':
        this._emit('text.delta', {delta: msg.delta || msg.text || ''});
        return;
      case 'response.output_audio.delta':
      case 'response.audio.delta':
        this._emit('audio.delta', {base64: msg.delta || msg.audio, rate: msg.sample_rate_hz || 24000});
        return;
      case 'response.function_call_arguments.done':
      case 'response.tool_call':
        this._emit('tool.call', {
          name: msg.name || msg.tool_name,
          arguments: this._parseArgs(msg.arguments || msg.tool_arguments || '{}'),
          call_id: msg.call_id || msg.tool_call_id || null,
        });
        return;
      case 'response.done':
        this._emit('response.done', msg.response || {});
        this._latency.mark('response.done');
        return;
      case 'input_audio_buffer.speech_started':
        this._emit('user.speaking', {});
        return;
      case 'input_audio_buffer.speech_stopped':
        this._emit('user.stopped', {});
        return;
      case 'error':
        this._emit('error', msg);
        return;
      default:
        this._emit('raw', msg);
    }
  }

  _onBinary(buffer) {
    this._emit('binary', {buffer});
  }

  _parseArgs(payload) {
    if (typeof payload !== 'string') return payload || {};
    try { return JSON.parse(payload); } catch { return {_raw: payload}; }
  }

  _emit(name, detail) {
    this.dispatchEvent(new CustomEvent(name, {detail}));
  }
}
