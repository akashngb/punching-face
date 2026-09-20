// The OMNI Live voice in the room. It sees (webcam keyframes), hears (your voice, hands-free), and
// speaks (streamed audio, with optional interruptions). Two modes, picked in the panel and sent with every turn:
// `face` — the head you are punching, talking back, which is the default; `coach` — a cornerman
// calling corrections. Only the system prompt and the labels change; the transport is identical.
//
// Perception stays on this device at 30 Hz; only a spoken question, up to four small keyframes and a
// few numbers leave it, once per turn, and only while the panel is switched on. The key never reaches
// this page: the loopback relay holds it.
import {
  downsample,
  encodeWav,
  bytesToBase64,
  base64ToBytes,
  pcm16ToFloat32,
  rms,
  VoiceGate,
} from './audio.js';
import { EventStream } from './sse.js';
import { obs } from './sentry.js';
import { createMouthSignal, createSyntheticSignal } from '../omni/mouth-signal.js';

const TAP =
  "class Tap extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('punching-face-tap',Tap)";
const FRAME_MS = 20,
  PREROLL_FRAMES = 15,
  KEYFRAME_MS = 700,
  KEYFRAMES = 4,
  QUIET_AFTER_TURN_MS = 7000,
  PUNCHES_PER_CUE = 8;
const MODE_KEY = 'punching-face-sponsors-mode';
const INTERACTION_KEY = 'punching-face-sponsors-voice-interaction';
const MODES = {
  face: {
    label: 'Face',
    tab: 'The Face',
    start: 'Wake the face',
    stop: 'Shut it up',
    speaker: 'Face: ',
    ask: 'Say something',
    see: 'Let it see me',
    talk: 'React to punches',
    idle: 'Listening.',
    off: 'The face is off.',
  },
  coach: {
    label: 'Coach',
    tab: 'Cornerman',
    start: 'Start coach',
    stop: 'Stop coach',
    speaker: 'Coach: ',
    ask: 'Ask a question',
    see: 'Let the coach see me',
    talk: 'React to combos',
    idle: 'Listening.',
    off: 'Coach is off.',
  },
};

export function createCornerman({ api, panel, config, stats, refreshConfig, onMode }) {
  panel.innerHTML = `
    <div class="sd-row" style="margin-top:0"><select data-k="mode" aria-label="Who is talking"><option value="face">Trash talk — the face</option><option value="coach">Coach — a cornerman</option></select></div>
    <div class="sd-row"><button class="primary" data-k="toggle" style="flex:1">Wake the face</button><span class="sd-badge" data-k="model"></span></div>
    <div class="sd-meter" data-k="meter"><i></i></div>
    <div class="sd-status" data-k="status">Ready.</div>
    <div class="sd-log" data-k="log" aria-live="polite"></div>
    <div class="sd-row"><input type="text" data-k="ask" maxlength="300"><button data-k="send">Ask</button></div>
    <label class="sd-check"><input type="checkbox" data-k="vision" checked><span data-k="seelabel">Let it see me </span><span class="sd-badge leaving" data-k="leaving"></span></label>
    <label class="sd-check"><input type="checkbox" data-k="voice" checked><span>Spoken replies</span></label>
    <label class="sd-status" for="cornerman-interaction">Voice interaction</label>
    <div class="sd-row"><select id="cornerman-interaction" data-k="interaction"><option value="ambient">Ambient · finish replies</option><option value="interrupt">Interrupt · talk over replies</option></select></div>
    <label class="sd-check"><input type="checkbox" data-k="proactive" checked><span data-k="talklabel"></span></label>
    <details><summary>OMNI key</summary>
      <div class="sd-status">Stored only on this computer (<code>.local/secrets/omni.json</code>, mode 0600). Get one from the Huawei form; the gateway is <span data-k="gateway"></span>.</div>
      <div class="sd-row"><input type="password" data-k="key" placeholder="API key" autocomplete="off"><button data-k="save">Save</button></div>
    </details>`;
  const el = Object.fromEntries(
    [...panel.querySelectorAll('[data-k]')].map((n) => [n.dataset.k, n]),
  );
  let ctx = null,
    mic = null,
    node = null,
    out = null,
    streamOut = null,
    gate = null,
    mouth = null,
    synthetic = null,
    enabled = false,
    starting = false,
    session = 0,
    busy = false,
    controller = null,
    nextTime = 0,
    lastTurnAt = -Infinity,
    ambientReadyAt = 0,
    ignoringAudio = false,
    sinceTurn = 0,
    keyTimer = null;
  let pending = new Float32Array(0),
    preroll = [],
    recording = null,
    frames = [],
    history = [];
  const playing = new Set(),
    spokenReplies = new Set(),
    grab = document.createElement('canvas');
  let mode = (() => {
    try {
      return MODES[localStorage.getItem(MODE_KEY)]
        ? localStorage.getItem(MODE_KEY)
        : 'face';
    } catch {
      return 'face';
    }
  })();
  let interaction = (() => {
    try {
      return localStorage.getItem(INTERACTION_KEY) === 'interrupt'
        ? 'interrupt'
        : 'ambient';
    } catch {
      return 'ambient';
    }
  })();
  const voice = () => MODES[mode];
  const speaking = () => playing.size > 0 || spokenReplies.size > 0;
  el.mode.value = mode;
  el.interaction.value = interaction;

  const status = (text, error = false) => {
    el.status.textContent = text;
    el.status.classList.toggle('error', error);
  };
  const say = (who, text) => {
    const p = document.createElement('p');
    p.className = who;
    p.textContent = (who === 'you' ? 'You: ' : voice().speaker) + text;
    el.log.append(p);
    el.log.scrollTop = el.log.scrollHeight;
    return p;
  };
  function paint() {
    const omni = config().omni;
    el.model.textContent = omni.configured ? omni.model : 'MOCK · no key';
    el.model.classList.toggle('mock', !omni.configured);
    el.gateway.textContent = omni.gateway;
    el.leaving.textContent = !enabled
      ? ''
      : el.vision.checked
        ? `≤${KEYFRAMES} keyframes + voice per turn → ${omni.gateway}`
        : 'voice + numbers only';
    el.toggle.textContent = enabled ? voice().stop : voice().start;
    el.toggle.classList.toggle('danger', enabled);
    el.toggle.classList.toggle('primary', !enabled);
    el.ask.placeholder = voice().ask;
    el.seelabel.textContent = voice().see + ' ';
    el.talklabel.textContent = voice().talk;
    el.mode.value = mode;
    onMode?.(voice().tab);
  }

  function stopSpeaking() {
    for (const source of playing) {
      try {
        source.stop();
      } catch {
        /* already ended */
      }
    }
    playing.clear();
    spokenReplies.clear();
    nextTime = 0;
    window.speechSynthesis?.cancel();
    synthetic?.stop();
    if (gate) gate.ratio = 3.2;
  }
  function play(samples, rate) {
    const buffer = ctx.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(samples, 0);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(out);
    const at = Math.max(ctx.currentTime + 0.04, nextTime);
    source.start(at);
    nextTime = at + buffer.duration;
    playing.add(source);
    // Interrupt mode keeps a stiffer gate against speaker echo. Ambient mode ignores
    // the microphone throughout playback and its cooldown instead.
    gate.ratio = 8;
    source.onended = () => {
      playing.delete(source);
      if (!speaking()) {
        if (gate) gate.ratio = 3.2;
        ambientReadyAt = performance.now() + QUIET_AFTER_TURN_MS;
      }
    };
  }

  function keyframe() {
    const video = document.getElementById('webcam');
    if (!enabled || !el.vision.checked || !video?.videoWidth || video.readyState < 2)
      return;
    grab.width = 320;
    grab.height = Math.round((320 * video.videoHeight) / video.videoWidth);
    grab.getContext('2d').drawImage(video, 0, 0, grab.width, grab.height);
    frames.push(grab.toDataURL('image/jpeg', 0.6).split(',')[1]);
    if (frames.length > KEYFRAMES) frames.shift();
  }

  async function turn({ audio = null, text = null, trigger = null }) {
    if (busy) return;
    busy = true;
    controller = new AbortController();
    sinceTurn = 0;
    lastTurnAt = performance.now();
    const sent = el.vision.checked ? frames.slice() : [];
    const body = {
      mode,
      frames: sent,
      telemetry: stats.snapshot(performance.now(), trigger),
      history: history.slice(-6),
      voice: el.voice.checked,
    };
    if (audio)
      body.audioWav = bytesToBase64(
        encodeWav(downsample(audio, ctx.sampleRate, 16000), 16000),
      );
    else if (text) body.text = text;
    if (text) say('you', text);
    else if (audio) say('you', '(spoke)');
    const line = say('coach', '…');
    let said = '',
      mock = false,
      started = performance.now(),
      firstAt = null,
      served = null;
    try {
      await obs.span(
        'coach.turn',
        {
          'coach.mode': mode,
          'coach.trigger': trigger || (audio ? 'voice' : 'text'),
          'coach.frames': sent.length,
          'coach.audio_ms': audio
            ? Math.round((audio.length / ctx.sampleRate) * 1000)
            : 0,
        },
        async (span) => {
          const response = await fetch(api + '/sponsors/coach/turn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
          });
          if (!response.ok)
            throw new Error(
              (await response.json().catch(() => ({}))).error ||
                'The relay refused the request.',
            );
          const reader = response.body.getReader(),
            decoder = new TextDecoder(),
            stream = new EventStream();
          for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            for (const { event, data } of stream.feed(
              decoder.decode(value, { stream: true }),
            )) {
              if (event === 'meta') {
                mock = data.mock;
                span.setAttribute('coach.model', data.model);
                span.setAttribute('coach.mock', !!data.mock);
                // A relay that does not echo the mode back predates the selector, so it is still
                // serving its old system prompt no matter what this page sends.
                served = data.mode ?? null;
                span.setAttribute('coach.mode_served', served || 'unknown');
              } else if (event === 'text') {
                firstAt ??= performance.now();
                said += data.delta;
                line.textContent = voice().speaker + said;
                el.log.scrollTop = el.log.scrollHeight;
              } else if (event === 'audio') {
                firstAt ??= performance.now();
                if (el.voice.checked && ctx)
                  play(pcm16ToFloat32(base64ToBytes(data.pcm16)), data.rate || 24000);
              } else if (event === 'error')
                throw new Error(
                  data.message +
                    (data.detail ? ' ' + String(data.detail).slice(0, 160) : ''),
                );
            }
          }
          span.setAttribute(
            'coach.first_response_ms',
            Math.round((firstAt ?? performance.now()) - started),
          );
        },
      );
      if (!said)
        line.textContent =
          voice().speaker + (mode === 'coach' ? '(no reply)' : '(nothing)');
      // The stand-in has no voice of its own; the browser reads it aloud so the loop can be rehearsed. It is labelled.
      if (mock && said && el.voice.checked && 'speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(said);
        spokenReplies.add(utterance);
        if (gate) gate.ratio = 8;
        // speechSynthesis cannot be routed into WebAudio, so the analyser hears nothing.
        // Drive the stand-in envelope off the utterance instead, as long as it speaks.
        utterance.onstart = () => {
          if (spokenReplies.has(utterance))
            synthetic?.speakFor(Math.max(1.2, said.split(/\s+/).length / 2.6));
        };
        utterance.onend = utterance.onerror = () => {
          if (!spokenReplies.delete(utterance)) return;
          synthetic?.stop();
          if (!speaking()) {
            if (gate) gate.ratio = 3.2;
            ambientReadyAt = performance.now() + QUIET_AFTER_TURN_MS;
          }
        };
        try {
          speechSynthesis.speak(utterance);
        } catch (error) {
          spokenReplies.delete(utterance);
          throw error;
        }
      }
      if (said)
        history.push(
          {
            role: 'user',
            content: text || (audio ? '[spoken question]' : '[asked for a cue]'),
          },
          { role: 'assistant', content: said },
        );
      if (served !== null && served !== mode)
        status(
          `The relay answered as "${served}", not "${mode}". Restart it: npm run sponsors`,
          true,
        );
      else if (served === null)
        status(
          'The relay is running older code and ignored the mode. Restart it: npm run sponsors',
          true,
        );
      else
        status(
          (mock ? 'Mock reply (add an OMNI key for the real model). ' : '') +
            (firstAt ? `First response in ${Math.round(firstAt - started)} ms.` : ''),
        );
    } catch (error) {
      if (error.name === 'AbortError') {
        line.textContent = voice().speaker + (said || '…') + ' (interrupted)';
      } else {
        line.remove();
        status(error.message, true);
        obs.error(error, { feature: 'coach' });
      }
    } finally {
      busy = false;
      controller = null;
      ambientReadyAt = performance.now() + QUIET_AFTER_TURN_MS;
    }
  }

  function resetListening() {
    pending = new Float32Array(0);
    preroll = [];
    recording = null;
    if (gate) {
      // Preserve the learned room noise, but never keep a partial utterance across modes.
      gate.speaking = false;
      gate.loudMs = gate.quietMs = gate.spokenMs = 0;
      gate.levels = [];
      gate.ratio = speaking() ? 8 : 3.2;
    }
    el.meter.classList.remove('open');
    el.meter.firstElementChild.style.width = '0';
  }

  function onAudio(chunk) {
    if (!enabled || !ctx || !gate) return;
    const now = performance.now();
    if (interaction === 'ambient' && (busy || speaking() || now < ambientReadyAt)) {
      if (!ignoringAudio) resetListening();
      ignoringAudio = true;
      // Drop speaker echo and room noise, without buffering it or allocating frames.
      if (busy || speaking()) ambientReadyAt = now + QUIET_AFTER_TURN_MS;
      return;
    }
    ignoringAudio = false;
    const joined = new Float32Array(pending.length + chunk.length);
    joined.set(pending);
    joined.set(chunk, pending.length);
    pending = joined;
    const size = Math.round((ctx.sampleRate * FRAME_MS) / 1000);
    while (pending.length >= size) {
      const frame = pending.slice(0, size);
      pending = pending.slice(size);
      const level = rms(frame),
        event = gate.push(level, FRAME_MS);
      el.meter.firstElementChild.style.width = Math.min(100, level * 600) + '%';
      el.meter.classList.toggle('open', gate.speaking);
      if (recording) recording.push(frame);
      else {
        preroll.push(frame);
        if (preroll.length > PREROLL_FRAMES) preroll.shift();
      }
      if (event === 'start') {
        if (interaction === 'interrupt' && (speaking() || busy)) {
          stopSpeaking();
          controller?.abort();
        }
        recording = preroll.slice();
        preroll = [];
        status('Listening…');
      } else if (event === 'end' || event === 'discard') {
        const spoken = recording;
        recording = null;
        if (event === 'discard' || !spoken) continue;
        const audio = new Float32Array(spoken.reduce((n, f) => n + f.length, 0));
        let offset = 0;
        for (const f of spoken) {
          audio.set(f, offset);
          offset += f.length;
        }
        status('Thinking…');
        turn({ audio });
        if (interaction === 'ambient') {
          resetListening();
          return;
        }
      }
    }
  }

  async function start() {
    if (enabled || starting) {
      if (ctx?.state === 'suspended') {
        try {
          await ctx.resume();
        } catch (error) {
          status(error.message, true);
        }
      }
      return;
    }
    const currentSession = ++session;
    starting = true;
    try {
      ctx = new AudioContext();
      const audioContext = ctx;
      status('Starting audio… If it stays paused, tap ' + voice().start + '.');
      await audioContext.resume();
      if (currentSession !== session) return;
      out = ctx.createGain();
      out.connect(ctx.destination);
      streamOut = ctx.createMediaStreamDestination();
      out.connect(streamOut);
      // The head's mouth follows whatever is coming out of this bus. An analyser is a
      // leaf tap, so neither the speakers nor the LiveKit guest stream are affected.
      mouth = createMouthSignal(ctx, out);
      synthetic = createSyntheticSignal();
      window.__faceSpeech = {
        read: (dt) => {
          const a = mouth.read(dt),
            b = synthetic.read(dt);
          return a.open >= b.open ? a : b;
        },
      };
      gate = new VoiceGate();
      ambientReadyAt = 0;
      ignoringAudio = false;
      enabled = true;
      keyTimer = setInterval(keyframe, KEYFRAME_MS);
      paint();
      try {
        const microphone = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        if (currentSession !== session) {
          microphone.getTracks().forEach((track) => track.stop());
          return;
        }
        mic = microphone;
        const workletUrl = URL.createObjectURL(
          new Blob([TAP], { type: 'application/javascript' }),
        );
        try {
          await audioContext.audioWorklet.addModule(workletUrl);
        } finally {
          URL.revokeObjectURL(workletUrl);
        }
        if (currentSession !== session) return;
        node = new AudioWorkletNode(audioContext, 'punching-face-tap');
        node.port.onmessage = (e) => onAudio(e.data);
        ctx.createMediaStreamSource(mic).connect(node);
        status('Learning the room noise… then just talk.');
        setTimeout(
          () => currentSession === session && enabled && !busy && status(voice().idle),
          900,
        );
      } catch (error) {
        if (currentSession !== session) return;
        status('No microphone (' + error.name + '). Typed questions still work.', true);
        obs.warn('coach.mic_unavailable', { reason: error.name });
      }
      obs.crumb('coach', 'started', {
        vision: el.vision.checked,
        voice: el.voice.checked,
      });
      window.dispatchEvent(
        new CustomEvent('cornerman:audio', { detail: streamOut.stream }),
      );
    } catch (error) {
      if (currentSession !== session) return;
      stop();
      status(error.message, true);
      obs.error(error, { feature: 'coach' });
    } finally {
      if (currentSession === session) starting = false;
    }
  }
  function stop() {
    session++;
    starting = false;
    enabled = false;
    clearInterval(keyTimer);
    controller?.abort();
    stopSpeaking();
    if (node) node.port.onmessage = null;
    node?.disconnect();
    mic?.getTracks().forEach((t) => t.stop());
    ctx?.close();
    mouth?.dispose();
    synthetic?.dispose();
    mouth = synthetic = null;
    window.__faceSpeech = null;
    ctx = mic = node = out = streamOut = null;
    frames = [];
    resetListening();
    ignoringAudio = false;
    el.meter.firstElementChild.style.width = '0';
    paint();
    status(voice().off);
  }

  el.toggle.onclick = () => (enabled ? stop() : start());
  el.vision.onchange = () => {
    if (!el.vision.checked) frames = [];
    paint();
  };
  el.interaction.onchange = () => {
    interaction = el.interaction.value === 'interrupt' ? 'interrupt' : 'ambient';
    try {
      localStorage.setItem(INTERACTION_KEY, interaction);
    } catch {
      /* private mode */
    }
    resetListening();
    ignoringAudio = false;
    status(
      interaction === 'interrupt'
        ? 'Interruptions on.'
        : 'Ambient mode. Replies finish before listening.',
    );
  };
  // Switching who is talking cuts the current line off and drops the history: the two personas
  // would otherwise read each other's turns back and answer in the wrong voice.
  function selectMode(nextMode) {
    if (mode === nextMode) return;
    mode = nextMode;
    try {
      localStorage.setItem(MODE_KEY, mode);
    } catch {
      /* private mode */
    }
    controller?.abort();
    stopSpeaking();
    history = [];
    el.log.replaceChildren();
    paint();
    status(
      enabled
        ? `Switched to ${voice().label.toLowerCase()}. Carry on.`
        : `${voice().label} is off. Nothing is being sent.`,
    );
  }
  el.mode.onchange = () => selectMode(MODES[el.mode.value] ? el.mode.value : 'face');
  const ask = () => {
    const text = el.ask.value.trim();
    if (!text) return;
    el.ask.value = '';
    turn({ text });
  };
  el.send.onclick = ask;
  el.ask.onkeydown = (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') ask();
  };
  el.ask.onkeyup = (e) => e.stopPropagation();
  el.key.onkeydown = (e) => e.stopPropagation();
  el.save.onclick = async () => {
    const apiKey = el.key.value.trim();
    if (!apiKey) return;
    el.key.value = '';
    try {
      const r = await fetch(api + '/sponsors/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ group: 'omni', apiKey }),
      });
      if (!r.ok) throw new Error((await r.json()).error);
      await refreshConfig();
      paint();
      status('Key saved on this computer.');
    } catch (error) {
      status(error.message, true);
    }
  };
  paint();

  return {
    startFace() {
      selectMode('face');
      el.voice.checked = true;
      el.proactive.checked = true;
      paint();
      return start();
    },
    // Called for every landed punch, local or remote. The face answers back at natural beats,
    // never over a person who is speaking, and never back-to-back.
    onPunch(triggers) {
      sinceTurn++;
      if (
        !enabled ||
        !el.proactive.checked ||
        busy ||
        gate?.speaking ||
        speaking() ||
        performance.now() - lastTurnAt < QUIET_AFTER_TURN_MS
      )
        return;
      if (triggers.length || sinceTurn >= PUNCHES_PER_CUE)
        turn({
          trigger:
            triggers[0] || `${PUNCHES_PER_CUE} punches since it last said anything`,
        });
    },
    get outputStream() {
      return streamOut?.stream || null;
    },
    repaint: paint,
  };
}
