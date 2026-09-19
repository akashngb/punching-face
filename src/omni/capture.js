// Off-main-thread capture for OMNI Realtime.
//
// Frames: a Web Worker samples a <video> element at 1–2 fps, downscales to
//   ~512 px long edge, encodes JPEG ~70. An extra frame is triggered on every
//   engine contact event (see omni/events).
//
// Audio: an AudioWorklet taps the mic, chunks into 20 ms frames, downsamples
//   to 16 kHz mono PCM16, and yields base64. A tiny voice gate is included so
//   we only ship audio while the user is speaking; barge-in is the caller's
//   business (they call OmniSession.cancelResponse()).
//
// Both streams call callbacks the OmniSession consumer wires up. Nothing here
// touches the render loop; capture is a subscriber to a <video> that main.js
// already renders for its own tracking.

const FRAME_WORKER_SRC = `
self.onmessage = async ev => {
  const {bitmap, quality} = ev.data;
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const blob = await canvas.convertToBlob({type: 'image/jpeg', quality});
  const buffer = await blob.arrayBuffer();
  self.postMessage({buffer}, [buffer]);
};
`;

// AudioWorklet: emit Float32 20 ms frames back to main.
const AUDIO_WORKLET_SRC = `
class OmniTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0][0];
    if (ch && ch.length) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('omni-tap', OmniTap);
`;

const FRAME_MS = 20; // AudioWorklet frame length
const TARGET_RATE = 16000; // 16 kHz PCM16 is Realtime's default
const KEYFRAME_MS_DEFAULT = 700; // ~1.5 fps
const LONG_EDGE_PX = 512; // OMNI.md §6.1
const JPEG_QUALITY = 0.72;

export class FrameCapture {
  constructor({ video, session, keyframeMs = KEYFRAME_MS_DEFAULT } = {}) {
    this.video = video;
    this.session = session;
    this.keyframeMs = keyframeMs;
    this.worker = null;
    this.timer = null;
    this.running = false;
    this.pending = 0;
    this._workerUrl = URL.createObjectURL(
      new Blob([FRAME_WORKER_SRC], { type: 'application/javascript' }),
    );
  }

  start() {
    if (this.running) return;
    this.running = true;
    if (!this.worker) {
      this.worker = new Worker(this._workerUrl);
      this.worker.onmessage = (event) => {
        this.pending = Math.max(0, this.pending - 1);
        const b64 = _bufferToBase64(new Uint8Array(event.data.buffer));
        this.session?.sendFrame(b64);
      };
    }
    this.timer = setInterval(() => this.snapshot(), this.keyframeMs);
  }

  stop() {
    this.running = false;
    clearInterval(this.timer);
    this.timer = null;
    this.worker?.terminate();
    this.worker = null;
  }

  /** Take one frame immediately (called on every contact event). */
  async snapshot() {
    if (!this.running) return;
    if (this.pending > 2) return; // don't back up the worker
    const video = this.video;
    if (!video?.videoWidth || video.readyState < 2) return;
    // ImageBitmap avoids canvas GPU work on the main thread; we ship to worker.
    let bitmap;
    try {
      const w = LONG_EDGE_PX;
      const scale = Math.min(1, w / Math.max(video.videoWidth, video.videoHeight));
      const width = Math.round(video.videoWidth * scale);
      const height = Math.round(video.videoHeight * scale);
      bitmap = await createImageBitmap(video, {
        resizeWidth: width,
        resizeHeight: height,
        resizeQuality: 'medium',
      });
    } catch {
      return;
    }
    this.pending++;
    this.worker.postMessage({ bitmap, quality: JPEG_QUALITY }, [bitmap]);
  }
}

export class AudioCapture extends EventTarget {
  constructor({ session, sampleRate = TARGET_RATE } = {}) {
    super();
    this.session = session;
    this.sampleRate = sampleRate;
    this.ctx = null;
    this.mic = null;
    this.node = null;
    this.running = false;
    this.speaking = false;
    this._pending = new Float32Array(0);
    this._noise = 0.005;
    this._speakingFrames = 0;
    this._quietFrames = 0;
    this._workletUrl = URL.createObjectURL(
      new Blob([AUDIO_WORKLET_SRC], { type: 'application/javascript' }),
    );
  }

  async start() {
    if (this.running) return;
    this.ctx = new AudioContext();
    await this.ctx.resume();
    await this.ctx.audioWorklet.addModule(this._workletUrl);
    this.mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.node = new AudioWorkletNode(this.ctx, 'omni-tap');
    this.node.port.onmessage = (event) => this._onChunk(event.data);
    this.ctx.createMediaStreamSource(this.mic).connect(this.node);
    this.running = true;
    this.dispatchEvent(new CustomEvent('started'));
  }

  stop() {
    this.running = false;
    try {
      this.node?.disconnect();
    } catch {
      /* not connected */
    }
    this.mic?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
    this.ctx = this.mic = this.node = null;
    this.speaking = false;
    this._pending = new Float32Array(0);
  }

  _onChunk(chunk) {
    // Assemble into 20 ms frames at the source sample rate.
    if (!this.running || !this.ctx) return;
    const rate = this.ctx.sampleRate;
    const size = Math.round((rate * FRAME_MS) / 1000);
    const merged = new Float32Array(this._pending.length + chunk.length);
    merged.set(this._pending);
    merged.set(chunk, this._pending.length);
    this._pending = merged;
    while (this._pending.length >= size) {
      const frame = this._pending.slice(0, size);
      this._pending = this._pending.slice(size);
      this._pushFrame(frame, rate);
    }
  }

  _pushFrame(frame, sourceRate) {
    const level = _rms(frame);
    // Voice gate: cheap RMS threshold with hysteresis. The relay does semantic
    // interruption if the model supports it; this only stops us shipping silence.
    this._noise = this._noise * 0.98 + level * 0.02;
    const active = level > this._noise * 3.2 + 0.008;
    if (active) {
      this._speakingFrames++;
      this._quietFrames = 0;
    } else {
      this._speakingFrames = 0;
      this._quietFrames++;
    }
    const wasSpeaking = this.speaking;
    if (!wasSpeaking && this._speakingFrames > 3) this.speaking = true;
    if (wasSpeaking && this._quietFrames > 30) this.speaking = false;
    if (this.speaking !== wasSpeaking) {
      this.dispatchEvent(
        new CustomEvent(this.speaking ? 'speech.start' : 'speech.end'),
      );
      if (!this.speaking && this.session) this.session.commitAudio();
    }
    if (!this.speaking) return; // don't ship silence
    // Downsample linearly to 16 kHz, encode PCM16 little-endian.
    const downsampled = _downsample(frame, sourceRate, this.sampleRate);
    const pcm16 = new Int16Array(downsampled.length);
    for (let i = 0; i < downsampled.length; i++) {
      const s = Math.max(-1, Math.min(1, downsampled[i]));
      pcm16[i] = (s * 32767) | 0;
    }
    const bytes = new Uint8Array(pcm16.buffer);
    this.session?.appendAudio(_bufferToBase64(bytes));
  }
}

// ------------------------------ util ----------------------------------------

function _rms(frame) {
  let sum = 0;
  for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
  return Math.sqrt(sum / Math.max(1, frame.length));
}

function _downsample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const at = i * ratio;
    const i0 = Math.floor(at);
    const i1 = Math.min(input.length - 1, i0 + 1);
    const t = at - i0;
    out[i] = input[i0] * (1 - t) + input[i1] * t;
  }
  return out;
}

function _bufferToBase64(bytes) {
  // btoa is UTF-16 aware; feed it a binary string in chunks to avoid stack blow-ups.
  let s = '';
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}
