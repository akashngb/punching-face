// Audio plumbing for the OMNI coach. Pure functions, so they are unit-tested in Node.
// Up: mic Float32 -> 16 kHz mono PCM16 WAV (what Omni models accept as input_audio).
// Down: base64 PCM16 chunks at 24 kHz -> Float32 for gapless Web Audio scheduling.

export function downsample(samples, fromRate, toRate) {
  if (toRate >= fromRate) return Float32Array.from(samples);
  const ratio = fromRate / toRate,
    length = Math.floor(samples.length / ratio),
    out = new Float32Array(length);
  // Average each source window; a bare pick would alias speech sibilants into the band we keep.
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio),
      end = Math.min(samples.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += samples[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

export function encodeWav(samples, sampleRate) {
  const bytes = new Uint8Array(44 + samples.length * 2),
    view = new DataView(bytes.buffer);
  const text = (offset, value) => {
    for (let i = 0; i < value.length; i++) bytes[offset + i] = value.charCodeAt(i);
  };
  text(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return bytes;
}

export function bytesToBase64(bytes) {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000)
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(binary);
}

export function base64ToBytes(text) {
  const binary = atob(text),
    bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// Streamed chunks are raw PCM16, but a gateway may prepend a RIFF header to the first one.
export function pcm16ToFloat32(bytes) {
  const start =
    bytes.length > 44 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46
      ? 44
      : 0;
  const count = (bytes.length - start) >> 1,
    view = new DataView(bytes.buffer, bytes.byteOffset + start, count * 2),
    out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = view.getInt16(i * 2, true) / 0x8000;
  return out;
}

export const rms = (samples) => {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / Math.max(1, samples.length));
};

// Hands-free turn taking. Your fists are up, so there is no button to press: speech starts a
// turn and ~0.7 s of quiet ends it. Built for a loud hall: the noise floor is learned before the
// gate may open, falls quickly, rises slowly, and a drone that times a turn out lifts the floor
// to its own level, so steady noise costs at most one false turn instead of an endless loop.
export class VoiceGate {
  constructor({
    startMs = 120,
    endMs = 700,
    minMs = 350,
    maxMs = 9000,
    calibrateMs = 500,
    ratio = 3.2,
    floor = 0.004,
  } = {}) {
    Object.assign(this, {
      startMs,
      endMs,
      minMs,
      maxMs,
      calibrateMs,
      ratio,
      noise: floor,
      floor,
      speaking: false,
      loudMs: 0,
      quietMs: 0,
      spokenMs: 0,
      heardMs: 0,
      levels: [],
    });
  }

  // Returns 'start', 'end', 'discard' (too short to be speech) or null for each audio frame.
  push(level, frameMs) {
    if (this.heardMs < this.calibrateMs) {
      this.heardMs += frameMs;
      this.noise = Math.max(this.floor, this.noise * 0.8 + level * 0.2);
      return null;
    }
    const loud = level > Math.max(this.floor * 2, this.noise * this.ratio);
    if (!this.speaking) {
      // Fall fast, rise very slowly: a faster rise swallows the first syllable of real speech.
      this.noise = Math.max(
        this.floor,
        this.noise + (level - this.noise) * (level < this.noise ? 0.2 : 0.002),
      );
      this.loudMs = loud ? this.loudMs + frameMs : 0;
      if (this.loudMs >= this.startMs) {
        this.speaking = true;
        this.quietMs = 0;
        this.spokenMs = this.loudMs;
        this.levels = [];
        return 'start';
      }
      return null;
    }
    this.spokenMs += frameMs;
    this.quietMs = loud ? 0 : this.quietMs + frameMs;
    this.levels.push(level);
    const timedOut = this.spokenMs >= this.maxMs;
    if (this.quietMs >= this.endMs || timedOut) {
      const voiced = this.spokenMs - this.quietMs;
      // Speech dips between syllables, a drone does not: the turn's quiet fifth is the honest floor.
      if (timedOut) {
        const sorted = this.levels.slice().sort((a, b) => a - b);
        this.noise = Math.max(this.noise, sorted[Math.floor(sorted.length * 0.2)] || 0);
      }
      this.speaking = false;
      this.loudMs = 0;
      this.quietMs = 0;
      this.spokenMs = 0;
      this.levels = [];
      return voiced >= this.minMs ? 'end' : 'discard';
    }
    return null;
  }
}
