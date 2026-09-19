// Incremental parser for the coach relay's event stream. fetch() delivers arbitrary byte
// boundaries, so an event may arrive split across reads; nothing is emitted until it is whole.
export class EventStream {
  constructor() {
    this.buffer = '';
  }

  feed(text) {
    this.buffer += text;
    const events = [];
    let cut;
    while ((cut = this.buffer.indexOf('\n\n')) >= 0) {
      const block = this.buffer.slice(0, cut);
      this.buffer = this.buffer.slice(cut + 2);
      let event = 'message',
        data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!data) continue;
      try {
        events.push({ event, data: JSON.parse(data) });
      } catch {
        /* a malformed event must not kill the stream */
      }
    }
    return events;
  }
}

// Arena data messages. Guests are untrusted: clamp everything before it touches the physics.
const clamp = (value, low, high) =>
  Math.max(low, Math.min(high, Number.isFinite(+value) ? +value : 0));

export function sanitizePunch(message) {
  if (!message || message.type !== 'punch') return null;
  return {
    u: clamp(message.u, -1, 1),
    v: clamp(message.v, -1, 1),
    lateral: clamp(message.lateral, -1, 1),
    speed: clamp(message.speed, 0.4, 4),
    side: message.side === 'left' ? 'left' : 'right',
    kind: message.kind === 'hook' ? 'hook' : 'straight',
  };
}

export const encode = (message) => new TextEncoder().encode(JSON.stringify(message));

export function decode(bytes) {
  try {
    const value = JSON.parse(new TextDecoder().decode(bytes));
    return value && typeof value === 'object' ? value : null;
  } catch {
    return null;
  }
}

// Per-guest rate limit: a real fist tops out near 4-5 punches a second; anything faster is a script.
export class RateLimit {
  constructor(perSecond = 6) {
    this.perSecond = perSecond;
    this.seen = new Map();
  }

  allow(id, now) {
    const recent = (this.seen.get(id) || []).filter((t) => now - t < 1000);
    if (recent.length >= this.perSecond) {
      this.seen.set(id, recent);
      return false;
    }
    recent.push(now);
    this.seen.set(id, recent);
    return true;
  }
}
