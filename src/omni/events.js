// Engine event bus → compact text context for OMNI.
//
// The bus is a scenario-neutral broadcaster of physical events (`press`,
// `release`, `strike`, `guard`, `consent`, `frame`, etc.). Every subscriber
// gets the same event object. A helper on top formats each event as ONE LINE
// of plain text so the model's context never balloons into JSON dumps.
//
// The line format is deliberately imperative and short so it plays well with
// the model's system prompt: `[EVENT] strike cheek-left force=55` — no key/value
// dumps, no commas.

export class EventBus extends EventTarget {
  constructor() {
    super();
    this._log = [];
    this._max = 200;
  }

  emit(event) {
    if (!event || typeof event !== 'object') return;
    const stamped = { ...event, t: event.t ?? performance.now() };
    this._log.push(stamped);
    if (this._log.length > this._max) this._log.splice(0, this._log.length - this._max);
    this.dispatchEvent(new CustomEvent('event', { detail: stamped }));
    this.dispatchEvent(new CustomEvent(stamped.type || 'unknown', { detail: stamped }));
  }

  history({ since = 0 } = {}) {
    return this._log.filter((e) => e.t >= since);
  }

  clear() {
    this._log = [];
  }
}

/**
 * Compact one-line summaries. Any new event type just adds a case.
 * Numeric fields are rounded so noise doesn't blow up the tokeniser.
 */
export function formatEvent(event) {
  if (!event) return '';
  const round = (v, d = 2) =>
    typeof v === 'number' && Number.isFinite(v) ? +v.toFixed(d) : null;
  switch (event.type) {
    case 'press': {
      const pressure = round(event.pressure, 2);
      return `[EVENT] press ${event.region}${pressure != null ? ` pressure=${pressure}` : ''}`;
    }
    case 'release': {
      const speed = round(event.speed, 2);
      const pain = event.painLevel != null ? ` pain=${event.painLevel}` : '';
      const reb = event.rebound ? ' rebound=true' : '';
      return `[EVENT] release ${event.region}${speed != null ? ` speed=${speed}` : ''}${pain}${reb}`;
    }
    case 'strike': {
      const force = round(event.force, 1);
      return `[EVENT] strike ${event.region}${force != null ? ` force=${force}` : ''}`;
    }
    case 'guard':
      return `[EVENT] guard ${event.state ? 'on' : 'off'}${event.region ? ` region=${event.region}` : ''}`;
    case 'consent':
      return `[EVENT] consent ${event.stage} ${event.granted ? 'granted' : 'denied'}`;
    case 'user.text':
      return `[USER] ${String(event.text || '').slice(0, 240)}`;
    case 'user.speaking':
      return '[USER] (speaking)';
    case 'user.silent':
      return '[USER] (silent)';
    case 'frame':
      return '[EVENT] frame';
    case 'session.start':
      return `[EVENT] session start${event.scenario ? ` scenario=${event.scenario}` : ''}`;
    case 'session.end':
      return `[EVENT] session end`;
    default:
      return `[EVENT] ${event.type || 'unknown'}`;
  }
}

/** Convenience: emit + push formatted context text into the model. */
export function relayEvent(session, bus, event) {
  bus.emit(event);
  const text = formatEvent(event);
  if (text && session) session.sendEngineEvent(event, text);
  return text;
}
