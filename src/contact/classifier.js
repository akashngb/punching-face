// Contact classifier (OMNI.md §5.1).
//
// Turns the existing per-frame contact stream (from main.js contact()) into
// typed events:
//
//   strike  {region, force}            — Arena: high velocity, short dwell
//
// Design constraints:
//   1. No new physics passes. We read the same collision data main.js already
//      computes. This module observes; it never re-computes contact geometry.
//   2. Stateful only in the smallest way: we track recent strikes per region
//      to debounce duplicate hits.
//   3. Region names come from the scenario's region map. The scenario passes
//      a `regionFromPoint(point)` closure that maps a local face coordinate
//      to a named region.
//
// Consumers: subscribe to `on('event', cb)`. Call `strike({...})` for
// discrete high-speed contacts. `observe()` / `releaseAt()` remain available
// for sustained-contact classifications a future scenario may need.

const RELEASE_THRESHOLD_MS = 120; // if no observe() for this long, treat as release
const PRESS_DWELL_MS = 180; // observe() this long before we call it a press
const STRIKE_MIN_SPEED = 1.5; // m/s
const REBOUND_WINDOW_MS = 350; // release speed above threshold within N ms of press-end

export class ContactClassifier {
  constructor({ regionFromPoint, onEvent }) {
    if (typeof regionFromPoint !== 'function')
      throw new Error('regionFromPoint required');
    this._regionFromPoint = regionFromPoint;
    this._onEvent = onEvent || (() => {});
    this._active = null; // {region, pressure, since, last, samples, pointAvg}
    this._lastStrikeAt = new Map();
    this._sweepTimer = null;
    if (typeof setInterval === 'function') {
      this._sweepTimer = setInterval(() => this._sweep(), 60);
      // Don't keep Node's event loop alive; browsers ignore .unref().
      if (typeof this._sweepTimer?.unref === 'function') this._sweepTimer.unref();
    }
  }

  dispose() {
    if (this._sweepTimer) clearInterval(this._sweepTimer);
    this._sweepTimer = null;
  }

  /**
   * A discrete high-speed impact — Arena's punch. Emits a strike event.
   * @param {{point:{x,y,z}, speed:number, region?:string, force?:number, mode?:string}} contact
   */
  strike(contact) {
    const region = contact.region ?? this._regionFromPoint(contact.point);
    if (!region) return null;
    const force =
      contact.force ?? Math.min(100, Math.max(0, (contact.speed || 0) * 32));
    // Debounce: two strikes on the same region within 80 ms are the same hit.
    // Use `has()` rather than a numeric sentinel — `performance.now()` starts
    // near zero in Node, which would falsely eat the very first strike.
    const now = performance.now();
    const last = this._lastStrikeAt.get(region);
    if (last !== undefined && now - last < 80) return null;
    this._lastStrikeAt.set(region, now);
    const event = {
      type: 'strike',
      region,
      force,
      speed: contact.speed || 0,
      mode: contact.mode || 'hook',
      t: now,
    };
    this._emit(event);
    return event;
  }

  /**
   * A per-frame sample: hand still on the mesh with a slow velocity. Called
   * every physics tick while the hand is in contact.
   * @param {{point:{x,y,z}, velocity:number, pressure?:number}} contact
   */
  observe(contact) {
    const now = performance.now();
    const region = this._regionFromPoint(contact.point);
    if (!region) {
      this._maybeRelease(now);
      return null;
    }
    const pressure =
      contact.pressure ??
      Math.min(
        1,
        Math.max(0.05, 0.15 + (contact.velocity ? 0.1 / contact.velocity : 0.35)),
      );
    if (!this._active || this._active.region !== region) {
      // Region change = end previous press, start a new press candidate.
      this._maybeRelease(now);
      this._active = {
        region,
        since: now,
        last: now,
        samples: 1,
        pressure,
        pointAvg: { ...contact.point },
        emittedPress: false,
      };
      return null;
    }
    this._active.last = now;
    this._active.samples++;
    this._active.pressure = 0.6 * this._active.pressure + 0.4 * pressure;
    this._active.pointAvg = _averagePoint(this._active.pointAvg, contact.point, 0.15);
    if (!this._active.emittedPress && now - this._active.since >= PRESS_DWELL_MS) {
      this._active.emittedPress = true;
      const event = {
        type: 'press',
        region: this._active.region,
        pressure: this._active.pressure,
        t: now,
      };
      this._emit(event);
      return event;
    }
    return null;
  }

  /**
   * Explicit end of contact (hand left the mesh). Optional; the sweep also
   * detects releases if the caller stops calling observe().
   * @param {{speed?:number, point?:{x,y,z}}} release
   */
  releaseAt({ speed = 0 } = {}) {
    this._maybeRelease(performance.now(), { speed });
  }

  _sweep() {
    if (!this._active) return;
    if (performance.now() - this._active.last > RELEASE_THRESHOLD_MS) {
      this._maybeRelease(performance.now());
    }
  }

  _maybeRelease(now, override) {
    const a = this._active;
    if (!a) return;
    this._active = null;
    if (!a.emittedPress) return; // we never registered a real press; nothing to release
    const releaseSpeed = override?.speed ?? 0.5;
    const rebound = releaseSpeed > 0.3 && now - a.last < REBOUND_WINDOW_MS;
    const event = {
      type: 'release',
      region: a.region,
      speed: releaseSpeed,
      rebound,
      t: now,
    };
    this._emit(event);
  }

  _emit(event) {
    try {
      this._onEvent(event);
    } catch (error) {
      console.warn('[classifier]', error);
    }
  }
}

function _averagePoint(prev, next, alpha) {
  return {
    x: prev.x * (1 - alpha) + next.x * alpha,
    y: prev.y * (1 - alpha) + next.y * alpha,
    z: prev.z * (1 - alpha) + next.z * alpha,
  };
}

// -------------------------- Region maps ------------------------------------
//
// Region maps live here so the tests can exercise them directly, separate
// from any single scenario's wiring.

/**
 * Arena regions: head, jaw, cheek-left, cheek-right, nose. `point` is in the
 * face pivot's local coordinates (metres) — same frame main.js uses for its
 * contact solver. Y is up, Z is out of the face.
 */
export function arenaRegions(point) {
  if (!point) return null;
  const { x, y, z } = point;
  if (Math.abs(z) > 0.14) return null;
  if (y < -0.055) return 'jaw';
  if (Math.abs(x) < 0.045 && y > -0.02) return 'nose';
  if (x < -0.028) return 'cheek-left';
  if (x > 0.028) return 'cheek-right';
  return 'head';
}
