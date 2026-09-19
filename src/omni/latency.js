// Latency overlay (OMNI.md §6.2). Timestamps every hop of the OMNI turn and
// renders a tiny fixed-position panel toggled by Ctrl-L. Judges score latency,
// so the overlay is loud, honest, and shows the numbers we can prove.
//
// Also exposed as `window.__omniLatency` for QA screenshots.

const KEYS = [
  ['contact.event', 'contact→event'],
  ['event.sent', 'event→relay'],
  ['reaction.audio', 'contact→cached'],
  ['frame.sent', 'last frame'],
  ['audio.commit', 'speech end'],
  ['response.requested', 'response asked'],
  ['response.first_delta', 'first delta'],
  ['response.done', 'response done'],
];

let singleton = null;

export function latencyOverlay() {
  if (!singleton) singleton = new LatencyOverlay();
  return singleton;
}

class LatencyOverlay {
  constructor() {
    this.marks = new Map();
    this.history = [];
    this._maxHistory = 50;
    this._el = null;
    this._visible = false;
    if (typeof window !== 'undefined') {
      window.addEventListener('keydown', (event) => {
        if (event.ctrlKey && (event.key === 'l' || event.key === 'L')) {
          event.preventDefault();
          this.toggle();
        }
      });
      window.__omniLatency = this;
    }
  }

  mark(key, at = performance.now()) {
    this.marks.set(key, at);
    if (key === 'contact.event') this._newRow(at);
    if (key === 'response.done') this._finalizeRow();
    // On the first text/audio delta after `response.requested`.
    if (key === 'text.delta' || key === 'audio.delta') {
      if (!this.marks.has('response.first_delta'))
        this.marks.set('response.first_delta', at);
    }
    if (this._visible) this._render();
  }

  _newRow(at) {
    this.history.push({ t: at, marks: {} });
    if (this.history.length > this._maxHistory) this.history.shift();
  }

  _finalizeRow() {
    const row = this.history[this.history.length - 1];
    if (!row) return;
    for (const [key] of KEYS) row.marks[key] = this.marks.get(key) ?? null;
  }

  toggle() {
    this._visible = !this._visible;
    if (this._visible && !this._el) this._mount();
    if (this._el) this._el.style.display = this._visible ? 'block' : 'none';
    this._render();
  }

  _mount() {
    const el = document.createElement('div');
    el.id = 'omni-latency-overlay';
    Object.assign(el.style, {
      position: 'fixed',
      right: '12px',
      bottom: '12px',
      zIndex: 9999,
      font: '11px/1.35 ui-monospace, monospace',
      color: '#d8e7cf',
      background: 'rgba(20,32,26,0.86)',
      border: '1px solid #46663a',
      padding: '10px 12px',
      borderRadius: '6px',
      minWidth: '260px',
      pointerEvents: 'none',
    });
    document.body.append(el);
    this._el = el;
  }

  _render() {
    if (!this._el) return;
    const row = this.history[this.history.length - 1];
    const lines = [
      /* HTML */ `<b>OMNI latency</b>
        <span style="opacity:.6"
          >(Ctrl-L to hide · ${this.history.length} turns)</span
        >`,
    ];
    if (!row) lines.push('<em>no turns yet</em>');
    else {
      const base = row.t;
      for (const [key, label] of KEYS) {
        const v = this.marks.get(key);
        if (v == null) continue;
        const rel = Math.round(v - base);
        const bar =
          '<span style="display:inline-block;width:' +
          Math.min(180, Math.max(2, rel / 2)) +
          'px;height:6px;background:#749b54;vertical-align:middle;margin:0 6px"></span>';
        lines.push(`${label.padEnd(14, ' ')}${bar}<span>${rel} ms</span>`);
      }
    }
    // Session state (plan, connection).
    if (typeof window !== 'undefined' && window.__omniSession) {
      const s = window.__omniSession;
      lines.push(
        /* HTML */ `<hr style="border:0;border-top:1px solid #34503a;margin:6px 0" />
          plan=<b>${s.plan}</b> · mock=${!!s.mock} · id=${s.sessionId || '-'}`,
      );
    }
    this._el.innerHTML = lines.join('<br>');
  }
}
