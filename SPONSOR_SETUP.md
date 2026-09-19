# Sponsor features: setup, run, demo

Three additions to PUNCHING FACE, built Sat Sep 19 at Hack the North 2026. Strategy and rubrics: [SPONSOR_TRACKS.md](SPONSOR_TRACKS.md).

| Feature | Sponsor track | What it does |
|---|---|---|
| **Cornerman** | Huawei OMNI Live | A sparring coach that *sees* you (webcam keyframes), *hears* you (hands-free voice) and *speaks* back (streamed audio you can interrupt), using `qwen3.5-omni-flash` |
| **Arena** | (LiveKit) | Streams the live 3D head into a room. Anyone who joins can punch it from their own device; hits land on the real physics, with per-person scoring |
| **Flight recorder** | Sentry | One click reads as one trace: browser → `server.py` → pipeline subprocess → every pipeline stage, plus AI-call monitoring, structured logs and Session Replay |

**Evidence tags:** **[verified]** = I ran it here and observed the result · **[unverified]** = written but not exercised, with the reason.

---

## Quick start (everything on this computer, no accounts)

```bash
npm run dev            # the app, as before            → http://127.0.0.1:5173
npm run sponsors       # coach relay + LiveKit tokens  → http://127.0.0.1:5176 (loopback only)
npm run livekit:dev    # local LiveKit server (already installed: brew `livekit`)
```

A **Cornerman · Arena** pill appears bottom-right. With no keys at all you get: a clearly labelled **mock** coach, and a
fully working Arena where guests are other tabs on this computer (`/guest.html`). If `npm run sponsors` is not running,
the pill says so and the rest of PUNCHING FACE is untouched.

Tests: `npm test` (JS, includes `tests/sponsors*.test.mjs`) and `npm run test:sponsors` (Python).

---

## Keys: what you need, where it goes

Paste keys into the dock (**OMNI key** / **LiveKit keys** sections). They go to the loopback service, which writes
`.local/secrets/*.json` with mode 0600, exactly like the existing OpenAI key. **No key ever reaches the browser**, and
`.local/` is git-ignored. Environment variables override the files.

| Service | How to get it | Fields / env vars |
|---|---|---|
| **OMNI** | Huawei's form: <https://luma.com/0fhypcu0> (200 keys, first come; one per team; arrives by email) | `apiKey` / `OMNI_API_KEY`. Optional: `OMNI_MODEL` (default `qwen3.5-omni-flash`), `OMNI_BASE_URL` (default `https://yibuapi.com/v1`), `OMNI_VOICE` (default `Ethan`; `Cherry`/`Chelsie` are rejected by yibuapi) |
| **Sentry** | Create a project at sentry.io; copy its DSN. One DSN works for both; two projects (browser + Python) read better | `.local/secrets/sentry.json`: `{"browserDsn":"…","pythonDsn":"…"}` or `SENTRY_DSN_BROWSER` / `SENTRY_DSN`. Python side needs `pip install -r requirements-sponsors.txt` in **both** envs (already done on this machine) |
| **LiveKit** (only for other devices) | Free project at cloud.livekit.io → Settings → Keys. Also switch on **Development token server** and copy its id | `url` (`wss://…`), `apiKey`, `apiSecret`, `tokenServerId`, `guestUrl` |

Gateway facts **[verified from its public catalogue, 2026-09-19]**: it is an OpenAI-compatible `POST /v1/chat/completions`;
models served include `qwen3.5-omni-flash`, `qwen3.5-omni-plus`, `qwen3.5-omni-plus-realtime`. An invalid key returns
`401 {"error":{"message":"Invalid token","type":"new_api_error"}}`, which the relay surfaces in the coach panel.

After changing Sentry settings, restart `npm run dev` and `npm run sponsors` (Python reads the DSN at start-up).

---

## Guests on other devices (phones, other laptops)

Browsers only allow camera access on **https**, and an https page cannot open an insecure `ws://` socket. So other
devices need both of these; there is no shortcut that keeps the camera working:

1. **LiveKit Cloud** keys (above), giving a `wss://` URL.
2. **The guest page on any https static host.** It is a standalone 672 KB build that talks only to LiveKit:
   ```bash
   npm run build:guest        # → dist-guest/  (relative paths: works from any sub-folder)
   ```
   Upload `dist-guest/` anywhere static (Netlify Drop, Cloudflare Pages, GitHub Pages…), then put its URL in
   **LiveKit keys → guest page** as `https://…/guest.html`.

With a **token server id** set, the invite is one short link/QR for everyone. Without it, each link admits one guest
(a LiveKit identity lives inside its token, so a shared token would make each new guest evict the last): press
**Next guest** for another. LiveKit's own warning applies: the development token server is for development only, since
anyone holding the id can mint tokens for your project. Switch it off after the event.

> Do **not** tunnel the Vite dev server (ngrok and similar) to reach phones. It would expose the local capture API,
> including face photographs and key settings, to the internet. The guest page exists so that never has to happen.

---

## How it fits together

```
 guest device                         LiveKit                     host laptop (this repo)
 ───────────────                      ───────                     ─────────────────────────
 MediaPipe hands (on device) ─ punch events, ~60 bytes ─────────► arena-host.js → clamp + rate-limit
 pads (fallback, no camera)                                        → window.__punchingFace.remotePunch()
                                                                   → contact() → Newton physics
 <video> ◄──────────── 3D head, WebGL canvas @30 fps ◄──────────── canvas.captureStream()
 toast / vibrate ◄──── hit · zone · speed · scoreboard ◄────────── telemetry.js (who, where, how hard)
 coach voice ◄──────── audio track ◄────────────────────────────── cornerman.js ◄─ SSE ─ sponsor_server.py ─► OMNI
```

Perception runs on each device at ~30 Hz; only tiny events and one coach turn at a time cross the network. That is the
edge/cloud split the OMNI rubric rewards, and it is why a hit lands with data-channel latency rather than a video round trip.

**Files.** `sponsor_server.py` (relay, tokens, settings) · `sponsor_obs.py` (Sentry, all no-ops without a DSN) ·
`src/sponsors/` (`boot`, `cornerman`, `arena-host`, `guest`, `sentry`, plus pure, unit-tested `audio`, `telemetry`,
`punch-detect`, `sse`) · `guest.html` · `vite.guest.config.js`.

**Hooks into shared files** (each tiny, each asserted by `tests/sponsors-hook.test.mjs`; if that test fails, a hook
was lost in an edit: restore it rather than deleting the test):

| File | Hook |
|---|---|
| `index.html` | second `<script>` loading `src/sponsors/boot.js` |
| `src/main.js` | appended `window.__punchingFace.remotePunch` (the only way a remote hit reaches `contact()`) |
| `server.py`, `physics_server.py` | two lines before `serve_forever()` |
| `face_pipeline.py` | import + `env=trace_env()` on the pipeline `Popen` |
| `scripts/build_photo_face.py` | `__main__` wraps `run()` in the trace |

---

## Demo scripts

**Huawei OMNI Live (one complete scenario, ~90 s).** Start coach → ask aloud *"how's my guard?"* while your hands are
up (point: voice is the only input you have) → throw a three-punch combo: the coach speaks up **unprompted** with your
real speed and zone → talk over it to interrupt → untick *Let the coach see me* and show the badge change to
"voice + numbers only" (privacy) → have a teammate join the Arena and land a combo: the coach addresses them by name.
That last step is the "multi-device collaboration" line in the rubric.

**Sentry (needs ≥ 2 products beyond errors; five are wired).** Tracing (browser → API → subprocess → stages) ·
Logs (each stage logs with its trace id, so a log line opens its trace) · AI agent monitoring (`gen_ai.*` spans for
OMNI calls with tokens, first-token latency and frames sent; the SDK also auto-instruments the existing `openai`
calls, without prompts or images since `send_default_pii` is off) · Session Replay (browser) · Profiling (**Python
services only**: browser profiling needs a `Document-Policy` response header, i.e. a `vite.config.js` change, which
was deliberately not made while Codex is editing). **The story must be true:** open Sentry, find
one real thing, fix it, keep the before/after. Two honest leads seen while building this, *not yet investigated*: a
single `/physics/step` that took **628 ms** in one tab, and how long `/physics/open` takes after every page reload.

**LiveKit.** Go live → show the QR → two people join → both land hits, scoreboard splits by name → edit any source
file so the dev server reloads the host: the session **rejoins by itself** and the guests never leave.

---

## Privacy and safety

- **Coach:** nothing is sent until *Start coach*. Per turn: your utterance, at most four 320 px keyframes, and numbers.
  The badge states what is leaving and to which host; *Let the coach see me* off → audio and numbers only.
- **Arena:** the 3D head is always shared (that is the feature); your webcam only if ticked. Guests choose whether to
  share their camera; hand tracking stays on their device either way.
- **Sentry:** Session Replay blocks all media (so the webcam never appears), the 3D canvas is never recorded, inputs are
  masked, request bodies are never attached, and transaction names drop query strings. Asserted in
  `tests/sponsor_obs_test.py`.
- **Guests are untrusted:** every event is clamped, rate limited (6/s per person) and sent through the same `contact()`
  path as a real fist. The hook also sanitises its own input: a `NaN` speed once took the Newton session down.
- The coach prompt puts safety first (stop if winded, dizzy or in pain) and never encourages hitting a person.
  Scan only yourself, or use the built-in avatar.

---

## What is verified, and what is not

**[verified] here, in a browser, against a real local LiveKit server and the user's real head model:**
- Host goes live; a guest decoded the 3D head at full resolution (1540×1460; 627 frames; host encoder 479+ frames).
- Two simultaneous guests, distinct identities, nobody evicted; 5 interleaved hits attributed `Akash 3 · Mira 2`.
- Hits land through real physics at the right anatomy (`cheek-L`, `nose`, `cheek-R`, `chin`; ~18 mm deformation).
- Data-channel round trip 4–18 ms; hit feedback, scoreboard and "no contact" reasons reach the right guest.
- **Reload-safe:** host reload → rejoins under the same identity, no ghost participant, no duplicate video track,
  guests stay, video re-attaches. Guest reload → straight back in under the same identity.
- Coach loop in mock mode, including an **unprompted** cue triggered by a *remote guest's* combo, by name.
- Graceful fallbacks: camera blocked → pads; microphone blocked → typed questions; sponsor service down → offline pill.
- Tests: 55 JS pass, of which 10 are new (7 logic + 3 hook guards) · 9 relay/token tests · 6 Sentry-wiring tests
  (Python 3.9: 6/6; Python 3.13: 4 pass, 2 skip because the photo pipeline is not importable in the physics env).
- Builds: `npm run build` and `npm run build:guest`; LiveKit/Sentry/QR are lazy chunks.

**[unverified], and why:**
- **The real OMNI model.** No key yet. The request is built to the OpenAI-compatible Omni schema and unit-tested, and
  the endpoint was probed live, but whether the gateway accepts audio input, several images and streamed audio output
  exactly as sent is unknown. If it returns 400, the panel shows the gateway's message: adjust `omni_request()`.
- **Voice in, voice out, barge-in, echo control.** The voice gate is unit-tested (including a loud hall), but no
  microphone or speakers exist in the test browser.
- **Guest hand tracking on a phone.** The detector is unit-tested on synthetic landmarks; camera access was blocked in
  the test browser, so real punches were thrown with the pads. Thresholds may need tuning on a real device.
- **Anything against real Sentry / LiveKit Cloud.** No accounts were used. Sentry output was captured in memory.
- **Simulcast** was tried and could not be verified headlessly, so it is deliberately not shipped.

**Operational hazards for demo day:**
- The dev server reloads every open page on any source edit (including Codex's). Sessions now survive that, but a
  reload costs ~5–10 s while Newton re-opens; guests see "No contact: the head is still loading" meanwhile.
- Keep the host tab **visible**: browsers pause rendering in background tabs, which freezes the stream and the physics.
- The physics service allows 4 sessions; every open PUNCHING FACE tab holds one for up to 2 minutes after closing.
