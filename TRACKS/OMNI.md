# OMNI.md — Huawei OMNI Live track for Punching Face

Source of truth for the OMNI Live integration. Anyone touching the OMNI engine reads this first.

---

## Build status (last updated: 2026-09-19, key live)

**Plan A is live.** Team key delivered; smoke test picks A: a Realtime WebSocket
session against `wss://yibuapi.com/v1/realtime?model=qwen3.5-omni-plus-realtime`
with voice `Ethan`, streaming text + audio + `video_tokens` on every turn. The
final `response.done` reports 585 total tokens for a two-modality probe.

Confirmed facts from `.venv/bin/python scripts/omni_smoke_test.py`:

- **Enabled models**: `qwen3.5-omni-flash`, `qwen3.5-omni-plus`, `qwen3.5-omni-plus-realtime`, `qwen3.8-omni-flash`, `gemini-3.1-flash-live-preview`.
- **Accepted voices for this key**: `Ethan`, `Serena`, `Dylan` (verified). `Cherry` / `Chelsie` return 400 InvalidParameter on this gateway.
- **Voice cloning**: `/v1/audio/voices` returns 404 on yibuapi — no cloning. We ship the stock `Ethan` voice with emotion control instead.
- **Event ordering constraint**: within a turn, `input_audio_buffer.commit` must precede any `input_image_buffer.append`. The relay and client both respect this; the client primes with 40 ms of silent PCM16 for text-only turns.
- **Key expires 2026-09-20 08:00 EDT.** Report deadline: 2026-09-20 23:59 EDT.

**Usage logging is mandatory** and wired everywhere the API is called:
`omni_relay.py`, `scripts/omni_smoke_test.py`, and `sponsor_server.py` all
invoke `yibu_audit.append_audit_record` on every request (success and
failure). Ledger at `.local/usage/yibu_api_calls.jsonl`; run
`npm run omni:report` to produce `usage_summary.json` +
`usage_by_model_key_purpose.csv` for the organizers.

Both transports verified end-to-end 2026-09-19:

- Plan A: Realtime WS via `omni_relay.py`, 585-token turn, `video_tokens: 64`.
- Plan C: chat/completions via `sponsor_server.py`, 244-token turn at 644 ms.
- Ledger picked up both, plus one aborted-handshake failure — 10 rows total.

| Area | Status | Evidence |
| --- | --- | --- |
| Engine relay (`omni_relay.py`) | built, mock-verified | `tests/omni-relay-smoke.test.mjs`; `/health` at :5177 returns `plan: "mock"` without a key |
| Engine session (`src/omni/session.js`) | built | typed events, keep-warm, backoff |
| Engine capture (`src/omni/capture.js`) | built | Worker frames + AudioWorklet PCM16 mono 16 kHz |
| Engine events + tools + latency + fallback | built | `tests/omni-events.test.mjs`, `tests/omni-tools.test.mjs`, Ctrl-L overlay |
| Contact classifier (`src/contact/classifier.js`) | built | `tests/contact-classifier.test.mjs` (arenaRegions, strike debounce, press/release with rebound) |
| Cached reactions (`src/omni/reactions.js`) | synth fallback in code | `scripts/build_reactions.py` will replace with ElevenLabs once key set |
| Arena on shared engine (flagged) | built | `src/scenarios/arena/engine-wire.js`, off by default (flag `arena_omni=1`) |
| README / DEMO.md / perf.md / .env.example | shipped | see repo root |
| Baseline perf numbers | methodology in `docs/perf.md`; numbers pending a browser session | — |
| Smoke test result | **Plan A** | `.local/omni-smoke/report.json` — Realtime WS + tools + streamed usage |
| Voice cloning verification | not offered by yibuapi | `/v1/audio/voices` returns 404; we use stock voice `Ethan` |
| Usage audit ledger (challenge requirement) | writing on every call | `.local/usage/yibu_api_calls.jsonl` |
| Submission report | generate `npm run omni:report` and email at demo time | `.local/usage/summary/usage_summary.json` + `.csv` regenerated on demand |

## 0. Do this first (hour 0)

- [x] **Apply for API credits.** Team key delivered 2026-09-19. Never commit the key. Stored 0600 at `.local/secrets/omni.json` and `.env`.
- [x] **Smoke test the key.** `.venv/bin/python scripts/omni_smoke_test.py` picks Plan A. Report at `.local/omni-smoke/report.json`.
- [x] **Confirm gateway capabilities.** Realtime WS available. Voice cloning NOT available on yibuapi (`/audio/voices` 404). Accepted voices for this key: Ethan / Serena / Dylan.
- [x] **Log every call.** Sponsor package extracted to `.local/third_party/yibuapi-examples/`, so
  `yibu_audit.append_audit_record` now resolves instead of silently falling back to `None` — before that,
  every call would have gone unrecorded with no warning. Ledger: `.local/usage/yibu_api_calls.jsonl`.
  **The extraction is per-machine**: check `append_audit_record` is not `None` on any laptop that will make calls.
- [ ] **Submit report by 2026-09-20 23:59 EDT.** Attach `usage_summary.json` and `usage_by_model_key_purpose.csv` to a reply to the approval email.

---

## 1. Official track description

## Huawei — OMNI LIVE Challenge

OMNI Live: Build the Next Generation of Real-Time Multimodal AI

AI is moving beyond traditional text-based chat toward real-time, multimodal interaction. OMNI Live challenges hackers to imagine what becomes possible when AI can see, hear, speak, and interact with users in real time.

Build a functional AI application for a real-world edge-device scenario using an OMNI multimodal model. Your application may call the OMNI model through cloud APIs; the model does not need to run locally on the device. What matters is a compelling, responsive, complete multimodal experience.

Capabilities to explore:

- Vision and video understanding
- Speech and audio understanding
- Natural voice interaction
- Streaming and low-latency responses
- Multimodal contextual understanding
- Adaptive voice, tone, or expression
- Continuous or interruptible human–AI interaction

Full challenge details: <https://github.com/cari-waterloo-rc/OMNI-Live-Build-the-Next-Generation-of-Real-Time-Multimodal-AI>

---

## 2. Why Arena uses OMNI

Arena is a live boxing partner. Your hands are busy and your eyes are on the target, so **voice + vision is the only viable interface** during a round.

| Modality | Arena use |
| --- | --- |
| Vision (video frames) | Stance, dropped guard, telegraphed punches |
| Speech in (voice) | "hold on", trash-talk, breathing, questions between rounds |
| Speech out | Grunts, taunts, coaching callouts |
| Language | Opponent persona, round summary |

A chatbot can't see your guard drop. A vision-only model can't answer "how did I look on that combo?" OMNI's streaming, interruptible loop closes that gap.

---

## 3. Architecture

```
Browser ─►  frame worker (1–2 fps JPEG ~512 px) ─┐
        ─►  audio worklet (20 ms PCM16 mono 16 kHz) ─┤
                                                     ▼
Physics ─►  contact classifier ─►  event bus  ─►  OmniSession (WebSocket)
   │                                    │                │
   ▼                                    ▼                ├─► speech audio ─► player
cached reaction (<50 ms)   text context "[EVENT] …"      └─► tool calls ─► dispatcher ─► rig/UI

                                        key held ▼
                                  127.0.0.1:5177  (omni_relay.py)
                                        upstream ▼
                               wss://yibuapi.com/v1/realtime  (Qwen3.5-Omni)
                                          or
                                Plan C: sponsor_server.py /coach/turn
```

The **existing pipeline** (reconstruction, rigging, render, hand tracking, collision, deformation) runs on-device unchanged. OMNI runs beside it, never in its critical path. Frame encoding and mic capture live in a Worker and an AudioWorklet respectively; network I/O is async; the render loop is not touched.

---

## 4. Transport plans

- **Plan A** — Realtime WebSocket. Confirmed on yibuapi 2026-09-19. Streaming text + audio + `video_tokens`, tool calls, per-turn usage in `response.done`. `qwen3.5-omni-plus-realtime` + voice `Ethan`. No voice cloning on this gateway.
- **Plan B** — Gateway lacks Realtime but Alibaba's DashScope intl endpoint works. Same client code, endpoint swapped via `OMNI_REALTIME_URL`.
- **Plan C** — chat/completions per turn → ElevenLabs voice. The relay auto-detects this; the scenario keeps the same UI. Loses semantic interruption and streaming but keeps the loop functional.

If the OMNI session drops mid-round, the classifier keeps firing local physical reactions and cached audio while a small "reconnecting" badge appears; the client reconnects with exponential backoff.

---

## 5. OMNI capabilities used

- Streaming audio + image (frame) input, text + audio output, over WebSocket
- Semantic interruption / barge-in (the AudioCapture voice gate + `session.cancelResponse()`)
- Function calling (arena tool registry in `src/scenarios/arena/tools.js`)
- Voice control (emotion + style flow through the persona and instructions)
- Voice cloning (Plan A only; smoke-test verifies availability at the gateway — not offered by yibuapi)

Models: `qwen3.5-omni-plus-realtime` by default — it is on the sponsored key's enabled list and `qwen3.5-omni-flash-realtime` is not. Voice `Ethan`; `Cherry`/`Chelsie` return 400 on yibuapi.

---

## 6. Usage ledger (challenge requirement)

Every API call is recorded to a JSONL ledger at `.local/usage/yibu_api_calls.jsonl` — the smoke test, the relay's Realtime bridge, and the sponsor_server's Plan-C fallback all write to it via the canonical `yibu_audit.append_audit_record` writer (imported from the sponsor's example package under `.local/third_party/`).

Records include the call id, timestamps, model, key suffix (last 4 chars only, never the full key), purpose, endpoint, transport, ok/fail, latency, and token counts.

Before submitting, run `npm run omni:report`, inspect the outputs, and reply to the approval email with **only** `usage_summary.json` and `usage_by_model_key_purpose.csv` attached. Never include the full key, raw prompts, or the ledger itself.

---

## 7. Privacy and safety

- **Scan consent gate:** subject faces the camera and says a consent phrase; OMNI verifies face and phrase before scanning proceeds. One 30-second gate before any capture is stored.
- **Key custody:** the API key lives in `.env` or `.local/secrets/omni.json` (mode 0600) and never leaves the relay process. The browser never sees the key.
- **Session-scoped data:** meshes and any cloned voice are deleted when the tab closes. Only downscaled frames and audio ever leave the machine, and only while the route is active.

---

## 8. Hard constraints (reprise)

1. **Do not break or slow the existing pipeline.** Reconstruction, rigging, rendering, hand tracking, collision, deformation must perform exactly as before. OMNI runs beside them, never in their critical path.
2. **Measure before and after.** `docs/perf.md` holds baselines. Re-measure after each OMNI change; fix regressions before moving on.
3. **The existing route stays at `/`.** Don't rename it, don't move its assets. OMNI features are opt-in behind `?arena_omni=1` until proven.

---

## 9. Latency overlay

Press `Ctrl-L` in the app to toggle a fixed-corner overlay showing per-hop timings: contact→event, event→relay, contact→cached audio, last frame sent, speech end, first delta, response done, plan/mock/session id. `docs/perf.md` captures baselines and post-OMNI numbers side by side.
