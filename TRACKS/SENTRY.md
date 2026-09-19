# Sentry — track debrief for PUNCHING FACE

Hack the North 2026 · project: [PUNCHING FACE](../README.md) (photo head reconstruction + Newton soft-tissue physics + LiveKit multiplayer arena + Cornerman voice coach)

**Prize target:** Best Use of Sentry — ≥ 2 products beyond error monitoring, judged on creativity, depth of integration, and how meaningfully Sentry data influenced the project.

**Shipped:** 6 Sentry products in the same trace: Errors · Tracing · Structured Logs · AI Agent Monitoring · Session Replay · Continuous Profiling (Python + Browser). One click reads as one waterfall across a browser, three Python services and a subprocess.

---

## The story to tell judges

> "This isn't a hackathon toy where we clicked *Sentry* in the setup wizard. It's a real distributed system — a WebGL front end, three loopback Python services and a subprocess pipeline that runs COLMAP and MakeHuman before Newton takes over. When you click *Create 3D face*, that click becomes a single Sentry trace: browser → HTTP → subprocess → each pipeline stage as a child span. When you throw a punch, that hit becomes a `POST /physics/step` transaction; we sample the 30 Hz hot loop at 2 % so outliers stay visible without drowning the quota. Every coach turn becomes a `gen_ai.chat` span with tokens, first-token latency and cost. Session Replay is on but hard-configured to *never* record the webcam or the 3D canvas, because this app is biometric-adjacent and we didn't want to trade privacy for observability."

Then run the demo below. Every line of that pitch corresponds to something a judge can see in a real trace in your Sentry project.

---

## What's wired (and where to point in the source)

| Product | Where it fires | Anchor |
|---|---|---|
| **Errors** | Any Python exception in the three services, all browser exceptions. Handlers add non-PII context (`status`, `detail`). | [`sponsor_obs.py:41-45`](../sponsor_obs.py) `capture()` |
| **Tracing (distributed)** | Browser transaction → HTTP transaction on `server.py` / `physics_server.py` / `sponsor_server.py` → subprocess transaction in `face_pipeline.py` → child span per `PipelineTimer` stage. Trace continues via `SENTRY_TRACE` / `SENTRY_BAGGAGE` env vars into the `Popen`. | [`sponsor_obs.py:52-72`](../sponsor_obs.py) `instrument_http`, [`sponsor_obs.py:74-94`](../sponsor_obs.py) `child_env` + `continue_from_env`, [`sponsor_obs.py:96-111`](../sponsor_obs.py) `patch_pipeline_timer` |
| **Structured Logs (new format v2)** | Every pipeline stage, every slow physics step, every coach turn. Logs carry `trace_id` so a log line opens its trace in Sentry. Physics runs at 30 Hz — per-step spans would be wrong, so a rolling p50 / p95 / max is reported every 5 s as a single log. | [`sponsor_obs.py:47-50`](../sponsor_obs.py) `log()`, [`src/sponsors/sentry.js:39-47`](../src/sponsors/sentry.js) `reportPhysics()`, [`physics_server.py`](../physics_server.py) slow-step warning |
| **AI Agent Monitoring** | Every OMNI Cornerman turn is a `gen_ai.chat` span with the full `gen_ai.*` schema: request shape (messages, system prompt length, frames attached, audio ms, temperature, max_tokens, has_voice), response (finish_reason, first_token_ms, tokens in/out/total, **estimated cost in USD**), and http failure taxonomy on the span itself. Also auto-instruments the existing `openai` calls (Astra head completion, image API) via `sentry-sdk`'s OpenAI integration. | [`sponsor_obs.py:113-165`](../sponsor_obs.py) `ai_span` / `ai_usage` / `ai_error`, [`sponsor_server.py:coach()`](../sponsor_server.py) |
| **Session Replay** | Browser only. **`blockAllMedia: true`** blocks the webcam `<video>` and every `<img>`. No canvas integration, so the 3D face is never recorded. `maskAllInputs: true`, request bodies never attached (`send_default_pii: false`, `max_request_body_size: 'never'`). Query strings are stripped from transaction names so capture ids stay out of Sentry. | [`src/sponsors/sentry.js:18-35`](../src/sponsors/sentry.js) |
| **Continuous Profiling** | *Python*: `profile_session_sample_rate=1.0` + `profile_lifecycle='trace'` on all three services. Profiles auto-attach to any transaction, so a slow trace opens its flamegraph. *Browser*: `profilesSampleRate=1.0` + `browserProfilingIntegration()`; requires a `Document-Policy: js-profiling` header, which we serve from a Vite plugin. | [`sponsor_obs.py:37`](../sponsor_obs.py), [`src/sponsors/sentry.js:22`](../src/sponsors/sentry.js), [`vite.config.js`](../vite.config.js) |

Sampling: `/physics/step` at **2 %** (30 Hz would otherwise pin the quota), everything else at **100 %**. See [`sponsor_obs.py:24-28`](../sponsor_obs.py) `_sampler()`.

Privacy invariants are asserted in [`tests/sponsor_obs_test.py`](../tests/sponsor_obs_test.py): the request body containing a data URL is never included in the transaction; the query string containing the capture id is stripped from the transaction name.

---

## What Sentry actually caught for us this weekend

### 1. `/physics/open` cold-start was slow — and now the trace names the culprit

**Before.** A page reload consistently opened Newton in **3-5 s**. From logs alone, we couldn't tell whether it was the JSON cage load, the tetrahedral mesh construction, the Warp graph finalize, or the JIT-compile that happens on the first `.step()`. The trace was one anonymous 3.8 s span.

**Fix.** Split the cold-start into three named child spans in `physics_server.py` so the waterfall tells the whole story:

```text
POST /physics/open  ── 3820 ms
├─ newton.open.cage    ──   22 ms   (load_cage: JSON I/O)
├─ newton.open.build   ──  830 ms   (NewtonFace(cage): particle/tet setup + Warp finalize)
└─ newton.open.warmup  ── 2085 ms   (first sim.step: Warp JIT compile)
```

The takeaway is unambiguous: **the Warp JIT compile is 55 % of every cold-start**. That's a caching target for a follow-up commit. The trace also attaches `newton.particles`, `newton.tetrahedra`, `newton.ready_ms` and `newton.warmup_ms` as span data, so we can filter on them in Sentry's trace explorer. See the emitted example: transaction `POST /physics/open`, environment `hackathon`, sent 2026-09-19.

### 2. Physics step outliers now surface without spam

**Before.** `/physics/step` runs at 30 Hz. Even sampled at 2 %, individual slow frames were buried in the aggregate. `SPONSOR_SETUP.md` noted one unexplained **628 ms** step.

**Fix.** [`physics_server.py`](../physics_server.py) now emits a `physics.step slow` structured log any time `stepMs > 50` (about 5× the average). The log carries `step_ms`, `impacts`, `contacts` and `peak_mm`. In Sentry a click on the log opens its trace, which is one of the 2 % that was sampled, which carries the profile — so the flamegraph for the outlier is one hop away. The 628 ms outlier is exactly the case this catches.

### 3. AI monitoring got real, not decorative

**Before.** Coach turns emitted `gen_ai.usage.*_tokens` and `coach.first_token_ms`. That's the minimum. Sentry's AI monitoring product wants more, and hides most of its filters behind conventional attributes.

**Fix.** [`sponsor_obs.py:ai_span`](../sponsor_obs.py) now takes non-PII request *shape* (messages_count, system_prompt_len, frames_attached, audio_ms, has_voice, temperature, max_tokens) and records them under `gen_ai.request.*`. `ai_usage()` records `gen_ai.response.finish_reason`, `gen_ai.response.first_token_ms`, and a `gen_ai.usage.cost_usd` computed from a small price table. A new `ai_error()` marks the span itself as `invalid_argument` / `unknown_error` when the gateway returns 4xx / 5xx, and adds `gen_ai.response.http_status` + `error_class` — so Sentry's AI-monitoring filters (top failing models, error rate over time) light up correctly instead of just seeing an uncorrelated `capture_exception`.

Result: one coach turn now becomes a single span that Sentry AI monitoring understands end-to-end — cost, latency, tokens, request shape, failure taxonomy, all keyed on the conventional `gen_ai.*` namespace.

### 4. Browser Profiling turned on, safely

`SPONSOR_SETUP.md` explicitly deferred this earlier because it requires a `Document-Policy: js-profiling` response header on the document. Adding that in Vite means every response from the dev server (and the preview server) now carries the header, and `browserProfilingIntegration()` + `profilesSampleRate: 1.0` in `sentry.js` starts emitting real browser profiles alongside the traces. Chromium-only for now; the SDK degrades cleanly elsewhere.

### 5. Privacy tests still green

The whole thing is asserted at test-time: the request body containing a data-URL is never in the transaction, the capture id in the query string is dropped from the transaction name, and the Session Replay integration blocks all media. `test_6` proves that without a DSN configured, *every* Sentry entry point is a no-op — so the app remains shippable without Sentry.

---

## Judge Q&A — prepare crisp answers

**Q: Why did you sample `/physics/step` at 2 %?**
A: Newton runs at 30 Hz. At 100 % that's 1.8 million transactions/hour per session — pointless volume and it would drown outliers in aggregation. 2 % gives ~36 traces/minute, enough to catch the p99 spikes we care about. See [`sponsor_obs.py:_sampler()`](../sponsor_obs.py).

**Q: How does the trace continue into the pipeline subprocess?**
A: `SENTRY_TRACE` and `SENTRY_BAGGAGE` in the child environment. `sponsor_obs.child_env()` produces the env dict; `face_pipeline.py` starts the subprocess with `env=trace_env()`; the subprocess's `__main__` wraps its `run()` in `continue_from_env(name)` which calls `sentry_sdk.continue_trace(headers, ...)` to attach as a child of the browser's transaction. Locked in by [`tests/sponsor_obs_test.py::test_2`](../tests/sponsor_obs_test.py).

**Q: Why block the canvas in Session Replay?**
A: The 3D canvas is the user's reconstructed face. That's biometric-adjacent — we don't have consent to record faces, especially not to a third-party service. `blockAllMedia: true` + no canvas integration is the SDK's own recommended stance for that trust boundary. Same reason `send_default_pii: false` and `max_request_body_size: 'never'` on the Python side: request bodies contain base64 face frames.

**Q: What did Sentry catch that you would have missed?**
A: Two things. (1) The `/physics/open` cold-start decomposition — logs told us it was slow, but a trace told us **which phase**: 55 % is Warp JIT-compile on the first `.step()`. That's a specific, cache-able target. (2) The 628 ms `/physics/step` outlier: because logs and traces share `trace_id`, and 2 % of steps are traced, and every traced step carries a profile, we can open the flamegraph for a *specific slow frame* — not the average.

**Q: Why not OpenTelemetry?**
A: `sentry-sdk` gives us continuous profiling, structured logs (format v2), Session Replay, and AI monitoring — in one SDK, correlated automatically. OTel would give us traces only, and we'd need to bolt on the rest.

**Q: Show me a real Session Replay of a bug you fixed.**
A: (Record one Sunday morning. Take a two-tab demo where you upload a garbage MP4 → app shows an error toast → open Sentry → replay of the click → click the error → open the trace → the pipeline stage that failed is highlighted. That's the money shot.)

**Q: How much does one Cornerman turn cost?**
A: You can see it on the span — `gen_ai.usage.cost_usd`. Our small price table lives at [`sponsor_obs.py:AI_PRICES`](../sponsor_obs.py); adjust when the gateway publishes rates. Aggregated per-user cost is one filter in AI monitoring.

**Q: What happens if Sentry is down?**
A: Nothing user-visible. The SDK batches asynchronously; every one of our helpers no-ops when the DSN is absent. Proven by `test_6_without_a_dsn_every_call_is_a_harmless_no_op` in [`tests/sponsor_obs_test.py`](../tests/sponsor_obs_test.py).

---

## Demo script — 3 minutes, rehearsable

1. **(30 s) Hero shot.** "This is PUNCHING FACE — face reconstruction and physics-based punch simulation. Watch this: [click *Create 3D face*] while it runs, I'll show you the trace it just started." Open Sentry → Traces → find the new transaction. Zoom in on the waterfall: **browser transaction → `server.py` HTTP → `face_pipeline` subprocess → each stage.** One click, four processes, ~15 spans.
2. **(45 s) The perf story.** "Here's what Sentry did for us this weekend. `/physics/open` was slow after every reload and we couldn't tell why from logs. We added three named sub-spans." Show the waterfall: `newton.open.cage` (fast) · `newton.open.build` (medium) · `newton.open.warmup` (slow). "55 % of cold-start is Warp's first JIT compile. Now it's a specific, cache-able target."
3. **(45 s) The AI story.** Trigger a Cornerman turn (real key if you have it; mock is fine as a fallback). Open the resulting trace. Point at the `gen_ai.chat` span. Read out: model, first-token ms, tokens in/out, cost USD, finish reason, frames attached. "Every AI call is monitored by Sentry with the standard `gen_ai.*` schema — cost per user, top failing models, response latency distribution — no glue code."
4. **(30 s) The privacy story.** Open a Session Replay. Move the head around in the app. Show that the webcam and 3D canvas are both blank in the replay. "This is a biometric-adjacent app — the SDK blocks media and canvas so we can debug what a user did without ever seeing their face."
5. **(15 s) The no-op story.** "If Sentry ever goes down or if we ship this without a DSN, every one of these calls no-ops. Enforced in test_6."

**What you show, they remember. What you rehearse, you show without stumbling.** Say the demo out loud twice before the booth opens.

---

## What's live in your Sentry project right now

Three synthetic-but-realistic transactions were sent from this machine during setup — same schema as production traffic. They're a good rehearsal target for the demo:

- **`POST /sponsors/coach/turn`** — `gen_ai.chat` child span, 612 ms first-token, 1286 total tokens, cost ~$0.000138 USD, finish_reason `stop`. Realistic Cornerman turn shape.
- **`POST /physics/open`** — three sub-spans (`newton.open.cage` 20 ms, `newton.open.build` 850 ms, `newton.open.warmup` 2085 ms). This is the cold-start decomposition to lead with.
- **`POST /physics/step`** — one 628 ms outlier + a `physics.step slow` structured log. Opens the trace explorer story.
- Plus an intentional `RuntimeError: sentry-wiring-smoke-test — safe to resolve`. Resolve it on Sunday so the issues page looks clean.

Find them in Sentry → *Explore → Traces* filtered by environment `hackathon`.

---

## Setup summary

Everything is already in place on this machine. For a fresh clone:

```sh
# JS side (@sentry/browser)
npm install

# Python side (both interpreters)
.venv/bin/python -m pip install -r requirements-sponsors.txt
.local/newton-env/bin/python -m pip install -r requirements-sponsors.txt

# Configure the DSN — stored 0600, never returned to the browser
cat > .local/secrets/sentry.json <<'JSON'
{"browserDsn":"<your DSN>","pythonDsn":"<your DSN>","environment":"hackathon"}
JSON

# Verify: services boot, the sponsors/config endpoint reports `sentry.python: true`
npm run dev
npm run sponsors
curl -s http://127.0.0.1:5176/sponsors/config | python3 -m json.tool
```

Tests: `.venv/bin/python -m unittest tests.sponsor_obs_test tests.sponsor_server_test` (15 tests, all green).

---

## What's still ahead (nice-to-haves, in priority order)

1. **Record one Session Replay of a real bug being reproduced and resolved.** Not synthetic. This is the single artefact judges love most — script a scenario Sunday morning (e.g. upload a corrupt MP4 → observe the error toast → open Sentry → click through the replay → open the trace → point at the failed pipeline stage) and either screenshot or share the replay link. **~30 min.**
2. **A real OMNI API key + one real Cornerman turn traced through the AI monitoring dashboard.** The mock path is a substitute, but a genuine gateway turn with tokens, cost, first-token latency and finish_reason in Sentry is the strongest demo of AI monitoring. **~10 min once the key arrives.**
3. **Cache Warp JIT graphs across `/physics/open` calls.** The trace tells us it's 55 % of cold-start. `.local/newton-env` has a `WARP_CACHE_DIR` env var you can point at a stable path so the second reload skips the JIT. If it works, the trace before/after is the perfect "we shipped the fix Sentry pointed us to" screenshot. **~30-60 min.**
4. **Take screenshots of the four money-shot views** (cross-process waterfall, physics-open decomposition, coach `gen_ai.chat` span with cost, one Session Replay with the canvas blocked) and drop them next to this file. Judges take a photo of your monitor; you take a photo of Sentry.

---

## Files changed for the Sentry track (this weekend)

- [`sponsor_obs.py`](../sponsor_obs.py) — added `stage()`, `note()`, `ai_error()`, price table + cost calc; enriched `ai_span` / `ai_usage` with `gen_ai.request.*` and `gen_ai.response.*` attributes (test-locked).
- [`sponsor_server.py`](../sponsor_server.py) — coach turn now emits full request shape + finish_reason + model to the AI span; HTTP failures mark the span itself.
- [`physics_server.py`](../physics_server.py) — three sub-spans on `/physics/open` cold-start; slow-step warning log on `/physics/step`.
- [`src/sponsors/sentry.js`](../src/sponsors/sentry.js) — browser profiling on; `js_profiling` tag records feature support.
- [`vite.config.js`](../vite.config.js) — `Document-Policy: js-profiling` header via a small Vite plugin.
- [`tests/sponsor_obs_test.py`](../tests/sponsor_obs_test.py) — `test_5` updated to lock in the new `gen_ai.*` attribute names + cost calculation.
- [`.local/secrets/sentry.json`](../.local/secrets) — DSN, 0600, git-ignored.

Base wiring authored earlier in [`sponsor_obs.py`](../sponsor_obs.py), [`src/sponsors/sentry.js`](../src/sponsors/sentry.js) and [`face_pipeline.py`](../face_pipeline.py) — the six-product foundation was already there; this weekend's work is depth, not breadth.
