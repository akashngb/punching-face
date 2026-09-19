# Sponsor-track plan: Hack the North 2026 (Sept 18–20)

Written Sat Sep 19 ~03:45 by Claude at the owner's request. Sources read that night: the HTN prizes page, the Huawei
OMNI Live challenge repo, the Baseten HTN starter repo. Requirements below are paraphrased: **re-read the originals
before you submit.** Anything marked *unverified* was not checked.

## One product, three sponsors, three different jobs

> **CONTACT: scan your own head, then spar with it, coached by an AI cornerman that watches, listens and talks.**

| Track | Its job in the product | State today |
|---|---|---|
| **OpenAI** (API + Codex) | Builds the *opponent*: capture director + bounded head-completion priors. Codex is the dev teammate | Mostly built at the event |
| **Huawei OMNI Live** | The cornerman's *eyes, ears and voice* (real-time vision + audio + language) | **Built** Sat ~05:00; runs in labelled mock mode until the API key arrives. See [SPONSOR_SETUP.md](SPONSOR_SETUP.md) |
| **Sentry** | The *flight recorder* across browser ↔ Python servers ↔ pipeline subprocesses ↔ AI calls | **Wired** Sat ~05:00; needs a DSN, then one real finding acted on |
| *(LiveKit Arena)* | Not a prize track: multiplayer. Streams the head to a room; guests punch it from their own devices | **Built and verified** locally; other devices need LiveKit Cloud + the deployed guest page |

Each sponsor gets a role nothing else in the stack duplicates. Judges spot checkbox integrations; this avoids them.
Keep the weird hook. The main HTN award favours playful, surprising projects, and "punch your own face" is that.

## Do these first (time-critical)

1. **Apply for the OMNI API key now**: <https://luma.com/0fhypcu0>. 200 keys, first come first served, one application
   per team, key arrives by email, 40 CAD limit, issued through the `yibuapi` gateway. Taking a key means you are
   expected to submit to that track.
2. Create a Sentry org + two projects (browser JS, Python).
3. Ask an organizer about pre-event code (see "Eligibility" at the bottom) before pitching anyone.

---

## Track 1: OpenAI API prizes

**Scored on two things:** how creatively and effectively the API powers the experience, and how meaningfully Codex
helped (planning, implementation, testing, debugging, iteration). The demo must show the working product, explain the
API's role, and give **one concrete way Codex improved the process or outcome**.

**Already true in this repo (all created at the event):**
- `openai_capture.py`: structured Responses call over ≤ 6 masked 512 px frames → `usableForMultiview`, `problems`,
  `nextCaptureInstruction`.
- `scripts/astra_photo_review.py`, `astra_face_review.py`: modelling priors (back-depth ratio, normal-offset limit,
  list of unobserved parts).
- `scripts/astra_head_completion.py`: bounded head-completion parameters **and accessory contours (the 3D glasses)**.
  Only unobserved skull vertices may move; the measured face stays fixed.

That is a genuinely non-chat use: a vision model supplies *bounded, labelled priors for what the camera never saw*.
The weakness is that none of it is visible in a demo.

**Tasks**
1. **Make it visible.** A "what the model decided" panel: capture problems + next instruction, and a head overlay
   coloured *measured / template / model-prior*. The data already exists in each job's `status.json` evidence.
2. **Move the capture review into the loop.** Run it every N accepted frames and show (or speak) the instruction while
   recording, instead of after. Capture quality is the measured cause of weak geometry (glasses, yaw coverage, blur).
3. *Optional, cheap:* natural-language control of the existing command box via tool calls (`setRig`, `setSoftness`,
   `hook`, `setView`, `export`). The `commands` map in `src/main.js` is the tool list.
4. **Collect Codex evidence now, not Sunday morning:** session transcripts, a 30 s screen clip of Codex working, test
   count over time, `AGENTS.md` as the handover mechanism.

**Candidate "one concrete way Codex helped" (all observable in this repo):** Codex built a synthetic fixture and quality
gates that *refuse* a bad mesh rather than fake one; when the Gaussian-splat route failed those gates it pivoted to the
photo-mesh + MakeHuman template pipeline within hours; `npm test` went from 20 passing tests to 40 in about two and a
half hours on Saturday night (measured 01:05 → 03:50), including one that went red and was fixed by Codex about ten
minutes later. Pick one and tell it with the artefact on screen.

**Be precise about division of labour if asked:** Codex wrote the application. `AGENTS.md`, `OPEN_SOURCE_STACK.md`,
`SPONSOR_TRACKS.md` and `scripts/setup_third_party.py` came from Claude as research hand-over.

---

## Track 2: Huawei OMNI Live

**Eligibility checklist (all required):** functional demo built at the event · uses an OMNI multimodal model (e.g.
Qwen3.5-Omni; cloud API is fine) · a specific real-world / edge-device scenario, not a generic chatbot · **vision/video
+ speech/audio + language all meaningfully used, together** · one complete end-to-end scenario · a repository with
setup/run instructions.

**Rubric:** scenario value & creativity 30 · use of OMNI capabilities 25 · demo completeness 20 · interaction experience
15 · technical implementation 10. Extra consideration for edge/cloud split, latency work, **privacy protection** and
**safety-aware design**. They explicitly prefer small-but-complete over ambitious-but-unfinished.

**Scenario: "Cornerman".** A sparring coach on the laptop you already own. The reason for multimodality writes itself:
**your hands are up and your eyes are on the target, so voice is the only possible input, and a coach that cannot see
you cannot coach.** "Was that hook better?" only makes sense to a model that saw the hook *and* heard the question.

**Architecture (edge → cloud):**
- *Edge, already built:* MediaPipe hand/pose tracking at ~30 Hz in workers; punch events, speed, hit region and guard
  state. Read them from `window.__contactLab.state` so `src/main.js` internals stay untouched.
- *Cloud, new:* per turn send the user's utterance (audio) + 2–4 downscaled keyframes from the last few seconds + a
  small JSON of recent punch telemetry → OMNI streams text + speech back.
- *Relay, new:* the key must never reach the browser. Mirror the OpenAI pattern (env var, else
  `.local/secrets/omni.json` mode 0600) in a new loopback service, e.g. `coach_relay.py` on 5176, allowing only the
  dev origin via CORS, so **`vite.config.js` does not need editing**.

**MVP loop (build this first, in new files such as `src/coach/`):**
1. Push-to-talk (or simple VAD) → record utterance.
2. Grab keyframes from the existing webcam `<video>`; attach telemetry.
3. Relay → OMNI → stream audio reply; play it.
4. Barge-in: stop playback the moment the user speaks.
5. Proactive turns on events (guard dropped > 2 s, combo landed, round end), so the coach speaks unprompted.

**Stretch:** tone adaptation (hype vs calm-down), spoken commands that act on the app ("softer target", "slow-mo
replay"), over-exertion / unsafe-form warnings (that is the "safety-aware" credit).

**Privacy credit, nearly free:** perception stays on-device; only sparse keyframes leave, behind an explicit toggle and
an on-screen indicator. Say so in the README.

**Unknowns (*unverified*):** what `yibuapi` actually exposes (plain streaming chat-completions vs a realtime
WebSocket), model slugs, audio formats, rate limits. **Design for plain streaming chat-completions** and treat anything
better as a bonus. Check the moment the key arrives.

**Framing:** lead with "sparring coach"; the personal head is the hook. Add a consent gate: scan only yourself, otherwise
use the built-in avatar.

---

## Track 3: Sentry

**Requirement:** use **at least two products beyond error monitoring** (Session Replay, Logs, Tracing, Profiling, Uptime,
MCP/AI-agent monitoring) and show how observability *shaped what you built*. Depth and influence are judged, not whether
the SDK is installed. Prize: guaranteed internship / new-grad interviews for every member.

**Why it fits honestly:** this project's defining problem was *silent failure*. Jobs died inside subprocesses and the UI
showed nothing. It is a four-process system (browser, `server.py` :5174, `physics_server.py` :5175, pipeline
subprocesses) plus external AI calls, which is exactly where distributed tracing earns its keep.

**Seams that already exist:**
- `pipeline_timing.py::PipelineTimer.mark(stage)`: one span per stage, almost for free.
- `face_pipeline.py::FaceStore.train()` spawns the subprocess: pass `sentry-trace` / `baggage` through env and
  `continue_trace` in the child so the browser click and the texture bake share one trace.
- `server.py` / `physics_server.py` are raw `http.server` handlers: continue the trace from request headers manually.
- `requirements.txt` already has `openai>=2`: Sentry's OpenAI integration gives AI monitoring (latency, tokens,
  failures) with no extra code. Add the OMNI relay calls as manual spans.

**Products to use:** Tracing (browser → `/api` → subprocess stages; `/physics/step`) · Logs (replace grepping
`pipeline.log`) · AI agent monitoring · Session Replay for the capture flow · optionally Profiling on the render loop.
`sentry-sdk` supports the Python 3.9 venv.

**Privacy:** Session Replay blocks media elements by default. Keep it that way so the webcam `<video>` is never
recorded, and do not enable canvas recording during personal scans.

**The story has to be true.** Instrument first, then act on one real finding and keep the before/after. Places worth
looking: the 30 Hz `/physics/step` JSON round trip, first-open Warp kernel compilation in `/physics/open`, and the
texture stage (a job was sitting in it while this was written). Do not decide the finding in advance.

---

## Alternates, and why they ranked lower

- **Cloudflare, "Best Agent with a Brain".** The best swap-in for Sentry *if a teammate knows Workers*: give the
  cornerman memory and agency (Durable Object per user for state + WebSocket hub, D1 for session history, tools such as
  `start_round` / `call_combo` / `slow_mo_replay`), and let the Worker double as the key-hiding relay. Roughly 4–6 h,
  and it shares a design with Track 2. Workers must be the real runtime; Pages alone does not qualify.
- **Baseten.** Tempting because the blocker here is "no CUDA" and they offer H100 workstations (ask at the booth). But
  the active pipeline no longer uses splats, so CUDA-only reconstruction would be a mid-hackathon re-architecture, and
  calling their LLM endpoint just to qualify would duplicate OpenAI/OMNI.
- **Backboard ("we judge ambition").** Plausible low-effort fourth if the coach's memory and LLM calls route through
  it, but it overlaps the Cloudflare role and rewards using *more* of their stack.
- **Everything else** (Solana/Thru, RBC, Federato, Intact, Zip, Shopify, Elastic, CSE, Dryft, QNX, Expo, the robot
  tracks, Composio, Linq, GPTZero, Tether, Devin): each would mean building a different project.

---

## Eligibility: sort this out before pitching

Facts from this repo's file creation dates: **35 source files were created on Thu Sep 17**, the day before the event
began (app shell, `src/physics.js`, hand tracking, arm capture, `server.py`, Poisson/LAM scripts). **48 were created
Sep 18–19 at the event**: the whole photo-head pipeline, all OpenAI integration, Newton physics, hair and glasses, most
tests. The git history is a single initial commit, so it does not show that split by itself.

Hackathons generally require work to be done at the event, and at least one track here (Intact) says so outright; I did
not find HTN's own rules text, so *unverified*. Ask an organizer, and state plainly in the submission what existed
beforehand. Everything in this plan would be new event work, which helps.

## Working agreement while Codex is live

Two agents and a teammate are editing one tree. Add **new files** (`src/coach/*`, `coach_relay.py`, a Sentry init
module) and leave one-line hooks for whoever owns `src/main.js`, `server.py` and `vite.config.js`. Commit often.
Never commit `.local/` (face photos and API keys live there; it is already ignored).
