# Punching Face — photo head reconstruction and contact prototype

A local Three.js head-model workflow: webcam photographs → recovered cameras → personalized face geometry → head/hair completion → photographic texture → facial controls → Newton soft-tissue contact. The active face pipeline does not train or render Gaussian splats.

**Arena** — a punching prototype at `/`. Reconstruction, rigging, rendering, hand tracking, collision and deformation perform exactly as before. The shared OMNI-driven sparring coach is opt-in behind `?arena_omni=1`.

The OMNI engine (Realtime WebSocket → Qwen3.5-Omni) is described under [OMNI integration](#omni-integration) below: architecture, capabilities, privacy, env vars, and run commands.

Run `npm run dev`, then open http://127.0.0.1:5173/. Vite runs on 5173, the capture/reconstruction API on 5174, the Newton CPU service on 5175, the OMNI relay on 5177. All bind to loopback.

## Capture and build

1. Choose **Record / upload head video**. Use **Record 360°** with the laptop webcam, or **Upload video** for MP4, WebM, or another format this browser can decode. Video import extracts up to 220 samples locally. The original recording and masked head PNG frames are saved in the private local scan folder. The scan panel includes a replayable source video and measured processing times; the original recording is not sent to the AI service.
2. Glasses can be reconstructed as a separate estimated accessory; removing them gives clearer eye and skin evidence. Keep a neutral expression, and keep hair, ears and chin inside the frame. Start facing the camera with your eyes open, looking toward the lens for a brief pause. Slowly turn the head and torso together through a full rotation, ending at the front. Alternatively, have someone film around a seated, still person with a phone. Avoid changing lenses/zoom or tilting the head. Live capture stops at 120 seconds or 240 accepted frames. **Stop & save** turns the camera off.
3. The recorder retains profile and rear images without inventing facial landmarks. These are labeled `head-only`, with unknown angle until camera reconstruction. At least 24 overlapping views and 12 tracked facial views are required. Front-only counts cannot be presented as 360-degree coverage. The build reports recovered angular span and registered rear views; “complete orbit” requires at least 300 degrees and three rear cameras.
4. **Create 3D face** recovers cameras with COLMAP, triangulates reliable facial landmarks, and fits a complete MakeHuman head template using a regularized global warp. Catmull-Clark subdivision preserves smooth facial loops, ears, skull and neck. A smooth hair envelope fits captured silhouettes. Reserved views check raw landmark reprojection, not independent ground truth or the accuracy of every template vertex.
5. Texture baking uses one frontal source for central features, bilinear sampling, smooth side transitions, exposure matching and depth visibility. Hidden mouth surfaces receive a neutral material. Eyeballs use their own iris, pupil, sclera and roughness textures, with a depth correction to keep them behind the fitted eyelids. Three.js uses a lit material for the fitted head so rotation and deformation change the visible shading. Template eyes, ear detail, mouth interior and unseen areas remain approximations.
6. Optional AI completion sends selected cropped views to `gpt-6-astra`. Its structured response supplies bounded posterior shape parameters, hair color/flow/hairline priors, and per-view eyeglass rim/bridge/temple contours. Those parameters are executed by the local mesh builder and cached by capture hash in `astra-head-completion.json`. The AI cannot displace the measured face or neck cut, and recovered rear views take precedence over posterior shape priors. Missing crown hair uses continuous Cartesian triplanar synthesis, with no spherical texture pole. Eyewear is lifted through the recovered frontal camera into independent rim, lens, bridge and temple meshes. Visible profile contours constrain the temple paths against the fitted head; bevelled acetate sections, tapered arms, hinge plates and clear lenses replace circular tubes; masked frame ink is inpainted out of the skin texture. Occluded skin, eyewear depth and temple fit remain estimates. If fewer than three rear views were recovered, it also attempts a rear appearance prediction using the image API. A recovered 360-degree scan uses its rear photographs. Measured geometry, fitting, texture baking and Newton physics run locally.
7. **Surface**, **Geometry** and **Wireframe** show the same editable mesh. Jaw, lip corner, brow and lid controls follow the facial anchors. The **3D glasses** checkbox shows or hides the rigid accessory. **Export GLB** exports the head, texture, four morphs and separate glasses meshes. **Save editable session** retains the rig, photo texture, glasses specification and Newton cage binding.

The uploaded `IMG_7496.MOV` reconstruction uses recovered rear photographs; it does not need an AI-generated rear reference. An earlier frontal-only scan used a labeled rear prediction from Codex's built-in image tool because the configured account returned a zero image-input allowance for GPT Image 2. Subsequent captures with insufficient rear views try the image API; if it is unavailable, the pipeline reports the failure and uses local material continuation from captured hair samples. It does not pretend that continuation is an AI-generated rear photograph. The image stage uses the unmodified Image Generation skill CLI at `~/.codex/skills/.system/imagegen/scripts/image_gen.py`; set `PUNCHING_FACE_IMAGE_CLI` to its location on another installation.

## Newton contact and facial rig

`newton_face.py` runs actual Newton 1.6 / Warp 1.17 on the CPU. A spaced facial cage drives three particle layers with tetrahedral FEM constraints: outer skin, soft tissue and a fixed inner support. Cheeks/lips, nose and forehead use different estimated stiffnesses. A kinematic spherical fist collider transfers local contact into the tissue. A backtracking safeguard prevents inverted tetrahedra. The dense rendering mesh receives barycentrically interpolated Newton displacements; UV seam copies share the same motion.

Use **Left hook** / **Right hook**, or **Q** / **E**, to test contacts. A landmark-fitted impact rig adds broad cheek compression, lateral mouth pull, jaw opening/shift and asymmetric eyelid squeeze over the local tissue response. The fields blend into the skull and neck so the full jaw silhouette follows the punch without a hard boundary at the Newton cage. These larger motions are expressive animation correctives, not displacements predicted by Newton. The reference/import preview uses the same rig over its surface springs.

**Hold peak deformation** freezes the coordinated pose for inspection; **Release deformation** resumes recovery. Turn off **Head recoil** to isolate surface deformation. **Slow motion** stretches the simulation timing for inspection. Softness and input speed scale the response; repeated hits blend within a bounded pose and return to the unchanged rest mesh. **Peak deformation** reports combined visible movement; Newton's own contact measurements remain separate in diagnostics. The renderer interpolates the slower CPU physics frames. The physics label reports the actual engine, and failures stop new contacts instead of silently switching to a different solver. Manual expression controls, exported morphs and saved sessions remain independent of the transient punch pose.

Facial controls and sculpt edits update the physics rest cage. This is a visual prototype with estimated tissue layers/materials, a coarse contact collider and heuristic landmark expression fields. It is not a measured fascial/muscle anatomy model, calibrated injury simulation, or a validated prediction of a real punch. Photographic lighting and lens reflections may remain in the texture. The eyeglasses attach rigidly to the head and are not a simulated breakable object; Newton currently solves facial tissue contact only. Rear appearance is measured only when rear camera views are recovered. Individual hair strands, internal anatomy and hidden ear detail are not measured. A nominal 20 cm hairline-to-chin height sets scale.

## Installation and checks

```sh
npm install
python3.9 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3.13 -m venv .local/newton-env
.local/newton-env/bin/python -m pip install -r requirements-newton.txt
npm run dev
```

MediaPipe/WASM assets are under `public/`. The head mask uses the official MediaPipe multiclass selfie segmenter, anchored to the initial frontal head box so rear frames do not depend on face detection. `scripts/setup_assets.py` can restore public models; it also contains legacy asset setup. The old Gaussian research utilities remain on disk for provenance but are not invoked by the active face builder.

```sh
npm test
npm run build
.venv/bin/python tests/face_pipeline_test.py
.venv/bin/python tests/head_template_test.py
.venv/bin/python tests/head_completion_test.py
.venv/bin/python tests/eye_detail_test.py
.local/newton-env/bin/python tests/newton_test.py .local/face-captures/CAPTURE_ID
```

The Newton regression checks actual rest stability, localized cheek/lip/nose displacement, positive tetrahedral volumes and recovery. Browser QA must also inspect front, side, rear, wireframe, a held contact and resumed recovery. The legacy spring tests only cover reference/import preview meshes; they are not Newton verification.

## Storage and other prototype features

Personal images and generated outputs are in `.local/face-captures/<id>/`, excluded from Vite direct filesystem serving. The server-side API key lives in a mode-0600 file under `.local/secrets/`; it is never returned to the browser or included in exports. Each accepted capture frame is saved immediately. **Delete scan** stops its reconstruction worker and removes that scan's local images and outputs. Independent GLB/session exports are separate files.

The webcam can drive estimated body/hand motion. Personal arm reconstruction remains a separate legacy experiment; no successful personal arm mesh is bundled. Demo hand shapes are hidden unless explicitly enabled. Room context currently accepts a panorama; the studio is a placeholder and a panorama does not supply translational depth.

Research dependencies: [Newton](https://github.com/newton-physics/newton), [Newton CPU installation](https://newton-physics.github.io/newton/latest/guide/installation.html), [COLMAP](https://colmap.github.io/), [MediaPipe Image Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter), [Astra model capabilities](https://developers.openai.com/api/docs/models/gpt-6-astra), [OpenAI image API](https://developers.openai.com/api/docs/guides/image-generation). The public Lee Perry-Smith reference has its attribution in `public/reference/LICENSE.txt`.

## Head template provenance

`public/head-template/base.obj` is the MakeHuman hm08 base mesh distributed as CC0 by [Anny](https://github.com/naver/anny/tree/main/src/anny/data/mpfb2), with the original asset header and license retained. `scripts/prepare_head_template.py` extracts its head/neck quads. `anchors.json` records approximate semantic template landmarks calibrated against a frontal rendering; this is a registration prior, not a learned identity model. The fitted head adds eye surfaces and a planar neck closure. It preserves smooth template structure instead of forcing a dense surface through every noisy image landmark.

Video import was checked with the user's 24.3-second HEVC `IMG_7496.MOV`: 51 masked head frames, 40 registered cameras, 236.4 degrees of recovered coverage, and 7 registered rear views. The video retraces part of its orbit, so uncovered regions remain estimates. The new watertight mesh has 19,799 vertices and 38,650 triangles. Actual Newton checks passed for localized cheek, lip and nose contact, positive tetrahedral volumes, recovery, and held-pose resume. An earlier 18-second import fixture accepted 41 views; its test-only scan was deleted.

## Hair and eyewear detail

Astra classifies scalp hair type and style and estimates top/side lengths, curls, density and flow. The normal capture result preserves the photographed hairstyle and fitted silhouette. Local image gradients provide strand directions and colors for a separate layer of sub-centimeter detail; unseen or ambiguous roots do not receive invented strands. These roots are attached to head triangles, so surface edits and head motion carry them. Selecting another hair type explicitly switches to an approximate procedural style preview. This is editable geometry, not a reconstruction of every individual hair.

The color atlas is 3072 × 3072. Photographic skin, eyebrows and hair are retained. The eye stage checks up to four front-facing moments for eye opening, native iris resolution (at least 24 pixels across), sharpness, glare and separable iris/pupil contrast. Imported recordings are decoded at their original resolution with rotation metadata; future captures retain the ten iris landmarks separately from the 468-point face cage. Reliable visible iris pixels are reused; occluded iris areas, sclera and eye geometry remain estimates. With AI completion enabled, `gpt-6-astra` evaluates the crops and generates bounded eye material parameters when detail cannot be recovered. Local code turns those parameters into iris fibers, pupils, limbal rings, sclera and highlights. This is an Astra-directed procedural material, not an AI photograph or a measured biometric iris map. `eye-detail.json` records per-eye provenance and rejection reasons, and the viewer explicitly labels generated eyes. An unavailable/disabled Astra uses a labeled generic fallback; it never claims a successful AI generation. Eye color and roughness survive session saves and GLB export/reimport. Opaque eyeglass rim contours receive narrow local cleanup; glasses remain separate frame, bridge, temple and lens meshes with their own sampled material. Clear-lens reflections or tint can still remain in photographed skin: a glasses-free reference is needed to verify the hidden appearance. The app does not claim complete optical removal or hyperrealistic recovery from missing observations.

Hair, glasses, controls, visibility and photographic material settings survive session saves and GLB export/reimport. GLB stores hair geometry and root bindings in extras. The accessories follow the head; they are not a separate hair-collision or breakable-glasses simulation.

Focused checks: `node --test tests/head-hair.test.mjs tests/head-accessories.test.mjs tests/appearance.test.mjs` and `.venv/bin/python tests/hair_eyewear_test.py`.

## Measured video-to-model timing

The scan panel's **Video to model** card persists video duration, extraction/import time, every reconstruction stage, the first model/physics load, and earlier attempts. Processing totals exclude recording duration and user idle time. Failed or partial runs are labeled and never shown as a completed end-to-end result. Rebuilding the same scan measures a new attempt; existing camera and AI caches can make a rebuild faster than a new recording.

A fresh successful run of the 24.3-second `IMG_7496.MOV` took **3m 0.6s** on this Mac with AI completion enabled. See [PIPELINE_BENCHMARK.md](PIPELINE_BENCHMARK.md) for the stage breakdown and the earlier failed attempt. Both camera reconstruction and AI annotations were recomputed for that successful run.

Source videos are stored as `.local/face-captures/<id>/source-video` and replayed through a loopback-only API with byte-range seeking. `source.json` stores the filename/duration/import measurement; `timing.json` stores reconstruction stages and attempts. These metadata files do not invalidate the image/camera cache hash. Deleting a scan also deletes its retained original video.

## OMNI integration

The full strategy lives in [`TRACKS/OMNI.md`](TRACKS/OMNI.md). This section is the shipped summary: what runs, why, and how to work with it.

### Why multimodal for Arena

| Modality | Arena |
| --- | --- |
| Vision (video frames) | Stance, dropped guard, telegraphed punches |
| Speech in (voice) | "hold on", trash-talk, breathing |
| Speech out | Grunts, taunts, coaching callouts |
| Language | Opponent persona, round summary |

Your hands are busy and your eyes are on the target, so **voice + vision is the only possible interface** during a live round. A chatbot can't see your guard drop; a vision-only model can't answer "how did I look on that combo?"

### Architecture

```
Browser  ──►  frame worker (1–2 fps JPEG ~512 px) ─┐
        ──►  audio worklet (20 ms PCM16 mono 16 kHz) ─┤
                                                       ▼
Physics  ──►  contact classifier ──►  event bus  ──►  OmniSession (WebSocket)
   │                                       │                  │
   ▼                                       ▼                  ├──► speech audio ──► player
cached reaction (< 50 ms)      text context "[EVENT] …"        └──► tool calls ──► dispatcher ──► rig/UI

                                              key held ▼
                                        127.0.0.1:5177 (omni_relay.py)
                                              upstream ▼
                                     wss://<gateway>/…realtime  (Qwen3.5-Omni)
                                                or
                                       Plan C: sponsor_server.py /coach/turn
```

The **existing pipeline** (reconstruction, rigging, render, hand tracking, collision, deformation) runs on-device unchanged. OMNI runs beside it, never in its critical path. Frame encoding and mic capture live in a worker and an AudioWorklet respectively; network I/O is async; the render loop is not touched.

### OMNI capabilities used

- Streaming audio + image (frame) input, text + audio output, over WebSocket
- Semantic interruption / barge-in (the AudioCapture voice gate + `session.cancelResponse()`)
- Function calling (per-scenario tool registries in `src/scenarios/*/tools.js`)
- Voice control (emotion + style flow through the persona and instructions)
- Voice cloning (Plan A; smoke-test verifies availability at the gateway)

Models: `qwen3.5-omni-flash-realtime` by default; swap to `qwen3.5-omni-plus-realtime` via `OMNI_REALTIME_MODEL` when latency headroom allows.

### Privacy and safety

- **Scan consent gate:** subject faces the camera and says a consent phrase; OMNI verifies face and phrase before scanning proceeds. Uses vision + voice in one 30 second gate.
- **Key custody:** the API key lives in `.env` or `.local/secrets/omni.json` (mode 0600) and never leaves the relay process. The browser never sees the key.
- **Session-scoped data:** meshes and any cloned voice are deleted when the tab closes. Only downscaled frames and audio ever leave the machine, and only while the route is active.

### Setup

```sh
npm install
python3.9 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3.13 -m venv .local/newton-env
.local/newton-env/bin/python -m pip install -r requirements-newton.txt

cp .env.example .env      # then fill in OMNI_API_KEY and any endpoints/models
npm run dev
```

`npm run dev` starts Vite (5173), the reconstruction API (5174), the physics service (5175), and the OMNI relay (5177) in one process supervisor.

### Environment variables

See [`.env.example`](.env.example) for the canonical list. Key ones:

- `OMNI_ENABLED` — master switch (default `true`). Set `false` to keep the routes running with OMNI off.
- `OMNI_API_KEY` — required for Plan A/B/C. Apply at https://luma.com/0fhypcu0.
- `OMNI_BASE_URL` — chat/completions gateway (Plan C).
- `OMNI_REALTIME_URL` — WebSocket endpoint (Plan A/B). Confirm with the smoke test.
- `OMNI_MODEL`, `OMNI_REALTIME_MODEL`, `OMNI_VOICE`.
- `OMNI_CLONED_VOICE_ID` — set once cloning is registered.
- `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` — fallback voice + `scripts/build_reactions.py`.

Precedence: process env > `.env` > `.local/secrets/omni.json`. Keys never ship to the browser.

### Run commands

- `npm run dev` — full stack including the OMNI relay.
- `npm run sponsors` — sponsor_server only (Cornerman/Arena chat path stays on 5176).
- `npm run omni:smoke` — smoke test; picks Plan A/B/C, writes to `.local/omni-smoke/report.json`. Every call it makes goes into the audit ledger.
- `npm run omni:relay` — the relay standalone.
- `npm run omni:report` — turn the audit ledger into `usage_summary.json` + `usage_by_model_key_purpose.csv` under `.local/usage/summary/`. **Required for submission.**
- `.venv/bin/python scripts/build_reactions.py` — pre-generate ElevenLabs cached reactions (fallback voice, not the OMNI voice).
- `node --test tests/*.test.mjs` — full JS test suite (OMNI engine + relay end-to-end + existing pipeline).

### Usage reporting (Huawei OMNI Live challenge)

Per the sponsor's rules, every API call the app makes is recorded to a JSONL
ledger at `.local/usage/yibu_api_calls.jsonl` — the smoke test, the relay's
Realtime bridge, and the sponsor_server's Plan-C fallback all write to it via
the canonical `yibu_audit.append_audit_record` writer (imported from the
sponsor's example package under `.local/third_party/`). Records include the
call id, timestamps, model, key suffix (last 4 chars only, never the full
key), purpose, endpoint, transport, ok/fail, latency, and token counts.

Before submitting, run `npm run omni:report`, inspect the outputs, and reply
to the approval email with **only** `usage_summary.json` and
`usage_by_model_key_purpose.csv` attached. Never include the full key, raw
prompts, or the ledger itself in the reply.

### Latency overlay

Press `Ctrl-L` on either route to toggle a small fixed-corner overlay showing per-hop timings: contact→event, event→relay, contact→cached audio, last frame sent, speech end, first delta, response done, plan/mock/session id. `docs/perf.md` captures baselines and post-OMNI numbers side by side.

### Fallback plans

- **Plan A** — the gateway supports Realtime WebSocket (**confirmed on yibuapi 2026-09-19**). Streaming text + audio + `video_tokens`, tool calls, and per-turn usage in `response.done`. `qwen3.5-omni-plus-realtime` + voice `Ethan`. No voice cloning on this gateway.
- **Plan B** — the gateway lacks Realtime but Alibaba's DashScope intl endpoint works. Same code path as Plan A with the endpoint swapped.
- **Plan C** — chat/completions per turn → ElevenLabs voice. The relay auto-detects this; the scenarios keep the same UI. Loses semantic interruption and streaming but keeps the loop functional.

If the OMNI session drops mid-encounter, the classifier keeps firing local physical reactions and cached audio while a small "reconnecting" badge appears; the client reconnects with exponential backoff.

### Limitations

- Voice cloning depends on the gateway's `/audio/voices` shape; the smoke test verifies. If unavailable we ship a stock OMNI voice with emotion control.
- The round summary is heuristic today; when OMNI is live the model's tool calls override the heuristic, and at least one summary line is judged from vision.

### Future work

- Multi-participant rounds (each boxer is a separate LiveKit participant; the coach addresses them by name).
- Headset support (WebXR passthrough; the scene coords are already metric).

## Demo scripts and rehearsal

See [`DEMO.md`](DEMO.md) for the 3-minute Arena script.
