# PUNCHING FACE — Design inventory

Read-only survey of every screen, element, event and asset a designer needs to redesign the app without reading the code. Compiled from `TRACKS/OMNI.md`, the HTML entry points, `src/**`, `public/**`, `sponsor_server.py`, and the Node tests. Where OMNI.md specifies something that isn't built yet, it is marked `[PLANNED]`. Anything present but unused or half-built is `[PARTIAL]`.

---

## 1. Product summary

PUNCHING FACE is a browser-based "spatial interaction lab" that turns a captured photo/video head into a real-time 3D face you can punch with tracked hands (webcam MediaPipe hand landmarks) or a remote guest phone (LiveKit data channel), with a soft-body physics rig deforming the mesh on contact. The existing `/` route ("Arena") is for hackathon demos and remote sparring; a planned `/clinic` route (an AI standardized patient a med student can interview and palpate) will share the same engine (OMNI Live session, contact classifier, tools, latency overlay).

---

## 2. Tech and platform constraints

### 2.1 UI framework & styling
- **No component framework.** Vanilla ES modules; UI is composed by writing template literals into `document.querySelector('#app').innerHTML` at boot (`src/main.js:20-53`).
- **Styling:** hand-written CSS. Main app in `src/style.css` (imported by `main.js`). Sponsor dock in `src/sponsors/sponsors.css`. Guest page in `src/sponsors/guest.css`. Legacy hair-review styles inline in `hair-review.html`.
- **Fonts (Google Fonts):** `DM Sans` (400/450/500/550/600/650/700), `Space Grotesk` (400/500/600/700). Imported by `src/style.css:1`.
- **3D:** Three.js `0.180.0` (renderer, OrbitControls, GLTFLoader, GLTFExporter). `@sparkjsdev/spark` for Gaussian splats. `@gltf-transform/*` for GLB compression.
- **CV / hand tracking:** `@mediapipe/tasks-vision 0.10.32` — `HandLandmarker`, `PoseLandmarker`, `FaceLandmarker`, `ImageSegmenter`. WASM at `public/wasm/`, models at `public/models/`.
- **Real-time:** `livekit-client 2.22.3` (host/guest arena), a local WebSocket relay (`omni_relay.py`) at `ws://127.0.0.1:5177/omni/realtime` for OMNI Live, sponsor SSE at `http://127.0.0.1:5176/sponsors/coach/turn`.
- **Observability:** `@sentry/browser 10.75.0` (loaded only when a DSN is configured).
- **Build:** Vite `8.3.0`. `vite.config.js` (main app on `127.0.0.1:5173`, proxies `/physics`→:5175, `/api`→:5174) and `vite.guest.config.js` (guest page → `dist-guest/`).

### 2.2 Target devices / viewports
- **Primary:** Chromium laptops with a webcam. `main` grid is `262px | minmax(360px,1fr) | 248px` (widens to `285 | 1fr | 272` above 1700 px, drops the right panel below 1100 px, and stacks vertically below 700 px). Landscape assumed; there is no explicit orientation lock.
- **Guest page (`/guest.html`):** phone-first, single column, dark theme (`#121a16`), uses `env(safe-area-inset-*)`, `100dvh`, `touch-action:manipulation`, `overscroll-behavior:none`. Optimised for one-handed portrait play.
- **AR/headset:** none. No WebXR. "AR" in OMNI.md refers to the illusion produced by the webcam + first-person view + tracked hands, not to a device session.
- **Support:** no polyfills for Safari/Firefox missing OffscreenCanvas or requestVideoFrameCallback — fallbacks are minimal (`setTimeout` when RVFC is missing).

### 2.3 Where UI renders
- **DOM overlaid on WebGL:** header, left/right panels, stage overlays (`.stage-top`, `.view-switch`, `.reticle`, `#impact-label`, `#slap-hud`, `.stage-bottom`, `#toast`, `#busy`, `#omni-latency-overlay`) sit above a `<canvas>` filled by Three.js. There is no on-camera overlay; the webcam preview is a small `<video>` inside `#slap-hud`.
- **In-scene 3D:** virtual hand skeletons (neon violet lines), scanned arm meshes (Gaussian splat or GLB), rig markers, wireframe, hair strands, glasses, studio walls/grid, panorama background.
- **Guest page:** DOM only. The host's live 3D head is received as a remote video track (`<video>` tag). The guest's own hands are tracked locally via MediaPipe.

### 2.4 What design MUST NOT change
- The `<canvas>` inside `#stage` (created by `renderer.setSize` on a ResizeObserver) — the render loop must own its bounds.
- Element IDs that are wired into event handlers by literal string (e.g. `#webcam`, `#stage`, `#slap-preview`, `#capture-dialog`, `#face-scan-dialog`, `#omni-latency-overlay`). Renaming an ID silently breaks handlers.
- The existing route stays at `/`. OMNI features are opt-in via `?arena_omni=1` or `localStorage['contact-sponsors-arena-omni']='1'` (`src/scenarios/arena/engine-wire.js:22-30`).
- OMNI capture and audio must stay off the main thread: frames in a Worker (`omni/capture.js`), audio in an AudioWorklet. This constrains any UI that wants to peek at the mic level or the OMNI frame stream — do it through events, not by re-reading the media stream.

### 2.5 Existing tokens & theme
Declared in `src/style.css:2`:
- `--line: #d9ddd3` (overridden later to `#e2e5dc`)
- `--muted: #748074` (overridden to `#7a8478`)
- `--green: #28634b`
- `--accent: #d2eea4`
Primary text `#222923` on `#eceee8` body, stage shell `#141d1c` (later overridden to `#f6f7f2`), panels `#f9faf5`, header `#f9faf5`, panel section headings `#4a5a48` uppercase Space Grotesk.
Guest page (`src/sponsors/guest.css:2`): same `--green`, `--accent`; adds `--line: #2b3a32` on a `#121a16` dark background.
Sponsor dock (`src/sponsors/sponsors.css`): reuses the same CSS variables. Live pulse `#d24b3a`, warn `#d9a441`.
Slap HUD (`src/style.css:16-66`) uses its own dark palette (`#0f1917`, `#c9f378` armed, `#ffca6a` uppercut, `#a9c8ff` jab).

---

## 3. Routes and user flows

The app is served through Vite. The "routes" are the HTML files at repo root, plus the sponsor dock inside `/`.

### 3.1 `/` — Arena (BUILT)
Entry: `index.html` → `src/main.js` (main app) + `src/sponsors/boot.js` (sponsor dock, added later on the same page).

Flow:
1. **Bootstrap.** Cold load; the app fetches `/api/face-captures`; picks the most recent complete photo model, otherwise the recovered session in `sessionStorage`, otherwise the packaged reference head (`public/reference/face-poisson.json`). During this, `#busy` overlay is shown with `#busy-text`.
2. **Idle scene.** Reference head visible on the studio backdrop. `#stage-status` shows `● DEMO INPUT`. Virtual hand skeletons idle at rest with `demoPose`. Q/E/Space run demo hooks.
3. **Capture guide (dialog).** Optional. User opens `#capture-dialog` via `Capture guide ↗`. Three cards: "Try your face now" (webcam portrait proxy), "Scan for fidelity" (opens the face-scan dialog), "Bring your own head" (GLB import).
4. **Face scan (dialog).** `#face-scan-dialog` (`src/face-capture.js`). Record 360°, upload video, or import photos → server-side reconstruction → poll `/api/face-status` → autoload when complete.
5. **Camera on.** User clicks `Connect laptop webcam`. Tracking worker starts; `#slap-hud` becomes active. Guard calibration required for arm-tracked hands; slap-detector works without calibration.
6. **Play loop.** Punch → `SlapDetector` classifies (hook/jab/uppercut) → `fireSlap` runs an impulse via `FaceDynamics` → mesh deforms → `#impact-label` flashes → contact counter increments → `#signal` trace updates.
7. **Fullscreen (immersive).** Header `⤢` toggles `body.immersive`; both side panels are hidden.
8. **Sponsor dock.** Loaded async from `src/sponsors/boot.js`; a floating `#sponsor-dock` bottom-right. Two tabs:
   - **Cornerman** — OMNI Live coach (see 3.1a).
   - **Arena** — LiveKit host (see 3.1b).
9. **Optional shared-engine Arena.** With `?arena_omni=1`, `src/scenarios/arena/engine-wire.js` also runs alongside Cornerman, using the OMNI relay's WebSocket transport, the contact classifier, cached reactions, and tool schemas. Purely additive; the Cornerman path is untouched.

Branches / error states:
- **Sponsor service down** → dock shows `Sponsor services are not running…` with a Retry button.
- **No OMNI key** → Cornerman badge reads `MOCK · no key`; replies are labelled `Mock reply (add an OMNI key for the real model)`.
- **Camera denied** → toast + `Camera unavailable: <name>`; tracking status text guides the user.
- **HMR reload** → `sessionStorage['contact-camera-wanted']` auto-reconnects the webcam; `sessionStorage['punching-face-arena-session']` rejoins the LiveKit room.
- **remotePunch hook missing** → sponsor Arena warns `main.js lost its remotePunch hook…` and falls back to clicking the on-screen L/R hook button.

### 3.1a Cornerman sub-flow (inside `/`)
`src/sponsors/cornerman.js`.
Idle → **Start coach** → mic calibration ("Learning the room noise…") → **Listening** → voice detected → **Thinking…** → text + audio stream (barge-in mid-sentence) → back to Listening. Auto-cues fire on combos / personal bests at natural beats. `Stop coach` returns everything to idle. Typed "Ask" input always works even without mic.

### 3.1b Arena sub-flow (inside `/`)
`src/sponsors/arena-host.js`.
Fill Name/Room → **Go live** → LiveKit token from `/sponsors/livekit/join` → publish the WebGL canvas as a video track → show invite QR + link → wait for guest joins → each remote data event is sanitized then applied via `window.__punchingFace.remotePunch` → hit flash on the corresponding participant tile → scoreboard update. **End session** disconnects.

### 3.2 `/guest.html` — Join the ring (BUILT)
Entry: `guest.html` → `src/sponsors/guest.js`.
Flow:
1. Invite link is `<host>/guest.html#r=<room>&d=<devServer>` (or `&u=<url>&t=<token>` for direct tokens).
2. **Join screen.** Name field, "Share my camera with the room" checkbox (default on), Join button. Errors surface in the note under the button.
3. **Ring.** Live 3D head video fills the stage. Small self-camera preview upper right. HUD with `Hits`, `Best`, `RTT`, Leave. Bottom pads `LEFT HOOK` / `JAB` / `RIGHT HOOK`. Scoreboard below.
4. On a landed hit: green inset flash, toast `<zone> · <speed> m/s`, vibration.
5. On disconnect: falls back to the Join screen with "Disconnected. The host may have ended the session."
6. **`&adaptive=0`** query keeps receiving video in a background tab (used for headless tests).

### 3.3 `/hair-review.html` — Hair reconstruction review (BUILT, standalone)
Entry: `hair-review.html` → `src/hair-review.js`. Requires `scripts/prepare_hair_review.py` to have generated `/generated/hair-review/*`. Shows source video frame vs. reconstructed hair render at recovered camera angles. Toggles: Hair fibers, Photo overlay, Inspect shape. Not part of Arena/Clinic; an internal review page.

### 3.4 `/head-template-preview.html` — Head template landmark preview (BUILT, standalone)
Entry: `head-template-preview.html` (inline script). Renders `head-template/head.json` or a scanned mesh (`?scan=<id>`), overlays FaceLandmarker output. Standalone tool; not linked from the main UI.

### 3.5 `/clinic` — AI Standardized Patient (PLANNED)
Not present in the codebase. OMNI.md's build-status table and cut-order describe it as a second scenario that will share the engine (`OmniSession`, `EventBus`, `ContactClassifier`, `ToolRegistry`, latency overlay, reactions), consuming a case file and rubric per patient. No route, no HTML entry, no `scenarios/clinic/` directory exists yet. All Clinic-specific items in this document are `[PLANNED]`.

---

## 4. Screen-by-screen inventory

### 4.1 Arena — main stage (`/`) — BUILT
File: `src/main.js` (`document.querySelector('#app').innerHTML=…` at line 20–53); styles in `src/style.css`.

**Purpose:** live target for tracked/demo/remote punches with a soft-body face rig and rich telemetry.

**Layout regions**
- Header (`<header>` height 70 px): brand svg + `PUNCHING FACE` wordmark; right side `#meshy-panel`, fullscreen icon, `Capture guide ↗`.
- Left panel (`.panel.left`, 262 px): sections **Face**, **Camera**, **Room**.
- Stage (`.stage-shell`): the WebGL canvas plus overlays (`.stage-top`, `.view-switch`, `.reticle`, `#impact-label`, `#slap-hud`, `.stage-bottom`, `#toast`, `#busy`).
- Right panel (`.panel.right`, 248 px): sections **Contact response**, **Surface & rig**, export row.

**Header elements**
- Brand SVG (inline path, stroke `#46663a`).
- `#meshy-toggle` — checkbox "Photo → 3D self". Effect: calls `/api/meshy-headshot` with a cropped face portrait, replaces the head with the Meshy import.
- `#compress-toggle` — checkbox "Compress GLB". Effect: run gltf-transform (`dedup, prune, weld, quantize`) on the incoming bytes, toast reports before/after KB.
- `#beat-yourself` — button. Rebuild/load Meshy head, then fire an uppercut + left hook + right hook combo (`fireSlap` scripted at 300 / 1100 / 1800 ms).
- `#fullscreen` — icon-btn `⤢`, `aria-label="Toggle fullscreen"`. Toggles `body.immersive`.
- `#capture-open` — opens `#capture-dialog`.

**Left panel — Face section**
- `#model-name` (`<strong>` heading) — dynamic label. Real: `Reference head`, `Your Meshy head`, `Your photo model`, `Your portrait preview`, `<filename>.glb`.
- `#model-kind` (`<small>`). Real examples: `Public mesh · legacy spring preview`, `Meshy AI head · textured`, `Photo proxy · estimated depth`, `Restored editable session`, `Imported textured mesh · preview physics`.
- `#scan-face` primary button — "Record / upload head video". Opens face scan dialog.
- `#import-face` — "↑ Upload GLB head". Triggers `#face-file` (accept `.glb,.json`).
- `#reference` — "↺ Reset to reference". Loads `/reference/face-poisson.json`.
- `#photo-count` (`<p class="muted">`) — e.g. `24 recovered photo views`, `Public reference mesh`, `Single photo preview`.
- `#mesh-count` — e.g. `Loading reconstruction…`, `31,842 triangles · editable`.

**Left panel — Camera section**
- `#camera` primary — toggles webcam. Text switches `Connect laptop webcam` ↔ `Disconnect webcam`.
- `#webcam` `<video playsinline muted>` — the tracked stream, `object-fit:cover; transform:scaleX(-1)`. Hidden until active.
- `#tracking-status` — dynamic text: `Connect your camera for tracking.`, `Starting local hand tracking…`, `Webcam connected · show your face and hands, then calibrate guard`, `Guard calibrated · close your fist and move across the target`, `Camera could not start. Allow camera access in the browser, then reconnect.`, etc.
- `#calibrate` (small, disabled until camera on) — "Calibrate guard position".
- `#scan-arms` (small) — "Scan my arms". Opens arm scan dialog.
- `#arm-appearance` muted — status about scanned arms: e.g. `Captured arms loaded. Calibrate with your face, shoulders, elbows and wrists visible.`, `Following your left and right arm. Capture proportions retained; pose and skin weights are estimates.`
- `Demo hand shapes` (`#demo-hands`) — checkbox, present but no handler in main.js (`[PARTIAL]`; the ID isn't wired to behavior).

**Left panel — Room section**
- `#import-room` (small) — "↑ Add room panorama". Triggers `#room-file` (accept `.jpg,.jpeg,.png`).
- `#room-label` — e.g. `Studio environment · placeholder`, `<filename>.jpg · panorama, rotation only`.
- `#room-fields` (hidden until a panorama loads): three range sliders and a Restore studio button.
  - `#room-yaw` (–180…180°), `#room-yaw-value` `0°`
  - `#room-scale` (0.2…3 step 0.01), `#room-scale-value` `1×`
  - `#room-height` (–3…3 m step 0.01), `#room-height-value` `0 m`
  - `#reset-room` small full "Restore studio".

**Stage overlays**
- `.stage-top` top row: `.scene-name` `#scene-name` (e.g. `Reference head <span>CPU Poisson surface</span>`, `Your Meshy head <span>Editable mesh</span>`); `#stage-status` (`● DEMO INPUT`, `● WEBCAM · VIRTUAL POV`).
- `.view-switch` centered pill: `#view-mesh` (Surface, active), `#view-clay` (Geometry), `#view-wire` (Wireframe).
- `.reticle` — a 13×13 CSS cross-hair at `50%, 46%` in colour `#c8e6b294`.
- `#impact-label` — flashes on every contact with `TRACKED CONTACT`, `DEMO CONTACT`, or `REMOTE · <NAME>` (also `LEFT HOOK`/`RIGHT HOOK`/`UPPERCUT`/`STRAIGHT` when slap-detector fires).
- `#slap-hud` (right-side floating panel; see 4.2).
- `.stage-bottom` — key-hint row (Q/E/Space/Drag) + `.bottom-line` (`#view-label` `FIRST-PERSON · VIRTUAL HANDS`/`FIRST-PERSON · CAPTURED ARM MESHES`; `#slap-status-line` `Connect webcam · palm toward camera = punch`).
- `#toast` — role=status, 5 s auto-hide. Real toast strings appear throughout `src/main.js` (e.g. `Panorama loaded. Use a 2:1 equirectangular image for a correct 360° view.`, `Photo model and Newton ready. Inspect Geometry or Wireframe, then try a hook.`).
- `#busy` — full-stage overlay with spinner + `#busy-text` + muted "Processed on this computer.". Text is set at each call, e.g. `Preparing surface…`, `Loading reference mesh…`, `Importing face asset…`, `Meshy is texturing your head (up to a few minutes)…`, `Compressing GLB with gltf-transform…`.

**Right panel — Contact response**
- `.metric-grid`: `#impacts` (padded to two digits, e.g. `00`, `07`, `12`) label "Contacts"; `#speed` label "Last speed" with `<em>m/s*</em>`.
- `#signal` canvas (400×100) — a rolling deformation trace, green line `#749b54` on grid `#d8dfd0`.
- `#compression` output — peak deformation in mm, e.g. `2.4 mm`.
- `#softness` range `0…1` step 0.01 default `.6`; `#softness-value` `60%`.
- `#distance` range `.35…0.8` m default `.55`; `#distance-value` `55 cm`.
- `#left-hook`, `#right-hook` small buttons.
- `#hold-peak` checkbox "Hold peak deformation".
- `#resume-impact` small "Release deformation".
- `#head-recoil` checkbox (checked by default) — enables recoil rotation.
- `#slow-motion` checkbox (checked) — physics time-scale ×0.38.
- `#region-readout` muted — e.g. `cheeks: 1.7 mm · lips: 0.4 mm · forehead: 0.1 mm · jaw: 0.0 mm · nose: 0.0 mm`.
- `#physics-engine` muted — e.g. `Preview springs + facial impact rig`, `Starting Newton CPU solver…`, `Newton CPU · 60 Hz + facial impact rig`.

**Right panel — Surface & rig**
- `#wire` checkbox (Wireframe), `#rig` checkbox (Rig markers), `#sculpt` checkbox (Sculpt mode).
- `#undo` / `#redo` small buttons.
- Sliders (each 0…1 step 0.01 with a `%` output):
  - `#jaw` "Jaw" → `#jaw-value`
  - `#smile` "Smile" → `#smile-value`
  - `#brow` "Brow raise" (nested inside Accessories) → no value output
  - `#squint` "Lid compression" → no value output
- `<details><summary>Accessories & alignment</summary>` block containing:
  - `#glasses` checkbox (disabled until a glasses spec is installed) — "3D glasses".
  - `#eye-detail-status` — dynamic status about eye detail.
  - `#hair-visible` checkbox (disabled until hair spec present).
  - Nested `#hair-controls` details "Hair style" with:
    - `#hair-description` muted, e.g. `Detected: quiff. 12,410 photo-guided fibers following captured locks. Individual fibers and unseen crown detail remain estimated.`
    - `#hair-type` select — options `straight, wavy, curly, coily, braided, locs`.
    - `#hair-length` range 1…450 mm step 1.
    - `#hair-curl` range 0…1 step 0.01.
    - `#hair-lift` range 0…12 mm step 0.1.
    - `#hair-frizz` range 0…1 step 0.01.
  - `#face-yaw` range –180…180.
  - `#face-pitch` range –180…180.
  - `#first-person` small full "Reset view".
  - `#impact-references` small full "Impact reference library". Opens `#reference-dialog`.
- Export/save row:
  - `#reset` small "Reset face".
  - `#export` small primary "Export GLB ↗".
  - `#save` small full "Save editable session".
  - `<details><summary>Reconstruction evidence</summary><div id="stats-detail">` — pretty-printed JSON of the current mesh's provenance stats.

**States**
- Empty (no camera): `#tracking-status` = `Connect your camera for tracking.`; slap HUD shows `Connect camera to see the detector view`; `#stage-status` = `● DEMO INPUT`.
- Loading: `#busy.active` with contextual message; `#mesh-count` = `Loading reconstruction…`.
- Success: mesh installed, `#view-mesh` selected; `#region-readout` = `Facial impact rig · ready`.
- Error: `toast(message)` for 5 s (`Camera unavailable: NotAllowedError`, `This GLB contains no meshes to import.`, `Use a GLB or editable session smaller than 180 MB.`, etc.).
- Immersive: `body.immersive`, both panels hidden.

**Transitions**
- Capture button → dialog. Dialog closes → back to stage with new mesh.
- Camera connect → webcam active state; disconnect → demo state.
- Any drop of `#stage` size → renderer `setSize` (ResizeObserver).

### 4.2 Slap HUD (in-stage overlay) — BUILT
Files: markup in `src/main.js:26-47`, styles in `src/style.css:16-66`.

**Purpose:** show the punch-detector's live inputs so the user can see why a swing was or wasn't accepted.

**Layout regions**
- Title bar: `PUNCH DETECTOR` + `#slap-state-badge` (states: `idle`, `searching`, `armed`, `watching`, `hit`).
- Detector preview (`.slap-view`): mirrored webcam `#slap-preview`; overlay guides (`.slap-strike-zone` left/right, `.slap-strike-uppercut` centered arc); SVG hand skeleton `#slap-view-hand` with polygon fill, path links, joint circles; `#slap-view-flash` (radial gradient per punch type); `.slap-view-legend` (`LEFT · UP · RIGHT`); `#slap-view-empty` fallback text.
- Body: four metric rows.

**Elements**
- `.slap-metric` "Approach" → `#slap-growth-fill` bar + `#slap-growth-mark` threshold + `#slap-growth-val` (e.g. `4.2/s`).
- "Rise" → `#slap-vy-fill` signed bar (rising = orange) + `#slap-vy-val` (`-1.7/s up`).
- "Palm size" → `#slap-width-fill` + `#slap-width-mark` + `#slap-width-val` (`0.14`).
- "Last hit" → `#slap-last` — either `—` or, e.g., `<span class="tag-hook">LEFT HOOK</span> · 480 ms ago`.
- "Latency" → `<b id="cv-latency">—</b>ms cv · <b id="hit-latency">—</b>ms hit · <b id="fps">—</b> fps`.

**Reasons** (badge text, produced by `SlapDetector`):
- `waiting for hand`, `no hand visible`, `palm far (w=0.03 < 0.05)`, `slow approach (1.4/s < 2.6/s)`, `unsustained (1.1/s below 1.43/s)`, `open hand (fist=0.20 < 0.35) — wave, not punch`, `slow wrist (0.35/s < 0.65/s)`, `cooldown 120ms`, `HOOK · LEFT`, `UPPERCUT · UP`, `JAB · JAB`.

### 4.3 Capture guide dialog — BUILT
`#capture-dialog` in `src/main.js:53`. Three cards:
1. **Try your face now** — `#snapshot` primary "Use webcam portrait", `#photo-import` small "↑ Choose a face photo".
2. **Scan for fidelity** — `#record` "Record a face scan".
3. **Bring your own head** — `#glb-import` primary "↑ Upload a GLB head".
Ordered list of steps, tail note, `#capture-result` status text.

### 4.4 Face scan dialog — BUILT
`#face-scan-dialog` in `src/face-capture.js:10`. Header title "Scan your whole head.".
Elements: `#face-scan-video` live preview (mirrored), `#face-scan-preview` last-accepted crop canvas, `#face-coverage` (e.g. `12 saved views · front ✓ · side A ○ · side B ○ · 3 profile/rear views saved (angles pending reconstruction)`), `#face-scan-feedback` note. Controls: `#face-scan-record` primary "Record 360°", `#face-scan-stop`, `#face-scan-video-import` "Upload video", `#face-scan-import` "Import head photos", `#face-cloud-review` checkbox "AI head completion", `#face-timing` pipeline stats block, `#face-scan-build` primary "Create 3D face", `#face-job-state` status, `#face-scan-load`, `#face-scan-delete`, `#face-scan-saved` select. Camera calibration and OpenAI API settings collapsible.
Job states: `No reconstruction started.`, `Verifying camera poses, then training a Gaussian splat…`, `Each accepted frame is saved immediately on this computer.`, `Face loaded into the interaction scene. Close this panel to inspect the surface and try a hook.`.

### 4.5 Arm scan dialog — BUILT
`#arm-dialog` in `src/arm-capture.js:10`. Title "Scan your arm, one side at a time.". `#arm-left`/`#arm-right` toggle, `#arm-length` numeric input, `#arm-live` mirrored preview, `#arm-preview` mask crop, `#arm-state` note, `#arm-start` primary, `#arm-stop`, `#arm-train` full, `#arm-import` small, `#arm-result` status. Sample states: `Move the whole arm away from your face so the fist and skin stay unobstructed.`, `Close the selected hand into a fist and keep that pose throughout this scan.`, `Verifying camera poses, then training a Gaussian splat. This can take several minutes.`, `The arm is too small or blurred. Move closer and pause between angles.`.

### 4.6 Impact reference library dialog — BUILT [PARTIAL — relies on `/api/references` endpoint]
`#reference-dialog` in `src/impact-references.js:6`. Title "Study the surface response.". Controls: `#reference-search`, `#reference-select`, `#reference-info`, `#reference-canvas` (draws image + landmarks + contact circle), `#reference-wire` checkbox, `#reference-extract` primary, `#reference-save`, `#reference-status`. Depends on `/api/references` returning a catalog; a full inventory of the data model wasn't found in the JS repo.

### 4.7 Sponsor dock (`#sponsor-dock`) — BUILT
`src/sponsors/boot.js` injects the dock at bottom-right; open/collapsed state persisted in `localStorage['punching-face-sponsors-open']`.
Common elements: `.sd-pill` (with three `.sd-dot` indicators for `coach`, `arena`, `obs` — colour codes: default grey `#b9c2b3`, `.on` green, `.live` red pulsing, `.warn` amber). `.sd-tabs` "Cornerman" / "Arena". Two `<section data-panel>`s. Offline fallback panel prints `Sponsor services are not running, so the coach and the arena are off. The rest of PUNCHING FACE is unaffected.` with `Retry` button.

### 4.8 Sponsor dock — Cornerman tab (`data-panel=coach`) — BUILT
`src/sponsors/cornerman.js`.
Elements:
- Row 1: `data-k="toggle"` primary "Start coach"/"Stop coach"; `data-k="model"` badge showing e.g. `qwen3.5-omni-flash` or `MOCK · no key`.
- `data-k="meter"` — thin bar showing mic RMS; `.open` when the gate is speaking.
- `data-k="status"` — dynamic status. Real: `Hands-free: just talk. Your fists are busy, so there is nothing to press.`, `Learning the room noise… then just talk.`, `Listening. Ask "how is my guard?" or throw a combo.`, `Listening…`, `Heard a noise, not a question.`, `Thinking…`, `Mock reply (add an OMNI key for the real model). First response in 640 ms.`, `Coach is off. Nothing is being sent.`.
- `data-k="log"` — chat transcript, alternating `.you` and `.coach` paragraphs prefixed `You:` / `Coach:`.
- Row 2: `data-k="ask"` typed question input (`maxlength=300`) + `data-k="send"` "Ask".
- `data-k="vision"` checkbox "Let the coach see me" with `data-k="leaving"` badge (e.g. `≤4 keyframes + voice per turn → yibuapi.com`, `voice + numbers only`).
- `data-k="voice"` checkbox "Spoken replies (interrupt by talking)".
- `data-k="proactive"` checkbox "Speak up on combos and personal bests".
- `<details>` "OMNI key" with `data-k="gateway"` label, `data-k="key"` password input, `data-k="save"` button. Copy: `Stored only on this computer (.local/secrets/omni.json, mode 0600). Get one from the Huawei form; the gateway is <span>.`

### 4.9 Sponsor dock — Arena tab (`data-panel=arena`) — BUILT
`src/sponsors/arena-host.js`.
Elements:
- Row: `data-k="name"` (Host) + `data-k="room"` (random `ring-abcd`).
- Row: `data-k="toggle"` primary "Go live"/"End session"; `data-k="mode"` badge (`LIVE`, `cloud`, `dev`, `not configured`).
- `data-k="cam"` checkbox — "Also share my webcam with the room (the 3D head is always shared)".
- `data-k="status"` message line (e.g. `Live in "ring-abcd". Share the invite; hits from guests land on the head.`, `Invite copied.`, `The 3D stage is not ready yet.`, `main.js lost its remotePunch hook…`).
- `data-k="invite"` block (`hidden` until live): `<canvas data-k="qr" 132×132>`, `data-k="kind"` badge (`ONE LINK FOR EVERYONE` / `ONE GUEST PER LINK`), `data-k="link"` code, `data-k="copy"`, `data-k="another"`, `data-k="reach"` text.
- `data-k="tiles"` grid of remote participants (aspect 4:3, name overlay, `.hit` outline flash).
- `data-k="board"` table — three columns per row: name, `<count> hits`, `<max> m/s`.
- `<details>` "LiveKit keys" with `url`, `apiKey`, `apiSecret`, `guestUrl`, `tokenServerId` inputs and Save button.

### 4.10 Guest page (`/guest.html`) — BUILT
`src/sponsors/guest.js` + `src/sponsors/guest.css`.

**Join screen**
- Heading "Join the ring".
- Paragraph "You'll see the host's live 3D head. Throw punches at your camera and they land on it.".
- `#name` text input (`maxlength=24`).
- Checkbox `#share` (default checked) with copy "Share my camera with the room. Your hands are tracked on this device either way; without this, only punch events leave it.".
- `#join` primary button "Join".
- `#note` — either `Room: <r>` or an error message in red.

**Ring screen**
- `.stage` filling the flexbox: `video.model` = host's 3D head; `#wait` overlay "Waiting for the host's head…"; `.self` self-cam preview (top-right, 30% wide, 4:3, mirrored) with `#track` status (`camera off`, `loading hand tracking…`, `tracking · camera shared`, `tracking · camera private`, `no camera · use the pads`); `#flash` full-stage green inset on hit; `#toast` bottom.
- `.hud`: `Hits <b>0</b>`, `Best <b>0.0</b> m/s`, spacer, `RTT <b>—</b>`, `#leave` button.
- `.pads` three big buttons: `LEFT HOOK` (`data-pad="left"`), `JAB` (`data-pad="jab"`), `RIGHT HOOK` (`data-pad="right"`).
- `.board` div — inline scoreboard `<name> <b>count</b>` pills.

**States**
- Disconnected → back to Join screen with `Disconnected. The host may have ended the session.`.
- No camera → `no camera · use the pads`, toast `No camera (<name>). The pads still work.`.
- Miss (physics not ready) → toast `No contact: <reason>`.
- Hit → toast `<zone> · <speed> m/s`, flash, `navigator.vibrate?.(30)`.

### 4.11 Latency overlay (`#omni-latency-overlay`) — BUILT
`src/omni/latency.js`. Hidden by default. Ctrl-L (in the app or in Clinic) toggles it. Fixed bottom-right (`right:12px; bottom:12px`), 260 px min-width, colour `#d8e7cf` on `rgba(20,32,26,0.86)`.
Displays: title "OMNI latency (Ctrl-L to hide · N turns)", one row per timed hop with a green bar and `<ms>`:
- `contact→event`
- `event→relay`
- `contact→cached`
- `last frame`
- `speech end`
- `response asked`
- `first delta`
- `response done`
Footer: `plan=<A|B|C|mock> · mock=<bool> · id=<sessionId>` (from `window.__omniSession`).

### 4.12 Hair review (`/hair-review.html`) — BUILT (standalone tool)
Cards for the pipeline (`1 · Extract` … `5 · Compare`), a nav of angle buttons (rendered by `src/hair-review.js`), three checkboxes (`Hair fibers`, `Photo overlay`, `Inspect shape`), status text `Loading local reconstruction…`, split view with photo and reconstruction canvas.

### 4.13 Head-template preview (`/head-template-preview.html`) — BUILT (standalone tool)
No panels. Full-window renderer at 768×768 with a raw hidden `<pre id="landmark-result">` of FaceLandmarker output; not styled for humans.

### 4.14 Clinic — Consent gate, Brief, Encounter, Debrief — PLANNED
No files. OMNI.md §7 specifies a 30-second consent gate ("subject faces the camera and says a consent phrase; OMNI verifies face and phrase before scanning proceeds"). Everything else about Clinic screens must be designed from scratch.

---

## 5. In-world / AR elements

### 5.1 Studio scaffolding (BUILT)
Four flush `panel()` meshes forming floor + back + side walls (`src/main.js:69-73`). Four `GridHelper`s etched onto them for scale (`src/main.js:74-78`). Lights: hemisphere (2.7), warm key (2.4), rim (1.3), fill (0.9). Fog `#f6f7f2` between 3–9 m.

### 5.2 Virtual hand skeletons (BUILT)
`src/hands.js` — `VirtualHand extends THREE.Group`. Neon-violet `LineBasicMaterial` `0xb14dff`, `renderOrder=20`, non-depth. 22 line segments per hand (21 MediaPipe bones + wrist→elbow). Drawn always when a hand is tracked and no scanned arm has taken over (`h.line.visible=!arm`).

### 5.3 Scanned personal arm meshes (BUILT)
`src/scanned-arm.js` (imported but not read here). Populated from `/api/arm-status` bundles. Replace the neon skeleton when active. When active, `#view-label` changes to `FIRST-PERSON · CAPTURED ARM MESHES`.

### 5.4 Head hair strands + accessory glasses (BUILT)
`src/head-hair.js`, `src/head-accessories.js`. Attached to `headPivot`. Editable via the Hair style sub-panel; imported/exported via GLB `userData.accessories`.

### 5.5 Rig markers (BUILT)
Small green (`0xcaf4a4`) spheres at 7 anchor points (brow×2, squint×2, smile×2, jaw) on the head. Toggle via `#rig` checkbox. Clicking one focuses its slider and toasts `<name> control selected — adjust its slider.`.

### 5.6 Reticle (BUILT)
CSS-only crosshair at 50%/46% of the stage.

### 5.7 Impact label (BUILT)
`#impact-label`, flashes near the top of the stage for ~1 s (`@keyframes flash`) with `TRACKED CONTACT`, `DEMO CONTACT`, `REMOTE · <NAME>`, or the punch-type label `LEFT HOOK`, `RIGHT HOOK`, `UPPERCUT`, `STRAIGHT`.

### 5.8 Deformation trace (BUILT)
`#signal` canvas — a rolling `dynamics.maxDisplacement` line, updated every 3 physics frames.

### 5.9 Slap HUD detector overlay (BUILT)
See 4.2. Strike-zone tints (left/right dashed columns, uppercut arc), hand skeleton and fingertip polygon in green (`#c9f378`) turning yellow (`#ffd66a`) when the palm is inside the trigger range, a full-panel radial flash per punch type on trigger.

### 5.10 Cached reaction burst (BUILT — synth fallback; real audio pending)
`src/omni/reactions.js`. Plays a short (100–400 ms) synthesized grunt/exhale ~50 ms after a strike, keyed `arena.grunt.<low|mid|high>` / `arena.tap`. When `/omni-reactions/<key>.wav` exists it replaces the synth (`[PARTIAL]` — the wav files aren't shipped yet; only a README is present at `public/omni-reactions/README.md`).

### 5.11 LiveKit remote tiles (BUILT)
`.sd-tile` — one per remote participant, aspect 4:3, mirrored video, name label. Flashes with a bright outline (`.hit`) on landed remote punch.

### 5.12 Clinic body highlights, exam markers, palpation feedback (PLANNED)
Not present. OMNI.md implies contact events (`press`, `release`) already surface at the classifier layer, but no visual for them exists yet.

---

## 6. Live data and events the UI can display

### 6.1 Engine event bus (`src/omni/events.js`)
Every event runs through `EventBus.emit`, which adds `t: performance.now()` and rebroadcasts as both a generic `event` and a typed event (`strike`, `press`, `release`, `guard`, `consent`, `user.text`, `user.speaking`, `user.silent`, `frame`, `session.start`, `session.end`, `tool.applied`, `tool.failed`).

| Type | Fields | Real example (from tests + code) |
|---|---|---|
| `strike` | `region:string`, `force:number`, `speed:number`, `mode:'hook'\|'jab'\|'uppercut'`, `t` | `{type:'strike', region:'cheek-left', force:55, speed:1.72, mode:'hook', t:12345.6}` |
| `press` | `region`, `pressure:number 0..1`, `t` | `{type:'press', region:'cheek-right', pressure:0.42, t:12500}` |
| `release` | `region`, `speed:number`, `rebound:boolean`, `t` | `{type:'release', region:'cheek-right', speed:0.71, rebound:true, t:12800}` |
| `guard` | `state:boolean`, optional `region` | `{type:'guard', state:true, region:'head'}` (formatted `[EVENT] guard on region=head`) |
| `consent` | `stage:string`, `granted:boolean` | `{type:'consent', stage:'scan', granted:true}` |
| `user.text` | `text:string` (truncated at 240 chars in the formatter) | `{type:'user.text', text:'How is my guard?'}` |
| `user.speaking` / `user.silent` | none | broadcast by AudioCapture's voice gate |
| `frame` | none (a hint that a fresh keyframe was posted) | — |
| `session.start` | optional `scenario` | `{type:'session.start', scenario:'arena'}` |
| `session.end` | none | — |
| `tool.applied` | `tool:string`, `args:object`, `result:any` | `{type:'tool.applied', tool:'react_to_hit', args:{location:'cheek-left', severity:2}, result:{applied:true}}` |
| `tool.failed` | `tool`, `args`, `error:string` | `{type:'tool.failed', tool:'nope', args:{}, error:'unknown tool: nope'}` |

Formatter output (`formatEvent`) — one line per event, e.g. `[EVENT] press cheek-right pressure=0.42`, `[EVENT] release cheek-right speed=0.71 rebound=true`, `[EVENT] strike cheek-left force=55`, `[EVENT] consent scan granted`, `[USER] How is my guard?`.

### 6.2 OmniSession events (`src/omni/session.js` and `src/omni/fallback.js`)
Dispatched as `CustomEvent`s on the OmniSession/FallbackSession. Wire these to UI:
- `health` — from `/health` prime response: `{plan:'A'|'B'|'C'|'mock', voice:'Ethan'|…, enabled:boolean}`.
- `offline` — `{reason:string}` when the relay didn't respond.
- `ready` — `{plan, mock:boolean, sessionId:string}`.
- `error` — `{reason:string}` (from relay or upstream).
- `session.updated` — server ack of `session.update`.
- `text.delta` — `{delta:string}` (assistant text chunk).
- `audio.delta` — `{base64:string, rate:24000}` (raw PCM16).
- `tool.call` — `{name:string, arguments:object, call_id:string|null}`.
- `response.done` — echoes the model's `response` object, including usage token counts.
- `user.speaking` / `user.stopped` — from `input_audio_buffer.speech_started/stopped` upstream events.
- `closed` — `{clean:boolean}`.
- `reconnecting` — `{waitMs, attempt}`.
- `binary` — `{buffer:ArrayBuffer}` (unused hook for future binary responses).
- `raw` — any unrecognised upstream message (for the latency overlay's debug view).

### 6.3 OMNI Arena tools (`src/scenarios/arena/tools.js`)
Registered when `?arena_omni=1`. Each schema is registered via `ToolRegistry.register`.

| Tool | Parameters | Effect |
|---|---|---|
| `react_to_hit` | `location:string ("head, jaw, cheek-left, cheek-right, nose, body-left, body-right")`, `severity:number (1–3)` — required | `avatar.reactToHit`; currently logs `event`, e.g. `react_to_hit cheek-left sev=2` |
| `set_expression` | `emotion:enum(smug\|stunned\|focused\|winded\|defiant)` required, `intensity:number` default 0.6 | Logs `expression <emotion> <intensity>` |
| `taunt_gesture` | `type:enum(chin-up\|come-on\|shrug)` required | Logs `taunt <type>` |
| `coach_callout` | `flaw:string` required, `urgency:number` default 0.5 | Logs `callout <flaw> u=<urgency>` |
| `tap_out` | `reason:string` optional | Logs `tap_out <reason>`; returns `{ended:true}` |

All handlers return an object (`{applied:true}` or `{ended:true}`) so the tool loop closes on `conversation.item.create` `function_call_output`.

### 6.4 Cornerman coach tool payload (`sponsor_server.py`, coach path)
The Cornerman relay accepts one POST per turn at `/sponsors/coach/turn`. Not a tool schema per se; the model reply arrives as SSE `text` chunks + `audio` chunks + `done`. Persona: `COACH` prompt embedded in `sponsor_server.py:32-39`. UI status text keys off `mock` (`meta.mock`), `first_response_ms`, `error`.

### 6.5 Latency metrics (`src/omni/latency.js`)
Ordered hops, all in **ms** relative to the last `contact.event`:
`contact.event → event.sent → reaction.audio → frame.sent → audio.commit → response.requested → response.first_delta → response.done`.
Also exposed as `window.__omniLatency` for QA. `history` retains up to 50 turns.

### 6.6 Session/connection statuses (surfaced in UI)
- `#stage-status`: `● DEMO INPUT`, `● WEBCAM · VIRTUAL POV`.
- Sponsor dock badges: `on`, `warn`, `live` classes on `.sd-dot`.
- Cornerman `model` badge: `qwen3.5-omni-flash` (real) vs `MOCK · no key`.
- Arena `mode` badge: `LIVE`, `dev`, `cloud`, `not configured`.
- OMNI reconnecting `waitMs=<exp backoff>` (PLANNED to surface as a small badge — `[PLANNED]`; no visual yet).
- Guest RTT: `#rtt` in ms.

---

## 7. Content models

### 7.1 Opponent persona (Arena, BUILT)
Produced by `arenaPersona({opponentName='The Sparring Partner'})` in `src/scenarios/arena/tools.js:78`. Injected as `session.instructions`. Real value:
```
You are The Sparring Partner, a boxing sparring partner played by an AI in an AR ring.
You see the user through their webcam; you hear their voice; you can be interrupted.
Stay in character. Trash-talk sparingly. Coach concretely.

RULES:
1. Physical events (`[EVENT] strike …`) come from the local hit classifier. Use the region and force provided.
2. When hit, call `react_to_hit` with the reported location and a severity 1–3.
3. Watch for dropped guard and telegraphed punches; call `coach_callout` with a short flaw.
4. If the user says "hold" or "stop", stop sparring immediately.
5. Speak in short bursts. Keep momentum with the physical loop.
```

There is no structured opponent-profile schema (no name, colour, avatar model, entrance audio). Any richer profile is `[PLANNED]`.

### 7.2 Cornerman coach persona (BUILT)
`sponsor_server.py:32-39` (`COACH` constant). Real value:
```
You are Cornerman, a boxing coach watching one or more people spar against a 3D head on a laptop.
You receive webcam keyframes of the person throwing, a spoken question (if any), and exact punch telemetry
measured on the device. Trust the telemetry for numbers and use the frames for form: guard height, elbow flare,
stance, whether they reset after punching. Speak like a coach between rounds: one or two short sentences, concrete,
one correction at a time, use names when several people are in the room. Never invent numbers that are not in the
telemetry. If a frame shows nothing useful, say what you need to see. Safety comes first: if someone sounds winded,
dizzy or in pain, tell them to stop and rest. This is solo training against a virtual target; never encourage
hitting a person.
```

### 7.3 Round telemetry snapshot (BUILT)
Emitted by `RoundStats.snapshot(now, trigger)` in `src/sponsors/telemetry.js`. Sent to the coach and (subset) to remote guests.
Shape (with a real synthesized example):
```json
{
  "participants": [
    {
      "name": "Akash",
      "count": 12,
      "avg": 1.72,
      "max": 2.94,
      "left": 5,
      "right": 7,
      "zones": {"cheek-L": 4, "cheek-R": 3, "jaw-R": 2, "chin": 3}
    }
  ],
  "last": {"name": "Akash", "zone": "cheek-L", "speed": 2.31},
  "trigger": "combo"
}
```
`trigger` may be `null`, `'combo'`, `'personal-best'`, or `'8 punches since the last cue'`.
`zoneOf(point)` outputs one of: `forehead`, `brow`, `eye-L`, `eye-R`, `nose`, `cheek-L`, `cheek-R`, `mouth`, `jaw-L`, `jaw-R`, `chin`.

### 7.4 Scoreboard row (BUILT)
`RoundStats.scoreboard()` → `[{id, name, count, max}]`, e.g. `[{id:'host', name:'Host', count:12, max:2.9}, {id:'guest-8a3f', name:'Sam', count:4, max:1.6}]`. Rendered as a 3-column table in the Arena dock and as inline pills on the guest page.

### 7.5 Sanitised remote punch (BUILT)
`sanitizePunch(msg)` in `src/sponsors/sse.js`. Fields (all clamped):
```json
{"u": -0.6, "v": -0.05, "lateral": 0.9, "speed": 1.8, "side": "left", "kind": "hook"}
```
Real values from the guest pads (`src/sponsors/guest.js:88`):
- Left hook: `{u:-.6, v:-.05, lateral:.9, speed:1.8, side:'left', kind:'hook'}`
- Jab: `{u:0, v:.05, lateral:0, speed:1.5, side:'left', kind:'straight'}`
- Right hook: `{u:.6, v:-.05, lateral:-.9, speed:1.8, side:'right', kind:'hook'}`

### 7.6 LiveKit data-channel messages (BUILT)
Sent both ways via `publishData` with a JSON body:
- `{type:'ping', t:<ms>}` / `{type:'pong', t:<ms>}` (RTT).
- `{type:'punch', ...sanitizePunch shape...}` (guest → host).
- `{type:'hit', id:'guest-8a3f', name:'Sam', zone:'cheek-L', speed:1.9, board:[…scoreboard…]}` (host → all).
- `{type:'miss', id:'guest-8a3f', reason:'the head is still loading'}` (host → thrower).
- `{type:'board', board:[…]}` (host → new joiner).
- `{type:'state', …}` (topic `state`, reserved).

### 7.7 OMNI Realtime session config (BUILT)
Payload sent by `OmniSession._sendSessionUpdate`:
```json
{
  "type": "session.update",
  "session": {
    "modalities": ["text", "audio"],
    "instructions": "You are The Sparring Partner, …",
    "tools": [ { "type":"function", "name":"react_to_hit", "function": { … } }, … ],
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "turn_detection": null,
    "voice": "Ethan"
  }
}
```

### 7.8 Saved editable session (BUILT)
`sessionData()` in `src/main.js:490`. Shape (excerpt):
```json
{
  "format": "punching-face-session",
  "version": 1,
  "physics": {"binding": {…}, "cage": {…}, "id": "<capture-id>"},
  "sourceTransform": {…},
  "name": "Your photo model",
  "geometry": {…THREE geometry JSON…},
  "original": [0.001, -0.02, 0.03, …],
  "rig": {"jaw": 0, "smile": 0.2, "brow": 0, "squint": 0.1},
  "softness": 0.6,
  "photo": "data:image/png;base64,…",
  "appearanceAtlas": {…},
  "textureFlipY": true,
  "unlit": false,
  "accessories": {
    "glasses": {…}, "hair": {…},
    "visibility": {"glasses": false, "hair": true}
  },
  "stats": {…},
  "distance": 0.55, "yaw": 0, "pitch": 0,
  "material": {…}
}
```

### 7.9 Contact classifier config (BUILT)
`ContactClassifier` needs `{regionFromPoint, onEvent}`. Region maps live in the classifier module:
- `arenaRegions(point)` → `null | 'jaw' | 'nose' | 'cheek-left' | 'cheek-right' | 'head'`.
Thresholds (constants in `src/contact/classifier.js`): `RELEASE_THRESHOLD_MS=120`, `PRESS_DWELL_MS=180`, `STRIKE_MIN_SPEED=1.5 m/s`, `REBOUND_WINDOW_MS=350`, strike debounce 80 ms.

### 7.10 OMNI usage ledger row (BUILT — server-side, not in UI)
Written to `.local/usage/yibu_api_calls.jsonl` by every API call. Not rendered in the UI; produced by `npm run omni:report` for organizer submission.

### 7.11 Clinic case file (PLANNED)
Not present. OMNI.md's `scenarios/clinic/` and its rubric, patient profile, and encounter report have no code yet. Any schema is `[PLANNED]`.

### 7.12 Slap detector tuning (BUILT — designer-editable via `window.__punchingFace.tuneSlap`)
`DEFAULT_TUNING` in `src/slap-detect.js:33`:
```js
{ minWidth: 0.05, minFractionalGrowth: 2.6, minFistScore: 0.35,
  minPeakSpeed: 0.65, minSustainedFraction: 0.55, minVerticalUp: 0.9,
  cooldownMs: 170, historyMs: 100, sideSlack: 0.08 }
```

### 7.13 Impact reference evidence (BUILT)
Saved by `#reference-save` (`src/impact-references.js:28`):
```json
{
  "format": "punching-face-impact-evidence",
  "referenceId": "…",
  "source": "…url…",
  "features": ["profile", "closed-fist glove"],
  "contact": {"x": 0.42, "y": 0.53},
  "landmarks": [{"x":0.51,"y":0.47,"z":0.02}, …468 entries],
  "limitations": [
    "Monocular detector estimate, not measured 3D deformation.",
    "Occluded landmarks are unverified.",
    "No neutral comparison frame or physical calibration."
  ],
  "imageSize": [1200, 800]
}
```

---

## 8. Shared components

There is no formal component library. The reusable UI pieces are:

| Name | Path | Where used | Notes |
|---|---|---|---|
| `toast(text)` | `src/main.js:101` | Main app | 5-second `role="status"` bar; single instance `#toast`. |
| `busy(active, message)` | `src/main.js:100` | Main app | Full-stage overlay + spinner. |
| `.sd-status` line | `src/sponsors/sponsors.css` | Sponsor dock (both tabs) | Uses `.error` modifier. |
| `.sd-badge` | `src/sponsors/sponsors.css` | Dock model/mode badges | Modifiers: `.mock`, `.leaving`. |
| `.sd-meter` | `src/sponsors/sponsors.css` | Cornerman mic meter | `.open` when speaking. |
| `.sd-log` | `src/sponsors/sponsors.css` | Cornerman chat | `.you` / `.coach` message classes. |
| `.slap-bar` / `.slap-threshold` | `src/style.css:53-57` | Slap HUD | Signed variant `.slap-bar-signed`. |
| `.metric` block | `src/style.css` | Right panel (Contact response) | Uses `.metric-grid` container. |
| `.check`, `.controls-label`, `.row`, `.full`, `.small`, `.primary`, `.icon-btn`, `.eyebrow`, `.note`, `.notice`, `.hidden` | `src/style.css` | Everywhere in the main app | Utility classes. |
| `<dialog>` pattern | `src/main.js`, `src/face-capture.js`, `src/arm-capture.js`, `src/impact-references.js` | Capture guide, Face scan, Arm scan, Impact references | Each has an `.close` × button and title/eyebrow header. |
| Tag pills | `src/style.css:62-65` | Slap HUD `#slap-last` | `.tag-hook` (green), `.tag-uppercut` (orange), `.tag-jab` (blue). |
| `VirtualHand` | `src/hands.js:25` | Both scenarios (Arena, will be Clinic) | Neon line skeleton. |
| `ImpactReferences` | `src/impact-references.js` | Right panel button | Modal picker. |
| `FaceCapture` | `src/face-capture.js` | Capture guide → face scan | Big dialog with camera + import + build + poll. |
| `ArmCapture` | `src/arm-capture.js` | Camera section → scan arms | Similar dialog for each arm. |
| `SlapDetector` | `src/slap-detect.js` | Main loop | Punch classifier — designer-tunable. |
| `ContactClassifier` | `src/contact/classifier.js` | Shared-engine Arena; PLANNED Clinic | Region-mapped strike/press/release. |
| `ToolRegistry` | `src/omni/tools.js` | Both scenarios | Tool schema + dispatcher. |
| `OmniSession` / `FallbackSession` | `src/omni/session.js`, `src/omni/fallback.js` | Both scenarios | Same public surface. |
| `FrameCapture` / `AudioCapture` | `src/omni/capture.js` | Both scenarios | Worker + AudioWorklet. |
| `latencyOverlay()` | `src/omni/latency.js` | Global | Singleton, Ctrl-L toggle. |
| Cached reactions (`play`, `keyForEvent`, `warmUp`) | `src/omni/reactions.js` | Both scenarios | `<50 ms` audio. |
| `EventBus` + `formatEvent` | `src/omni/events.js` | Both scenarios | Text formatter for the model. |
| `RoundStats` | `src/sponsors/telemetry.js` | Sponsor dock, guest scoreboard | Pure, unit-tested. |
| `VoiceGate` | `src/sponsors/audio.js` | Cornerman | Handles gate open/close. |
| `PunchDetector` | `src/sponsors/punch-detect.js` | Guest page | Different from `SlapDetector`; simpler, image-space only. |
| `EventStream` | `src/sponsors/sse.js` | Cornerman + FallbackSession | SSE frame parser. |
| `RateLimit` | `src/sponsors/sse.js` | Arena host | 6 punches/s per guest. |
| `obs` (`sentry.js`) | `src/sponsors/sentry.js` | Everywhere | No-op unless DSN configured. |

---

## 9. Copy inventory

Grouped by screen. All strings are lifted verbatim from the source files.

### 9.1 Global / brand
- Wordmark: `PUNCHING FACE`
- Sub-title on `index.html`: `PUNCHING FACE — Spatial interaction lab`
- Guest tab title: `PUNCHING FACE — join the ring`
- Fullscreen icon: `⤢` (title `Fullscreen`/`Exit fullscreen`)
- Capture guide open button: `Capture guide ↗`
- Meshy: `Photo → 3D self`, `Compress GLB`, `Beat yourself`
- Compression toast: `Compressed: 260 KB → 78 KB (70% smaller)` (template)

### 9.2 Arena — left panel (Face)
- Section h2: `Face`
- Default face info: `Reference head` / `Public mesh reference`
- Buttons: `Record / upload head video`, `↑ Upload GLB head`, `↺ Reset to reference`
- Muted lines: `Saved multiview photos`, `Loading reconstruction…`
- Toast on GLB with multiple meshes: `GLB has <n> meshes; using the largest (<name>) as the face surface.`
- Toast on import cap: `Use a GLB or editable session smaller than 180 MB.`
- Toast on load: `Editable session restored. Re-import a room asset separately.`, `Saved editable session and Photographs link locally.`

### 9.3 Arena — left panel (Camera)
- h2: `Camera`
- Buttons: `Connect laptop webcam`, `Disconnect webcam`, `Calibrate guard position`, `Scan my arms`
- Status: `Connect your camera for tracking.`, `Starting local hand tracking…`, `Webcam connected · show your face and hands, then calibrate guard`, `Personal arm loaded. Show your face, shoulders, elbows and wrists, then calibrate guard.`, `Camera off. Virtual hands are in demo mode.`, `Camera could not start. Allow camera access in the browser, then reconnect.`, `Personal proportions calibrated · body and palm tracking drive the captured meshes`, `Guard calibrated · close your fist and move across the target`
- Checkbox: `Demo hand shapes`
- Arm appearance: `Personal arm meshes: awaiting capture.`, `Captured meshes: left + right. Auto rig; inspect joints in motion.`, `Following your left and right arm. Capture proportions retained; pose and skin weights are estimates.`
- Toast on hook while camera on: `Disconnect webcam to run a demo hook.` / `Disconnect webcam to run a demo uppercut.`

### 9.4 Arena — left panel (Room)
- h2: `Room`
- Button: `↑ Add room panorama`, `Restore studio`
- Labels: `Heading`, `Scale`, `Height`
- Default label: `Studio environment · placeholder`
- Toast: `Panorama loaded. Use a 2:1 equirectangular image for a correct 360° view.`, `Choose a JPG or PNG panorama.`, `Use a panorama smaller than 250 MB for this prototype.`

### 9.5 Arena — stage overlays
- Scene name: `Reference head <span>CPU Poisson surface</span>` / `Editable mesh` / `Landmark depth estimate`
- Status: `● DEMO INPUT`, `● WEBCAM · VIRTUAL POV`
- View switch: `Surface`, `Geometry`, `Wireframe`
- Impact label: `CONTACT REGISTERED` (default) / `TRACKED CONTACT` / `DEMO CONTACT` / `REMOTE CONTACT` / `LEFT HOOK` / `RIGHT HOOK` / `UPPERCUT` / `STRAIGHT`
- Key hints: `Q Left hook`, `E Right hook`, `Space Uppercut`, `Drag Inspect`
- Bottom line: `FIRST-PERSON · VIRTUAL HANDS`, `FIRST-PERSON · CAPTURED ARM MESHES`, `Connect webcam · palm toward camera = punch`
- Busy: `Preparing surface…`, `Loading reference mesh…`, `Importing face asset…`, `Estimating a portrait mesh…`, `Loading your photo model and initializing Newton…`, `Meshy is texturing your head (up to a few minutes)…`, `Compressing GLB with gltf-transform…`, `Exporting mesh and expression controls…`, `Loading room context…`
- Busy muted: `Processed on this computer.`

### 9.6 Arena — Slap HUD
- Title: `PUNCH DETECTOR`
- Legend: `LEFT`, `UP`, `RIGHT`
- Metric labels: `Approach`, `Rise`, `Palm size`, `Last hit`, `Latency`
- Suffixes: `0.0/s`, `0.0/s up`, `0.00`
- Latency: `— ms cv · — ms hit · — fps`
- Empty state: `Connect camera to see the detector view`
- Badge reasons: `waiting for hand`, `no hand visible`, `palm far (w=… < …)`, `slow approach (…/s < …/s)`, `unsustained (…/s below …/s)`, `open hand (fist=… < …) — wave, not punch`, `slow wrist (…/s < …/s)`, `cooldown …ms`, `HOOK · LEFT`, `HOOK · RIGHT`, `UPPERCUT · UP`, `JAB · JAB`, `searching`, `idle`

### 9.7 Arena — right panel (Contact response)
- h2: `Contact response`
- Labels: `Contacts`, `Last speed`, `Peak deformation`, `Softness`, `Distance`
- Buttons: `↗ Left hook`, `Right hook ↖`, `Release deformation`, `Reset face`, `Export GLB ↗`, `Save editable session`
- Checkboxes: `Hold peak deformation`, `Head recoil`, `Slow motion`
- Muted defaults: `Facial impact rig · ready`, `Select a reconstructed photo model.`
- Physics-engine texts: `Preview springs + facial impact rig`, `Reference springs + facial impact rig`, `Meshy import - preview springs + facial impact rig`, `Imported preview · no Newton cage attached`, `Starting Newton CPU solver…`, `Restored textured mesh · welded physics`
- Export toast: `Saved .local/exports/punching-face.glb with geometry, appearance, and four morph targets.`

### 9.8 Arena — right panel (Surface & rig)
- h2: `Surface & rig`
- Checkboxes: `Wireframe`, `Rig markers`, `Sculpt` (with toasts `Drag on the face: up pulls, down pushes.` / `Inspect mode enabled.`), `3D glasses`, `Strand hair`
- Buttons: `↶ Undo`, `Redo ↷`, `Reset view`, `Impact reference library`
- Labels: `Jaw`, `Smile`, `Type`, `Top length, mm`, `Curl`, `Root lift, mm`, `Frizz`, `Brow raise`, `Lid compression`, `Face heading`, `Face tilt`
- Hair select options: `Straight`, `Wavy`, `Curly`, `Coily`, `Braided`, `Locs`
- Hair description: `Rebuild a head scan with AI hair analysis to add editable strands.` / `Detected: <style>. <n> photo-guided fibers following captured locks. Individual fibers and unseen crown detail remain estimated.`
- Rig marker toast: `<name> control selected — adjust its slider.`

### 9.9 Capture guide dialog
- Eyebrow: `Bring yourself into the scene`
- Title: `Capture a face. Then a room.`
- Card titles: `Try your face now`, `Scan for fidelity`, `Bring your own head`
- Body copies: `A frontal image becomes a textured landmark mesh. Fast likeness preview; estimated depth and no back of head. Use Record for multiview geometry.`, `Keep a neutral expression and even light. Have a helper move a phone around your still head, including both profiles, ears, chin, and crown. Keep overlapping views.`, `Skip capture. Upload a GLB head or bust from Sketchfab, Meshy, Ready Player Me, Blender, etc. The mesh is auto-scaled and wired into the impact rig.`
- Buttons: `Use webcam portrait`, `↑ Choose a face photo`, `Record a face scan`, `↑ Upload a GLB head`
- Ordered steps: `Capture 60–150 overlapping sharp face images with fixed exposure. A multiview capture is needed to preserve unseen features.`, `Press Create 3D face. Inspect Geometry and Wireframe to check depth. Gray areas mark estimated, unseen head surfaces.`, `Import a 360° room panorama for the surrounding view. A panorama supplies rotation only; it does not measure room depth.`
- Footer note: `Your laptop camera drives virtual hand articulation. A room scan supplies novel background views; it cannot reveal your hands' hidden surfaces. Capture each arm separately to reconstruct its appearance. Monocular depth and automatic rigging remain estimates.`

### 9.10 Face scan dialog
- Eyebrow: `Photographs → 3D mesh → Newton physics`
- Title: `Scan your whole head.`
- Rich intro paragraph (see `src/face-capture.js:10`).
- Coverage line template: `<n> saved views · front <✓|○> · side A <✓|○> · side B <✓|○> · <n> profile/rear views saved (angles pending reconstruction)`
- Feedback samples: `The camera stays off until you press Record. Imported video and extracted frames stay in this local project.`, `Saved view 12 · facial landmarks tracked. Continue slowly around the whole head.`, `Capture stopped. 24 views saved locally. Last skipped view: Motion blur: turn more slowly and pause at each angle.`
- Buttons: `Record 360°`, `Stop & save`, `Upload video`, `Import head photos`, `Create 3D face`, `Load face`, `Delete scan`, `AI head completion`
- Timing section: `Video to model`
- Extra copy: `Create 3D face sends selected cropped views to Astra…` (long paragraph), `Import at most 240 overlapping photos.`, `Use a video smaller than 500 MB.`, `Use a head rotation video between 3 seconds and 5 minutes.`
- Job states: `No reconstruction started.`, `Extracting video locally · 40% · 8 useful views saved`, `Video processed. 42 head views saved locally. Create 3D face will verify cameras and fit the head template.`, `Face loaded into the interaction scene. Close this panel to inspect the surface and try a hook.`, `Scan and its local reconstruction outputs deleted.`
- Camera calibration: `Horizontal field of view, degrees`, placeholder `Blank = estimate from images`
- OpenAI API section: `Server key configured · gpt-6-astra`, `No API key configured. Local reconstruction is available.`, `Checking server configuration…`, `The key is never included in exports or browser storage. Replace any key shared in chat.`

### 9.11 Arm scan dialog
- Eyebrow: `Personal geometry · multiview capture`
- Title: `Scan your arm, one side at a time.`
- Intro: `Keep your shoulder, elbow, wrist, and closed fist visible…`
- Buttons: `Left arm`, `Right arm`, `Start capture`, `Finish capture`, `Reconstruct this arm`, `Import a reconstructed arm bundle`
- Label: `Elbow to wrist, cm (blank = estimated 27 cm)`
- Sub-note: `Live framing — keep the entire selected arm inside this view.`, `Last accepted segmentation crop`, `Aim for 60–100 sharp views…`
- Prompts: `Connect the webcam, then prepare your arm. Captures are saved only on this computer.`, `Your <side> shoulder, elbow, wrist is not clearly visible. Step back and show the whole selected arm.`, `Move the whole arm away from your face so the fist and skin stay unobstructed.`, `Close the selected hand into a fist and keep that pose throughout this scan.`, `The arm is too small or blurred. Move closer and pause between angles.`, `Verifying camera poses, then training a Gaussian splat. This can take several minutes.`, `Status connection interrupted. Retrying…`

### 9.12 Impact reference dialog
- Eyebrow: `Evidence → landmarks → deformation controls`
- Title: `Study the surface response.`
- Buttons: `Search the web for more references`, `Extract visible face landmarks`, `Save landmark evidence`
- Checkbox: `Estimated landmark wireframe`
- Status samples: `Estimating facial landmarks locally…`, `468 estimated landmarks. Review the glove occlusion and projected wireframe before using them.`, `No reliable face detection. Occlusion or profile is too strong for this detector.`, `Evidence saved locally with provenance and uncertainty.`
- Footer: `Yellow topology is a detector estimate. Occluded landmarks behind a glove are predictions, not observations. Compare neutral and impact frames of the same person before fitting displacement targets.`

### 9.13 Sponsor dock — offline
- Message: `Sponsor services are not running, so the coach and the arena are off. The rest of PUNCHING FACE is unaffected.`
- Instruction: `Start them with npm run sponsors, then:`
- Button: `Retry`

### 9.14 Sponsor dock — Cornerman
- Buttons: `Start coach`, `Stop coach`, `Ask`, `Save`
- Placeholder: `…or type a question`, `API key`
- Checkboxes: `Let the coach see me`, `Spoken replies (interrupt by talking)`, `Speak up on combos and personal bests`
- Badges: `MOCK · no key`, `qwen3.5-omni-flash`, `≤4 keyframes + voice per turn → yibuapi.com`, `voice + numbers only`
- Statuses: `Hands-free: just talk. Your fists are busy, so there is nothing to press.`, `Learning the room noise… then just talk.`, `Listening. Ask "how is my guard?" or throw a combo.`, `Listening…`, `Thinking…`, `Heard a noise, not a question.`, `Mock reply (add an OMNI key for the real model). First response in 640 ms.`, `Coach: (no reply)`, `Coach: … (interrupted)`, `Key saved on this computer.`, `Coach is off. Nothing is being sent.`, `No microphone (<NotAllowedError>). Typed questions still work.`
- OMNI key note: `Stored only on this computer (.local/secrets/omni.json, mode 0600). Get one from the Huawei form; the gateway is <gateway>.`

### 9.15 Sponsor dock — Arena
- Placeholders: `Your name`, `room`, `wss://your-project.livekit.cloud`, `API key`, `API secret`, `https://…/guest.html (deployed guest page)`, `Development token server id (one link for everyone)`
- Buttons: `Go live`, `End session`, `Copy link`, `Next guest`, `Save`
- Checkbox: `Also share my webcam with the room (the 3D head is always shared)`
- Badges: `LIVE`, `not configured`, `dev`, `cloud`, `ONE LINK FOR EVERYONE`, `ONE GUEST PER LINK`
- Statuses: `The 3D stage is not ready yet.`, `Connecting…`, `Live in "<room>". Share the invite; hits from guests land on the head.`, `Connect the tracking webcam first, then share it.`, `New single-guest link ready.`, `LiveKit settings saved on this computer.`, `Invite copied.`, `Copy failed; select the link instead.`, `main.js lost its remotePunch hook (see tests/sponsors-hook.test.mjs). Falling back to the scripted left/right hook.`, `<participant> joined. <n> in the ring.`, `<participant> left.`, `Session ended. Nothing is being streamed.`, `Disconnected from the room.`
- LiveKit note: `Stored only on this computer (.local/secrets/livekit.json, mode 0600). With nothing here, a local livekit-server --dev is used and guests must be on this computer. Other devices need LiveKit Cloud plus the guest page on an https host.`
- Reach: `Works for: <invite reach>.`

### 9.16 Guest page
- Join heading: `Join the ring`
- Join body: `You'll see the host's live 3D head. Throw punches at your camera and they land on it.`
- Placeholder: `Your name`
- Checkbox copy: `Share my camera with the room. Your hands are tracked on this device either way; without this, only punch events leave it.`
- Button: `Join`
- Note (empty): `Room: <r>` or `Room: —`
- Note (error): `This link has no room in it. Ask the host for a fresh invite.`, `This invite is incomplete. Ask the host for a fresh link.`
- Wait card: `Waiting for the host's head…`
- Track states: `camera off`, `loading hand tracking…`, `tracking · camera shared`, `tracking · camera private`, `no camera · use the pads`
- Toasts: `No camera (<name>). The pads still work.`, `Not sent: reconnecting…`, `<zone> · <speed> m/s`, `No contact: <reason>`
- Disconnect: `Disconnected. The host may have ended the session.`
- HUD labels: `Hits`, `Best`, `RTT`, `Leave`
- Pads: `LEFT HOOK`, `JAB`, `RIGHT HOOK`

### 9.17 Latency overlay
- Header: `OMNI latency  (Ctrl-L to hide · N turns)`
- Empty: `no turns yet`
- Rows: `contact→event`, `event→relay`, `contact→cached`, `last frame`, `speech end`, `response asked`, `first delta`, `response done`
- Footer template: `plan=<X> · mock=<bool> · id=<sessionId>`

### 9.18 Hair review page (standalone)
- Back link: `← Back to the model`
- Title: `From your video to your hair.`
- Intro: `Compare the source photograph with the reconstruction from the same recovered camera. Check the quiff, taper, hairline, eyebrows and facial hair separately.`
- Steps: `1 · Extract`, `2 · Align`, `3 · Fit`, `4 · Recover detail`, `5 · Compare` (with tag lines)
- Checkboxes: `Hair fibers`, `Photo overlay`, `Inspect shape`
- Status: `Loading local reconstruction…`, `Look for agreement in the outline and hairline. Use the overlay to expose alignment differences.`
- Note: `Hair direction and outer shape use the captured views. Individual fibers, hidden roots and unseen crown detail remain estimates…`

---

## 10. Assets

### 10.1 Fonts
- Google Fonts CSS `@import` in `src/style.css:1` — DM Sans, Space Grotesk.
- Guest page and standalone tools use system-ui fallbacks first, then the same Google Fonts (loaded transitively when they exist).

### 10.2 Icons
- Brand logo: inline SVG at `src/main.js:21` (16-point path, stroke `#46663a`).
- Fullscreen: text glyph `⤢`.
- Arrows and marks: text glyphs (`↑`, `↺`, `↗`, `↖`, `↶`, `↷`, `×`, `●`, `✓`, `○`).
- Sponsor dock uses dot elements only (`.sd-dot`); no icon font.

### 10.3 Images / textures
- Reference head diffuse: `public/reference/Map-COL.jpg` (Lee Perry-Smith model, CC-BY 3.0, `LICENSE.txt`).
- Reference full GLB: `public/reference/LeePerrySmith.glb`.
- Reference mesh JSON: `public/reference/face-poisson.json` — the fallback surface loaded by `#reference` button and on cold-start when no capture is available.
- Reference PLY: `public/reference/face-surface-fixture.ply` (used by tests/fixtures).
- Head template: `public/head-template/base.obj`, `head.json`, `anchors.json`, licence in `LICENSE.md`, source in `SOURCE.md` (MakeHuman hm08 basemesh, CC0).
- MediaPipe canonical face: `public/models/canonical_face_model.obj`.

### 10.4 Sounds
- `public/omni-reactions/` — only `README.md` present. Cached-reaction WAV files (`arena.grunt.low.wav`, `arena.grunt.mid.wav`, `arena.grunt.high.wav`, `arena.tap.wav`) are `[PLANNED]`; until they exist, `src/omni/reactions.js` synthesizes the tones inline.
- Cornerman audio is streamed from the model; when the mock coach replies, `speechSynthesis.speak()` reads it (browser TTS).

### 10.5 3D models referenced by the UI
- `public/head-template/head.json` (default template preview).
- `public/reference/LeePerrySmith.glb` (offered by the "Bring your own head" flow implicitly when the user has none).
- `AKASH.glb` (repo root, 55 MB) — a demo head kept out of the runtime path; not fetched by the UI code seen.
- Runtime-generated meshes: photo model at `/api/face-asset?id=<id>&asset=<file>`, Meshy import via `/api/meshy-headshot`, arm bundles via `/api/arm-status`.

### 10.6 CV models
- `public/models/face_landmarker.task`
- `public/models/hand_landmarker.task`
- `public/models/pose_landmarker.task`
- `public/models/selfie_multiclass_256x256.tflite`
- `public/wasm/` — MediaPipe tasks-vision WASM runtime.
- `public/vendor/vision_bundle.cjs` — MediaPipe classic-worker bundle used by `face-worker.js` and `tracking-worker.js`.

### 10.7 Workers
- `public/face-worker.js` — reconstruction-side face detection + segmentation.
- `public/tracking-worker.js` — live hand + pose landmarker.
- Inline workers built at runtime (as `Blob`) in `src/omni/capture.js` (frame encoder + AudioWorklet).

---

## 11. Gaps and conflicts

### 11.1 OMNI.md requirements with no UI yet
- **`/clinic` route** — not present.
- **Scan consent gate** (OMNI.md §7) — not present (the classifier emits `consent {stage, granted}` events, and `formatEvent` recognises them, but no consent screen exists).
- **"Reconnecting" badge** (OMNI.md §4) — `OmniSession` emits a `reconnecting` event, but no visual surfaces it in the DOM.
- **Cached reaction audio files** — only the README exists; deployment ships with synthesized tones.
- **ElevenLabs voice fallback pipeline** — env keys exist (`.env.example`), no UI or wiring found.
- **Rich opponent profile** — code only has a one-liner name string; OMNI.md implies richer per-opponent config in `scenarios/arena/`.
- **Case files, rubric, patient profile, encounter report** (Clinic) — all `[PLANNED]`.
- **Baseline perf numbers** — `docs/perf.md` has the methodology but the table rows are `_pending_`.

### 11.2 Inconsistencies
- **Region naming.** The classifier and Arena tools use hyphenated `cheek-left`, `cheek-right`. The sponsor telemetry (`zoneOf`) uses `cheek-L`, `cheek-R`. Two vocabularies coexist for the same anatomy.
- **Voice default.** `.env.example` sets `OMNI_VOICE=Cherry`; `omni_relay.py` defaults to `Cherry`; but yibuapi rejects Cherry on the delivered key. OMNI.md and the sponsor server default fall back to `Ethan`. Anywhere the UI shows a voice name, it can disagree with what is actually used.
- **Persona strings live in two places.** `arenaPersona()` (opponent) and `COACH` (coach) both describe a boxing scenario in different tones. If a designer surfaces "who is speaking" in the UI, they'll need to reconcile.
- **Slap detector vs. sponsor punch detector.** `SlapDetector` runs on the host (`src/slap-detect.js`) with rich UI feedback. `PunchDetector` runs on the guest (`src/sponsors/punch-detect.js`) with a subset of features. Design decisions in one don't propagate to the other.
- **Meshy toggle vs. Beat yourself.** Enabling the "Photo → 3D self" checkbox and pressing "Beat yourself" both trigger the same pipeline; the two entry points share no visual grouping cue beyond `#meshy-panel`.
- **Route note.** OMNI.md says "The existing route stays at `/`". The prompt for this task said "at its current URL" (implicitly not `/`). The code confirms `/` (index.html at repo root).

### 11.3 Dead / duplicate / half-built UI
- `#demo-hands` checkbox has no handler in `main.js` — `[PARTIAL]`.
- `#capture-result` in the capture dialog is targeted only from the (older) photoFace path in error cases; often stays empty.
- The head-template preview and hair-review pages are internal review tools and not linked from the main app.
- `public/online-face` exists but isn't referenced anywhere in `src/` — historical artifact.
- `guest.html` has an inline `#guest` root plus a `<script>` that references `window.exports`; the actual guest UI is built by `src/sponsors/guest.js`. The unused stub in `hair-review.html`'s legacy inline styles and `head-template-preview.html`'s inline module code duplicate the responsibility of `src/`.

---

## 12. Open questions

1. **Where should Clinic live in the URL space?** The task prompt says `/clinic`; OMNI.md never fixes the path. Any redirect, sub-domain, or nav treatment is undecided.
2. **Global navigation between Arena and Clinic** — is there a shared shell (header links, a mode switcher) or two independent SPAs? Nothing in code today addresses this.
3. **Consent gate wording and modality** — is it a full-screen takeover, a dialog, or a Cornerman-style side panel? OMNI.md gives 30 seconds and a face+phrase requirement, no UI direction.
4. **Case-file schema** (Clinic) — is it JSON per case, YAML front-matter, an authoring tool? Not addressed.
5. **Standardized-patient avatar identity** — reuse the current reference head, generate per-case, or ship a bank? No decision or asset.
6. **Report / debrief format** — plain text, structured rubric, downloadable PDF? No stub exists.
7. **Multi-user Clinic** — is the med student always host? Does the patient talk back through Cornerman-style voice? The tool schemas in Arena presume single-player.
8. **`arena_omni=1` UX** — when the flag is on, does the user see any visible indication (a badge, a status)? Currently they don't.
9. **Voice selection UI** — the accepted voices for the current key are `Ethan / Serena / Dylan`, but the UI never lets the user pick one. Should Clinic expose voice choice?
10. **How the app degrades when the OMNI relay is down but the sponsor server is up** — code paths exist (`FallbackSession`), but no UI communicates "you're on Plan C" beyond the latency overlay footer.
11. **Colour semantics** — the `.hit` colour on tiles is the same accent green used everywhere; is there a designed "damage" or "pain" scale for Clinic (severity 1–3), or should Arena's approach be reused?
12. **Region vocabulary for Clinic palpation** — the classifier is scenario-agnostic but the only region map is `arenaRegions`. Clinic will need its own map (abdominal quadrants? thoracic zones?) — no design guidance in code.
13. **Cached reaction bank** — `arena.tap` is registered but no code path currently emits it; is it hooked to `press`?
14. **What "beat yourself" actually implies** — is this a demo joke or a Clinic-relevant capability (self-palpation)? Both scenarios might want a rename.
