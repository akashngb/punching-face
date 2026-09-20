# Target-perspective punch pipeline — architecture

Written 2026-09-19. This documents the rewrite of the target-camera CV pipeline
(`src/target-camera.js` + `src/punch-events.js` + `public/target-worker.js`), which replaced the
per-track online lifecycle that previously lived in `target-camera.js`. Read this before touching
detection behaviour; the test files pin most of what is written here.

## The goal, worked backward

The deliverable is a **per-punch report**: exact impact location on the 3D head mesh, contact
velocity, force direction, punch type (left/right jab, left/right hook, uppercut), one report per
physical punch. Camera: a single webcam sitting where the head is, looking back at the puncher.

Work backward from that report and the correct shape of the system falls out:

1. Every reported quantity is a property of a **whole trajectory**, not of any single frame.
   Impact location is where the trajectory crosses the head surface. Contact velocity is the
   derivative of the trajectory at the contact instant. Punch type is the shape of the terminal
   arc plus hand orientation. So the final stage must be a **solver over a completed observation
   window** — and everything upstream is only evidence collection.
2. "One punch = one event" is a statement about the trajectory too: a punch is
   *approach → apex → retreat*. The apex is the one unambiguous instant. Emit exactly at
   apex-confirmation, once, and retraction can never fire — retraction is the confirming half of
   the same pattern, not a separate motion to be filtered out.
3. The camera measures the image plane superbly and depth badly, and hand landmarks die exactly at
   the interesting moment (a 5 m/s fist smears ~80 mm per 60 Hz exposure; MediaPipe drops it).
   So detection may not *depend* on any single fragile stage: it needs a second, blur-proof
   evidence stream, and it must treat missing samples as normal, not as an error path.

## Why the previous architecture could not be patched

The old core was: per-frame rigid pose fit → Kalman filter → per-track online state machine
(idle/closing/spent) making irreversible fire decisions as data streamed in. Its bugs were
structural, and each fix was another layer:

- **Track identity was the event owner.** MediaPipe handedness flicker, blur teleports, and
  dropouts fragment tracks; every fragment ran its own lifecycle and fired its own event. Four
  separate dedup layers accumulated (nearest-centroid association, merge suppression of stale
  orphans, spend-all-tracks-on-fire, a global refractory window) *plus* a fifth in the app
  (`DUPLICATE_MISS_MS`). Needing five dedup layers means the event source itself is wrong.
- **Decisions at the stream edge.** The lifecycle decided "punch" online with hysteresis constants
  (2 opening samples, 70 ms quiet, 1.5 cm retreat…), so one noisy frame near a threshold produced
  duplicates or misses. Nothing ever looked at the whole punch, though a punch is only ~10 samples
  long and the answer is not needed until the apex anyway.
- **The noisiest signal was load-bearing.** Detection ran on d/dt of Kalman-fitted metric depth —
  the least observable monocular axis, differentiated, then thresholded. The best signals from a
  target-POV camera (image-plane position, apparent-scale growth/looming) were used only for
  display.
- **Report quantities came from single snapshots.** Impact aim from the one apex frame, direction
  from the one peak-speed frame (for a hook that is the mid-arc chord, not the terminal tangent),
  type classification from those two numbers. Hence unreliable location and direction.

## The new architecture

Three layers with one-way data flow. Capture ≠ evidence ≠ decision.

```
target-worker.js        HandLandmarker (2 hands, world landmarks)      per frame
 (worker)               Motion layer: 96-wide grayscale diff → blobs
                         (centroid, mass, spread, block-match flow,
                          expansion rate)                              blur-proof
        │ one result message per frame
        ▼
target-camera.js        Observation building: rigid 6-DOF fist fit
 (capture shell)         (fist-pose.js, unchanged) → camera-frame
                         metric sample + measured variance; blob
                         observations passed through
        │ observations
        ▼
punch-events.js         Slots: ≤2 soft-identity sample buffers (~1 s)
 (the core, pure)       Extraction: retrospective apex finding on the
                         fitted range signal r(t)
                        Arbitration: structural dedup across slots
                        Solver: terminal weighted-quadratic fit →
                         impact point, velocity, direction, type
        │ at most one PunchEvent per physical punch
        ▼
cv-debug.js / app       mesh raycast along reported travel direction,
                        normal/tangent decomposition, render
```

### Evidence streams

**Hand stream** (precise, fragile): MediaPipe landmarks + `worldLandmarks` → the existing rigid
fist fit (`fist-pose.js`: Horn alignment onto a learned per-hand template, Gauss-Newton
translation along the observed rays, measured covariance). Gives metric camera-frame position,
orientation (knuckle normal), closure, and an honest per-axis variance. When the fit fails but
landmarks exist, a span-prior depth fallback produces a degraded sample with wide variance rather
than no sample.

**Motion stream** (coarse, blur-proof): frame differencing at 96 px wide → connected components →
up to 2 blobs with centroid, mass, rms spread, block-matched mean translation and expansion rate
(similarity-flow fit). Motion blur *helps* this stream. It corroborates approaches, dates
reversals when landmarks are gone, and bridges dropout gaps with low-weight samples. It never
fires an event by itself.

### Slots, not tracks

At most two persistent **slots** hold time-ordered sample buffers. Association is
predicted-position nearest-neighbour with a gate that grows with the time gap (as before), but a
slot is a *bucket of evidence*, not an event owner: an association mistake means some samples land
in the other bucket, and the damage is contained by candidate arbitration below — there is no
per-frame decision that can double-fire. MediaPipe handedness never keys identity (the front of a
fist is chirally ambiguous); it is only one weighted vote in per-event hand classification.

### The range signal and retrospective extraction

Per sample, compute camera-frame metric position **p** = (x, y, z) (lateral from image position ×
depth — FOV-independent; depth fused from rigid fit / span prior / blob-expansion bridge, each
with its own variance) and **r = |p|**, the range to the head centre. Range, not depth: a hook
closes on the head laterally with almost no depth change; an uppercut rises; every punch that
lands closes range (this was the one deep insight of the old code and it is kept).

Extraction runs ~2–4 frames behind real time on the buffer, as a pure function
`buffer → candidates`:

- Smooth r(t) with a local weighted quadratic fit (tricube window ≈ ±90 ms, weights from sample
  variance). Derivatives come from the fit — no causal filter, so no lag/overshoot tuning, and a
  single noisy frame cannot cross a threshold.
- An **apex candidate** is a local minimum of r̂ that is *bracketed* — the retraction confirms
  the punch, and the confirmation must be **sustained**: at least two fitted samples ≥ 2 cm above
  the minimum, held for ~70 ms, with the trajectory still up there now. A single sample above the
  line is one bounce of near-camera depth noise, and taking it at its word fired the event
  mid-flight (a jab logged low and lateral — as a "shoulder" hit — while its real landing was
  consumed by the re-arm and never registered). A noise bounce fails all three conditions within
  a tick or two: the fist drives past the false minimum and the argmin simply moves. Or the
  candidate is *censored* — samples stop while the punch was still live: judged on the trailing
  window's PEAK closing, not the last sample's, because a hook that blurs out in its tangential
  phase was closing hard 100 ms earlier ("losing the fist at full extension IS the punch", kept
  from the old design as a first-class path, not a sweep-up hack). The motion stream either
  carries the trajectory to a measurable reversal or a ~150 ms timeout closes it.
- Gates on the candidate, all physical quantities measured on the fitted window: travel
  (range closed from the recent maximum, never reaching back across a previous fire) ≥
  `minTravel`, peak fitted closing speed ≥ `minPeak`, fist-closure evidence over the approach.
  Two kinds of truncated evidence soften the gates differently: a track born against a frame
  border flew mostly off-camera, so both peak and travel are under-measured and both gates soften
  (a hook is only visible for its last third and arrives already decelerating); a track born
  mid-frame but already fast — full 3D speed, not range rate: a hook picked up mid-arc moves
  3-5 m/s while closing range slowly (two consecutive fast pairs; one pair is indistinguishable
  from one step of depth noise) — both gates soften here too: the peak CLOSING of a
  mid-flight birth is measured only over the decelerating tail (a top-only uppercut reads
  ~0.95 m/s against the 1.2 gate), and the birth speed itself is the evidence the punch was fast. Minima that no approach ever drove into (noise wiggles at guard, the tail
  of a retraction) are consumed silently, so the last rejection reason on the readout always
  names the gate that turned a real punch away.
- **Instant fire**: r̂ crossing `contactDepth` while closing fast emits immediately (latency for
  deep straight punches) and consumes the window through the upcoming apex, so the apex cannot
  fire a second time. The line sits deep (12 cm) on purpose: a committed jab crosses a 20 cm line
  ~50–100 ms before full extension — and near-camera depth noise dips the fitted range under a
  shallow line even sooner — which reported mid-flight positions; at 12 cm the crossing IS the
  landing.

Why retraction cannot fire: pulling back is rising r. A candidate needs a falling segment ending
in a bracketed minimum with ≥ `minTravel` of range closed at ≥ `minPeak`; the settle-dip when the
fist re-enters guard closes a couple of centimetres slowly and fails both gates. No refractory
timer is involved in this property — it is the shape of the signal.

### Arbitration: dedup as a structural property

Each emitted event **consumes** its slot's samples through the confirmation time; candidate search
never reaches back across the consumption boundary. Across slots, a new candidate is merged into a
just-emitted event when their apexes are close in time (< 250 ms) *and* their terminal windows
overlap in image space — that is one fist seen under two identities (the blur-teleport case), and
the merged solve uses the union of both windows. Distinct-region candidates are allowed through at
≥ 120 ms spacing, so a real jab–cross combo is two events while a fragmented single punch is one.
Misses (trajectory crosses wide of the head) are decided by the same extractor and take part in
the same arbitration, which is what retires the old app-level "a miss right after a hit is the
same punch" patch.

### The solver: every reported number from the same fit

Given the event window (union of merged windows):

- **Contact time t\***: the fitted apex (or the plane-crossing time for instant fire; censored
  apexes are carried a short extrapolation past the last sample).
- **Terminal probes**: position and velocity are evaluated on a 10 ms grid over
  [t\*−220 ms, t\*], each probe its own per-axis local weighted quadratic. Local fits rather than
  one window-wide polynomial, deliberately: a punch's speed envelope swings from metres-per-second
  to zero inside that window, and a single quadratic is too stiff to carry a hook's rotating
  velocity through it — it reports the chord.
- **Direction**: averaged over the last ~60 ms of *qualifying* probes — a probe qualifies when
  its fitted speed clears an absolute floor (0.35 m/s, with a small relative backstop) AND it is
  actually **closing on the head**. Both constraints are load-bearing. The floor: at the apex of
  a shadow punch the fist has stopped, so velocity there is ill-conditioned; walking back to the
  last travelling moment reads the terminal tangent (the old peak-speed snapshot reported the
  mid-arc chord, ~0.5 rad wrong on a synthetic hook). The closing test and the averaging: local
  fits near the apex blend in the retraction, and a single such probe once reported one identical
  jab in eight as travelling OUT of the face — which sent the app's raycast in from behind and
  put the marker on the back of the skull, i.e. "the same punch lands on opposite parts of the
  mesh". A final invariant backstops it: a hit's direction must not travel meaningfully BACK toward
  the puncher (z > .25), else it is replaced by the terminal chord (stable for straight punches
  by construction). Only z is tested — a follow-through hook's terminal direction legitimately
  stops closing on the head centre while staying fully valid laterally, and a centre-closing form
  of this invariant wrongly collapsed such hooks to the fallback and classified them as jabs.
- **Speed**: peak fitted closing speed over the approach (the punch "at speed"). Reported velocity
  vector = direction × speed.
- **Impact point**: the FIRST point of the measured trajectory that touches the face — never
  where the fist happened to stop (`firstContact`). The fitted path is walked forward in time;
  only the slice within ~10 cm of range of the punch's closest approach is contact-eligible (a
  jab converging from a low guard crosses the silhouette laterally half a metre out, where
  nothing can touch); the earliest eligible point inside the head silhouette is the impact, with
  crossings interpolated onto the silhouette and near-skims within a fist-width grazing on. One
  rule for every punch type, no branch to flicker: a hook with follow-through lands on the cheek
  it CAME IN through rather than wherever past the centreline its fist decelerated (solving from
  the stopping point put hooks on the wrong side of the face); a shallow hook — more forward
  drive than sweep, i.e. every real hook at a laptop — lands on its entry side (an earlier
  sweep-must-dominate-depth gate sent exactly those to their stopping point); an uppercut takes
  the chin; a jab lands where it arrives. Two refinements close the truncated-evidence cases:
  (1) the contact instant is the FIRST arrival into the deepest range zone, not the deepest
  sample — a follow-through hook's retraction can re-cross the closest region and dip the range
  lower, and anchoring on the global minimum hung the terminal window and the direction on the
  pull-back, reversing the reported travel; (2) when blur ate the entry-side flight entirely
  (measured, as: almost no approach range observed before the path entered contact range — a
  guard-tracked punch shows 15-30 cm) and the punch classifies lateral, the entry is
  reconstructed backward from the first measured point along the arrival direction to the
  silhouette — where the face would have stopped the fist. The app then snaps onto the real mesh
  by raycasting along the travel direction. Two earlier formulations died here for the record: a
  normalised direction ray (no magnitude — a jab's lateral direction is pure noise, entries
  teleported to either cheek) and a sweep-vector walk from the apex (solved from the trajectory
  END).
- **Type**: classified once, from full-window evidence: terminal direction, knuckle normal
  (averaged over valid terminal samples), path curvature (velocity-direction rotation between the
  first and last trustworthy probes — measured ~0.9 rad for a synthetic hook against ~0.05 for a
  jab and ~0.1 for a wide straight, so the 0.2 rad auxiliary threshold separates them cleanly),
  entry origin (below head → uppercut prior, corroborated), and the punch's net **image sweep**.
  Truncated uppercuts get two dedicated carriers: the rise is low, fast, blurred and often below
  the frame, so the landmarker locks on only at the top where the upward velocity is spent — the
  measured direction reads near-pure −z and every travel-gated branch fails. But the fist's
  ORIENTATION is measured at the apex, exactly where tracking is good, so emphatic knuckles-up
  (n.y > .55, a jab's wrist→knuckle axis points at the target however the palm rotates) carries
  the classification with travel only required not to contradict; a strongly upward image sweep
  does the same. The classification then carries the entry: a truncation-reconstructed uppercut's
  arrival walks DOWN to the chin, since the class itself asserts the punch came from below.
  The sweep is the blur-proof feature: a hook crosses a third of the frame laterally even when
  every depth estimate is garbage and a censored punch's direction degraded to its chord, so it is
  what keeps a blur-censored hook a hook. Overwhelming lateral travel is checked before any
  uppercut prior — a rising hook must not fall into an uppercut branch on its upward tilt.
- **Hand** (left/right of the *puncher*): a weighted vote in which physically-grounded geometry
  outranks MediaPipe's guess — (1) the **wrist trail**: the forearm exits toward its own shoulder,
  measured as the windowed mean of the METRIC lateral wrist-minus-knuckles offset (metric, not
  image-space: the wrist is deeper than the knuckles and perspective drags deeper points toward
  the image centre, which reads as the wrong hand for any laterally-offset fist); (2) **metric
  chirality** of the fitted hand — the triple product of wrist→index/pinky/thumb, the one property
  no rotation can disguise (a right hand measures ~+0.18 in the sign-resolved fit frame), averaged
  over the window; (3) entry side, cast ONLY when trustworthy — a slow guard-anchored birth or a
  genuine outer-edge entry; a hook first detected mid-arc near frame centre must stay mute here,
  which is exactly how right hooks used to get logged as left; (4) travel azimuth (a right hook
  drives toward head −x), weighted up for strongly lateral punches; (5) the flip-corrected
  MediaPipe label as the weakest vote. The per-event tallies ship on the event as `handVotes`.

### Conventions (fixed facts of the setup, not settings)

- getUserMedia frames are ASSUMED unmirrored (the standard for built-in webcams); the puncher's
  right hand sits at low u. Scene x is the puncher's screen-right. Hence one x negation between
  camera frame and head frame, applied in exactly one place (`toHead`, at event assembly).
  Knuckle normals and velocities leave the pipeline already head-frame; consumers never flip
  anything.
- Some cameras (virtual cams, certain drivers/Continuity setups) deliver MIRRORED frames — and
  this cannot be auto-detected, because a mirrored world is geometrically self-consistent: fitted
  chirality, the wrist trail, sweep direction, entry side, and MediaPipe's label (itself computed
  from image geometry) all flip coherently. The live signature is every hook naming the wrong
  hand and landing on the wrong cheek. `TargetTracking.setMirrored(true)` declares the
  interpretation and un-flips every observation at the shell boundary; the cv-debug page exposes
  it as the persisted "Feed: RAW / MIRRORED" toggle.
- Lateral offset is FOV-independent (offset = (2u−1)·z·k with z ∝ 1/k), so impact *location*
  needs no camera calibration; speed scales with the FOV assumption and is a single global gain.
- Capture timestamps never run backward; a large rewind (worker restart, test harness re-drive)
  is a session reset and stale future-stamped slots must not adopt the new stream.
- When the pipeline says hit but the app's mesh raycast finds no surface (the ellipsoid and the
  mesh disagreeing about the silhouette), the app falls back to the ellipsoid entry point instead
  of contradicting a decided hit with a phantom miss — the old app-level duplicate-miss timer is
  gone entirely.

## What was kept

`fist-pose.js` (rigid fit) unchanged; `fist-filter.js` still serves the POV render path;
`strike-system.js` untouched (POV/app authority when the target camera is off); the
`TargetTracking` shell API (start/stop/tick/knobs/probe) and the stage-by-stage probe readout;
the head-frame mapping and `entryPoint` ellipsoid; all behavioural tests, ported.

## Invariants the tests pin

- One physical punch → exactly one event, under handedness flicker, track fragmentation,
  landmark dropout at the apex, and any single-frame noise.
- Retraction, guard drift, slow reaches, twitches, and constant-range lateral sweeps never fire.
- Jab/hook/uppercut land on front/cheek/chin respectively, with direction read at contact
  (a hook's direction is its terminal tangent, not the mid-arc chord).
- A jab–cross combo ≥ ~250 ms apart is two events.
- Impact location is independent of the assumed FOV; punch-to-your-right lands at scene +x.
- Reported speed is the punch at speed (peak fitted closing), not the stopped fist.
