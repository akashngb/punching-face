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
landmarks exist, a degraded sample keeps the trajectory alive — but ONLY while a recent rigid
fit can lend it depth. Its depth is DEPTH-INERTIAL, carried from that anchor, never derived from
the apparent span: apparent size conflates rotation with distance by construction (a stationary
fist rotating as the elbows lift halves its knuckle span, which span-depth read as a
tens-of-centimetres phantom approach-and-retreat; blur widening it the other way fabricated
superhuman closing speeds).

With no anchor there is no honest depth to be had, so **nothing enters the trajectory**: the hand
is merely OBSERVED (`observe()`), which keeps its identity alive — association prediction,
liveness, rest lineage — without inventing a position. Being tracked and being measured are
different things, and the tiers now say so. The old anchor-less `loose` tier guessed depth from
apparent knuckle width in exactly this situation; since a punch lasts ~200 ms while "no rigid fit
for 400 ms" only happens to a fist parked at the lens, it never once carried a real punch — it
only fabricated approaches out of fingers clipping the frame, at about one phantom jab per second
for as long as a fist was held out. It is gone.

**Motion stream** (coarse, blur-proof): frame differencing at 96 px wide → connected components →
up to 2 blobs with centroid, mass, rms spread, block-matched mean translation and expansion rate
(similarity-flow fit). Motion blur *helps* this stream. It corroborates approaches, dates
reversals when landmarks are gone, bridges dropout gaps with low-weight samples, and — critically
for dedup — **steers identity**: each slot keeps a blob-informed predicted position (the blob's
measured flow updates the velocity, its centroid the position), so when landmarks die mid-arc the
prediction follows the fist along the arc instead of extrapolating the stale pre-blur velocity.
Reacquired landmarks then land back inside the association gate of the identity they belong to,
which is what stops one punch forking into two identities in the first place. Blob centroids are
arm-biased, so they steer prediction only; the slot's measured position and landmark liveness
stay landmark-owned. The stream never fires an event by itself.

### Slots, not tracks

At most two persistent **slots** hold time-ordered sample buffers. Association is
predicted-position nearest-neighbour: the position comes from the blob-informed prediction (which
follows a curving, blur-dead fist — see the motion stream above), while the gate grows with
*landmark* silence, because identity uncertainty grows while nobody has actually seen the hand,
however confidently the motion stream tracked something. A slot is a *bucket of evidence*, not an
event owner: an association mistake means some samples land in the other bucket, and the damage
is contained by candidate arbitration below — there is no per-frame decision that can
double-fire. MediaPipe handedness never keys identity (the front of a fist is chirally
ambiguous); it is only one weighted vote in per-event hand classification.

Slots also learn their **rest position**: a fist sustainedly slow in the image (many consecutive
slow samples — an apex hold is a few frames, a guard is many) accrues an EMA of where it rests.
This is the hand-lineage evidence the classifier leans on below.

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
  carries the trajectory to a measurable reversal or a timeout closes it — a timeout running from
  the last EVIDENCE, blob bridge included, not from landmark death: while blobs still carry the
  trajectory the punch is not lost, it is heading for a bracketed apex or for landmark
  reacquisition into the same slot. Censoring on landmark silence alone decided the event ~160 ms
  into the blur, from half the punch, and the better-solved continuation then had to be
  deduplicated away.
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
- **Speed is a windowed linear slope over rigid samples.** The quadratic local fit is right for
  trajectories but wrong for sparse noisy speed reads: three rigid points ~100 ms apart fit a
  quadratic exactly, so 1 cm of depth noise becomes metres-per-second of phantom slope — a
  1.5 m/s jab reported anywhere from 1.1 to 3.4 depending on frame phase, and a real jab was
  rejected at "peak 1.13 < 1.20". The linear window reads the punch's sustained closing —
  stable, honest, ~25% under the true instantaneous peak — and the speed gates are calibrated to
  it (`minPeak` 1.0, `EDGE_PEAK` .35). Lazy reaches measure 0.2–0.5 on the same scale.
- **The approach must be measured, not manufactured**: the range a candidate claims to have
  closed must be substantially witnessed by MEASURED samples — rigid fits plus anchored `span`
  fallbacks, whose lateral is honest and whose frozen depth can only understate closure (≥ 40%
  of the claim, or a full unsoftened `minTravel` on measured evidence alone; rigid-only
  witnessing starved blurred hooks, which close range mostly laterally). Blob expansion may
  CARRY an approach through blur — a real punch is rigid-tracked
  until its last ~100 ms, and measured samples at the start of the retraction still witness how
  deep it got — but they can never constitute one. The same rule governs the re-arm: the withdrawal that lets a fist punch again
  must be rigid-OBSERVED, because fabricated depth swings used to clear it and let the next
  fabricated dip fire.
- **A punch ARRIVES**: its apex must come within `strikeRange` (default 45 cm) of the head, with
  censored candidates judged on imminent arrival (trailing closing × ~120 ms). Approach shape
  alone is not sufficient evidence — any inward lateral guard shift closes range geometrically
  (by the lateral slack r−z, 6–11 cm from a normal guard, squarely inside the travel gates) and
  brackets itself on the shift back, but its range minimum is the guard distance itself; no jab,
  hook or uppercut terminates half a metre from the face.
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
never reaches back across the consumption boundary. Across slots, another view of a just-emitted
strike is recognised two ways:

- **Image support** (< 280 ms, < 0.22 image units): a candidate at the same spot as the event is
  the same fist seen twice — the duplicate-detection case, where MediaPipe reports one fist as
  two overlapping hands.
- **Temporal exclusivity** (< 400 ms): blur can displace the image position arbitrarily far
  between two sightings of one fist — a hook crosses a third of the frame while its landmarks are
  dead — so no image gate can catch a *sequential* fragment. But blur cannot make one fist appear
  twice AT ONCE: two real fists coexist on screen, fragments take turns. A candidate whose
  identity never coexisted with the identity that fired (landmark spans overlapping < ~60 ms) is
  the same fist re-tracked, with one exemption: when BOTH sightings independently observed a
  complete punch (approach → apex → sustained retreat) they are two punches thrown through a
  tracking dropout — a real double — while a censored or instant fire means the evidence stopped
  mid-punch and whatever continues it belongs to it. This is what lets a jab–cross combo whose
  detections alternate (only one hand tracked at a time) stay two events while a
  censored-fragment-plus-reacquired-landing stays one.

Misses (trajectory crosses wide of the head) are decided by the same extractor and take part in
the same arbitration, which is what retires the old app-level "a miss right after a hit is the
same punch" patch.

One zone rule sits after the solve: an event arriving BELOW the head (contact or aim under
~1.15 × the head's half-height) is a chest/shoulder-height strike, and there is no head there to
hit — it is ignored outright (no hit, no miss, no log line; counted as `low` on the readout).
It still consumes its evidence and enters arbitration memory, so its fragments cannot resurface.
The check runs after classification steered the entry, deliberately: an uppercut travels THROUGH
chest height, and only its classification (which walks the arrival to the chin) proves it was
not a body shot.

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
- **Type**: classified once, from full-window evidence — and the evidence is **image-first**.
  The image plane is what this camera measures best, so where the punch's net **image sweep**
  (measured over the raw windup, so a mid-approach gate rejection cannot clip it) and the fitted
  3D direction disagree, the sweep wins: a hook translates across the image, an uppercut climbs
  it, a jab barely moves — it looms. The fitted direction is depth-starved exactly at contact
  (rigid fit dead → inertial depth → velocity-z collapses → the terminal direction degrades to
  pure lateral/vertical noise), which is how straight jabs read live as hooks and uppercuts.
  **Travel itself is solved on rigid samples only.** A fallback sample's depth is frozen, so a
  velocity fitted through one reads dz ≈ 0 — and for a straight punch, whose entire velocity IS
  dz, the normalised direction then collapses onto leftover lateral/vertical noise, which reads
  as a rise or a sweep. Jabs were logged live as uppercuts and hooks for exactly this reason.
  Position may use every sample (image position is honest in all tiers); velocity may not.
  Hence two hard rules: **no hook verdict without image support** (sweep ≥ .06 image units —
  depth-starved noise moves the image a few hundredths; a truncated hook still sweeps ≥ ~.09
  even when the apex rewind collapsed its window), and **upward image motion outranks every
  orientation prior**. The sweep itself is **divergence-corrected** first: an approaching fist's
  image position diverges radially away from the frame centre (u−.5 ∝ x/z), so a jab above the
  low-slung laptop camera "rises" ~.2 image units with zero vertical motion — the outward-radial
  component is looming, not travel, and is removed before classification (skipped near the
  centre, where divergence is negligible).

  **Where the fist POINTS is measured in the image, not in 3D.** A jab arrives with its knuckles
  square at the camera and an uppercut with them up, so the wrist→knuckle axis foreshortens to
  nearly nothing for one and stands tall up the frame for the other — the single most obvious
  difference between the two punches. It must be read in the image plane: in 3D that axis lies
  along the view axis for a straight punch, i.e. almost entirely in the DEPTH component, and
  MediaPipe's depth compression shrank it until the small honest vertical dominated after
  normalisation, reading jabs as uppercuts. The shell measures it from raw landmarks in
  fist-widths (`axis`), the core averages it over the terminal window (`handAxis`), and on real
  geometry a fist aimed at the camera reads ~0.16 while an uppercut reads 0.4–0.9. The middle
  band is left UNDECIDED, deferring to travel.

  **No uppercut verdict without image-plane evidence of a rise** — an upward sweep across the
  frame, or an entry from below the head. Nothing depth-derived may substitute. Three separate
  live misclassifications had one shape: a straight jab at 10–14 cm where the fitted direction,
  the 3D knuckle normal and (through perspective) even the image hand axis all tilted "up" while
  the punch demonstrably never rose in the frame. Rising up the frame is what an uppercut IS, so
  that is the signal to require. The one exemption is orientation from a TRUSTWORTHY distance:
  inside ~25 cm the fist subtends a huge angle and its wrist sits far behind its knuckles, so
  every orientation reading tilts upward whatever the punch is doing; beyond that range
  orientation is sound, which is what still recognises a top-only uppercut whose rise happened
  below the frame.

  **Which FACE of the hand shows completes the shape** (`facing`: knuckle-row span relative to
  hand scale, shipped with the axis). A guard fist is ALSO vertical — same axis as an uppercut —
  but it is seen EDGE-ON: back of the fist to the side, so the knuckle row points away from the
  camera and its apparent span collapses. Measured: guard ~.11 (a 20°-turned guard still only
  .40) against 1.0 for every face-on punch, so the .55 threshold has real margin. Uppercuts and
  hooks both show the BACK of the fist; guard shows its edge — so no orientation verdict is
  allowed from an edge-on hand (idling fists in guard were firing "uppercut" purely on their
  vertical axis; the axis measure even divided by the collapsed span, inflating it). The same
  measure gives hooks their hand-shape carrier: a face-on fist with knuckles to the SIDE
  (|du| > .55) plus lateral travel is a hook even when the apex rewind collapsed its sweep, and
  it counts as hook image-evidence alongside the sweep.

  **Orientation may not overrule clean travel**, though. Near the lens the fist fills the frame
  and perspective is extreme: the wrist sits much farther from the camera than the knuckles, so
  it projects low and the measured axis "stands up" even for a dead-straight jab — observed live
  at 10 cm range, logged LEFT UPPERCUT with the knuckles plainly facing the camera. So when the
  fitted travel says the punch drove straight in (dz < −.80, |dy| < .30, lateral < .45), an
  upward axis needs corroboration from an independent sign of a rise — entry from below, or an
  upward image sweep — before it can call an uppercut. A real uppercut supplies one of those even
  when blur truncates its travel, which is what keeps truncated uppercuts registering.

  A weaker fallback veto applies when no image axis is available (blob-only windows):
  **a fist pointing more FORWARD than up (n.y < |n.z|) is never an uppercut** — the knuckle normal is the wrist→knuckle axis, which
  aims at the target on a straight punch and upward on an uppercut. An earlier veto band
  (n.y < .35) left a gap: ordinary jab form angles the fist up 30–50°, putting n.y at .45–.75,
  unvetoed and close enough to the orientation branches to tip a jab into 'uppercut' on noise.
  Each classification branch is named, and the branch that decided ships on the event as `why`
  (shown in the impact log), so a misclassification reports which evidence convinced it. The 3D direction, knuckle normal, curvature (~0.9 rad for a hook vs
  ~0.05 for a jab) and entry origin (below head → corroborated uppercut prior) then resolve
  whatever the sweep left ambiguous. Truncated uppercuts keep a dedicated carrier: the rise is
  low, fast, blurred and often below the frame, so the landmarker locks on at the top where the
  upward velocity is spent — there, EMPHATIC knuckles-up (n.y > .75; ordinary jab form angles
  the fist to ~.5–.7, which at the old .55 threshold read jabs as uppercuts) carries the class
  with the image only required not to contradict. The classification then carries the entry: a
  truncation-reconstructed uppercut's arrival walks DOWN to the chin, since the class itself
  asserts the punch came from below — which also decides whether the below-head ignore rule
  applies, so classification quality gates registration for uppercuts.
- **Hand** (left/right of the *puncher*): a weighted vote in which physically-grounded geometry
  outranks MediaPipe's guess. The two strongest votes are not in the punch at all — they are
  **lineage**: (0a) where this identity *rested* before it flew (the slot's learned rest
  position; fists guard on their own side, the puncher's right at low u, and the guard was
  observed while tracking was good, however badly the punch itself blurred), and (0b)
  **elimination**, cast only when the striker has no rest history of its own (a mid-flight
  birth): a fist visibly resting somewhere else right now is not the fist that just landed, so
  the punch belongs to the other hand. The within-window votes follow —
  (1) the **wrist trail**: the forearm exits toward its own shoulder,
  measured as the windowed mean of the METRIC lateral wrist-minus-knuckles offset (metric, not
  image-space: the wrist is deeper than the knuckles and perspective drags deeper points toward
  the image centre, which reads as the wrong hand for any laterally-offset fist); (2) **metric
  chirality** of the fitted hand — the triple product of wrist→index/pinky/thumb, the one property
  no rotation can disguise (a right hand measures ~+0.18 in the sign-resolved fit frame), averaged
  over the window; (3) entry side, cast ONLY when trustworthy — a slow guard-anchored birth or a
  genuine outer-edge entry; a hook first detected mid-arc near frame centre must stay mute here,
  which is exactly how right hooks used to get logged as left; (4) travel azimuth (a right hook
  drives toward head −x), weighted up for strongly lateral punches; (5) the landmarker's label
  as the weakest vote — NOTE: the vendored bundle (tasks-vision 0.10.32) names the PHYSICAL hand
  on an unmirrored feed, i.e. it does NOT apply MediaPipe's documented selfie assumption
  (verified live: interpreting it selfie-style swapped every overlay letter); declared-mirrored
  feeds are label-flipped at the shell before the vote. The per-event tallies ship on the event
  as `handVotes`.

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
  landmark dropout at the apex, any single-frame noise — and however long the fist is HELD at
  full extension afterwards (fabricated depth neither qualifies a punch nor clears a re-arm).
- Retraction, guard drift, slow reaches, twitches, and constant-range lateral sweeps never fire.
- Guard shifting — lateral motion at unchanged distance — never fires: a punch must arrive
  within `strikeRange` of the head.
- Chest/shoulder-height arrivals are ignored outright; only the head can be hit.
- Jab/hook/uppercut land on front/cheek/chin respectively, with direction read at contact
  (a hook's direction is its terminal tangent, not the mid-arc chord).
- A jab–cross combo ≥ ~250 ms apart is two events.
- Impact location is independent of the assumed FOV; punch-to-your-right lands at scene +x.
- Reported speed is the punch at speed (peak fitted closing), not the stopped fist.
