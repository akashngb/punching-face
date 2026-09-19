# Facial reaction after a punch

The earlier NFR-inspired change limited local stretching but did not add the expressive
response the user wanted. `src/pain-expression.js` now supplies a separate, visible
wince to topology-backed live heads through `FaceImpactRig` in `src/impact-rig.js`.
The same path runs in preview physics, Newton, local punches and remote contacts.

## Visible behavior

- Eyelids tighten toward their own upper/lower midpoint.
- Inner brows draw down and inward, with a slightly stronger response on the struck side.
- Cheeks lift, the upper lip raises, and mouth corners pull down asymmetrically.
- The lower face moves around a jaw hinge after contact begins.
- Expression starts after 25 ms, reaches full weight at 170 ms, holds through 550 ms,
  and eases out by 1.65 seconds of simulation time. Slow motion stretches wall-clock
  playback. These are authored animation timings, not measured reflex latency.
- Hold peak waits for the expression crest on live heads. Clay still holds the original
  contact crest and has no facial reaction. Reset clears both transient layers.

The action choices are informed by Prkachin's [comparison across pain modalities](https://pubmed.ncbi.nlm.nih.gov/1491857/),
which reports brow lowering, eyelid tightening/closure and nose/upper-lip actions.
This is an authored expression, not a diagnosis, a measurement of pain or a reproduction
of a particular person's response. No additional models, training data or libraries
were downloaded. It is separate from the NFR gradient solver and does not claim to be
NFR neural inference.

## Geometry and state

The expression uses the head's landmarks. Photo heads also provide upper/lower lid
and inner-brow points in their verified, unrendered 468-point cage. Other meshes use
the existing rig's anchors and proportional fallbacks. The largest connected tissue
component supplies the skin mask; detached eyeballs and invisible cage nodes are not
deformed by the new expression. This assumes a connected head skin; fragmented imports
may need a more detailed semantic mask.

The impact and expression have separate envelopes. Precomputed contact-only,
expression-only and combined poses are blended with nonnegative normalized weights.
The combined poses receive triangle-area protection. Expression closure is not passed
through the contact field's isotropic gradient limit, which would suppress the wince.
The original contact-only field still runs through the NFR-inspired stage.

Expression never becomes a permanent dent. Clay and the existing strict damage
threshold retain their behavior. Rest geometry and saved identity remain unchanged.
The reconstructed mouth has no rigged interior, so the result is a grimace with jaw
motion; it does not invent teeth, a tongue or an anatomically correct open mouth.

## Review and verification

[rig-review.html](http://127.0.0.1:5173/rig-review.html) now compares the previous
contact-only response against the same contact with the pain layer. Both sides keep
the NFR-inspired correction. It starts at 0.35 seconds so the difference is apparent.
Replay shows compression, the lingering wince and recovery; the time slider can inspect
the whole sequence. The review omits Newton offsets and accessories to isolate the
facial response. The main app includes those layers.

Verified 2026-09-19:

- **103 tests passed, zero skipped**, with `FACE_IMPACT_CAPTURE` set to the local
  19,789-position, 38,630-triangle photo head. Production build and diff checks passed.
- Both full-strength hooks and the uppercut were sampled every 20 ms for 1.8 seconds:
  finite geometry, zero reversed triangles, and no crossing at the tested upper/lower
  eyelid anchors. This is sampled engineering validation, not a self-collision proof.
- At the 0.36-second reaction sample, both eyelid gaps were below 55% of their rest
  height, inner brows lowered over 4 mm, and both mouth corners lowered over 5 mm.
- Tests also cover delayed onset, lingering expression after contact recovery, complete
  recovery, paused-time stability, Reset, strength scaling, clay exclusion and fixed
  disconnected eyeballs/cage points.
- Inspected the textured left/right comparison and full app with Newton. The held
  left-hook pose in the full app had finite positions and zero reversed triangles
  (minimum area ratio 0.1638 in that inspected frame).

Commands:

```sh
FACE_IMPACT_CAPTURE=.local/face-captures/<local-id> npm test
npm run build
```
