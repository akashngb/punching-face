# Directional facial impact rig

The NFR-inspired gradient reconstruction added to this path is described in
[NFR_RIG.md](NFR_RIG.md), including its source attribution, comparison view and limits.
The visible wince, timing and expression validation are described in
[PAIN_REACTION.md](PAIN_REACTION.md).

The Impact lab and `window.__punchingFace.applyImpact()` use the same rendering and physics path as local and remote contacts. The old webcam detector and LiveKit hook remain compatible.

## CV integration

```js
const landed = window.__punchingFace.applyImpact({
  location: [-0.055, -0.005, 0.060], // metres; use the actual model surface
  direction: [0.8, 0.15, -0.6],    // travel INTO the skin; normalized internally
  magnitude: 0.75,                 // finite number in [0, 1]
  space: 'head',                   // default: head-local; alternatively 'world'
});

// The event interface accepts the same payload:
window.dispatchEvent(new CustomEvent('face-impact', { detail: {
  location: [x, y, z], direction: [dx, dy, dz], magnitude: 0.8, space: 'world'
}}));
window.__punchingFace.setHeadMode('clay'); // or 'live'
```

Locations and directions can also be `{x,y,z}` objects. The head uses Y up and +Z forward. `world` coordinates are transformed through the head's current pose, including recoil, heading, and tilt. Convert camera/pixel coordinates in the CV layer: a screen coordinate is not a head-space point. Locations within 45 mm of the visible surface are snapped to its nearest vertex; farther points return `false`. Invalid vectors, non-finite values, unknown coordinate spaces, and out-of-range magnitudes throw `RangeError`. Zero magnitude returns `false` with no impulse. A stopped Newton service rejects live contacts; clay remains local.

One call represents one contact, not one camera frame. Debounce a continuing contact in the CV detector. Direction is the incoming travel vector (not the outward surface normal). Existing legacy callers supplying `speed` retain their mapping, capped at 0.90 so an uncalibrated speed cannot silently trigger permanent damage. The normalized API, impact lab and demo strength slider can reach 1.0.

## Behavior and controls

- **Live head:** smooth compression crest at 120 ms, followed by damped recovery over roughly one second of simulation time. Slow motion changes the playback rate. A magnitude strictly greater than 0.90 adds retained displacement weighted by estimated bone support and normal incidence. Exactly 0.90 remains elastic. The threshold is a product rule, not a fracture predictor.
- **Clay head:** the impact field is committed during the onset and never decays. Further punches add deformation within the strain/displacement budget. Switching mode, releasing a held frame, and changing the display do not erase dents. Reset head clears all impact deformation and restores the original editable shape.
- **Magnitude:** zero to one, independent of softness. Increasing magnitude increases the contact footprint and displacement. Angle/elevation tilt the incoming vector relative to a selected region's normal. Drag to orbit and enable Click head to punch for arbitrary surface points.
- **Wireframe:** the actual triangles deform with the visible skin and texture. The tissue attachment map colors the heuristic support field (blue firm / coral mobile); it is not a segmented internal fascia scan. Rig markers follow the deformed surface.
- **Sessions/export:** editable sessions store impact mode and committed permanent displacement separately from the rest mesh. Saved sessions take priority during reload recovery. GLB export bakes the retained deformation while keeping expression morphs. Transient elastic peaks are not baked into the exported rest pose.

## Implementation

`src/tissue-field.js` welds coincident seam vertices into a surface graph and excludes isolated, unrendered cage landmarks from contact selection. Surface distances spread the contact footprint along triangles rather than through the head. The field separates normal indentation, broader tangential transport and a raised rim. Landmark fields vary compliance across cheeks, lips, nose bridge, zygomatic regions, mandible, temples and skull. Direction/contact-weighted cheek and chin correctives couple the mouth and jaw to the strike. Recoil uses impact position crossed with direction.

`src/impact-rig.js` manages transient and plastic layers. Individual fields have bounded displacement gradients. New plastic increments are limited against already committed and queued dents. Local triangle-area projection alternates with a constraint preserving existing dent direction, followed by a line search relative to the previous dent. This avoids erasing an earlier dent to repair a new impact. The cumulative displacement cap is 65 mm; it is an animation limit, not a measured tissue parameter. Geometry remains connected across welded UV seams. At capacity, queued plastic commitments are retained. These are animation constraints, not a volumetric constitutive solver or a guarantee against all self-intersections.

`src/physics.js` combines the field with preview tissue springs. `src/newton-dynamics.js` combines the same field with the existing remote Newton offsets for live heads. Clay does not consume elastic offsets. Reset/mode changes invalidate stale remote responses. The pipeline's observed-front Newton cage is unchanged; coverage outside it comes from the browser's surface field.

## Research and visual targets

The supplied photos were used qualitatively: broad cheek transport, compressed lips, lower-face/jaw motion, asymmetric eyelid compression, and head rotation. A single still photo does not specify force, deformation time, skin thickness, or a subject's skull.

Primary references consulted:

- Warburton & Maddock, *Physically-based forehead animation including wrinkles*: layered tissue and sliding constraints over a rigid skull. [University of Sheffield repository](https://eprints.whiterose.ac.uk/id/eprint/114904/)
- Ichim et al., *Phace: Physics-based Face Modeling and Animation*: separate skin, bone and jaw structures, including tissue attachments and sliding constraints. [Authors' paper](https://users.cs.utah.edu/~ladislav/ichim17phace/ichim17phace.pdf)
- Lee, Terzopoulos & Waters, *Realistic Modeling for Facial Animation*: coupled skin, muscle and skull models. [Authors' paper](https://web.cs.ucla.edu/~dt/papers/siggraph95b/p55-lee.pdf)

The implementation is original heuristic surface animation informed by these principles; it does not implement these papers' full solvers. No additional libraries, model weights, medical assets or license acceptances were needed. Fable was requested, but no callable Fable integration or installed CLI was available, and plugin discovery returned no Fable result. Research used primary web sources and verification used automated tests and the local rendered app instead.

## Validation and limits

`tests/directional-impact.test.mjs` covers contract validation, zero/graded magnitude, front/side/crown/back/chin contact, recovery, strict damage threshold, persistent and cumulative clay, mode changes, save/restore of dents, welded seams, repeated-hit bounds, and the invisible cage-point regression. Existing physics, Newton adapter, expression, and sponsor-hook tests also run. `tests/captured-impact.test.mjs` accepts `FACE_IMPACT_CAPTURE=/path/to/local/capture` for four consecutive hits on the actual fitted photo head, including monotonic indentation, triangle orientation, mode preservation and reset; private capture data is not checked in.

The photo model still has estimated geometry and inferred tissue/bone locations. There is no subject-fitted skull, volumetric fascia, ligament segmentation, bone fracture mechanics, skin self-collision, or tooth/lip contact solve. The result is a more controllable facial-impact animation prototype. Hyperrealism and anatomical/clinical accuracy have not been established. `impactQuality` reports finite geometry, triangle areas and normal reversals relative to rest; it is an engineering diagnostic, not clinical validation.
