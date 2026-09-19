# NFR-inspired facial impact deformation

The subsequent visible wince/after-impact expression is documented in
[PAIN_REACTION.md](PAIN_REACTION.md). The comparison page now demonstrates that layer.

The impact path now reconstructs a connected displacement field from bounded triangle
deformation gradients before applying the existing displacement, orientation,
accumulation and recovery constraints. It runs for every topology-backed `FaceImpactRig`
punch, including preview dynamics, Newton, directional contacts and clay mode.
The runtime hook is in `src/impact-rig.js`; the solver is in `src/deformation-gradient.js`.

## What comes from NFR

Reference: Qin et al., [Neural Face Rigging for Animating and Retargeting Facial Meshes
in the Wild](https://github.com/dafei-qin/NFR_pytorch), commit
`664488e224333deb46f85279b54b6dd1d02e81b1`.

The inspected upstream `deformation_transfer.py` constructs local triangle frames,
encodes deformation as Jacobians, and recovers vertex positions by an area-weighted
global least-squares solve. This implementation carries that gradient-domain
reconstruction idea into browser JavaScript. The source is pinned in
`scripts/third_party_manifest.json`, checked out under `.local/third_party/NFR_pytorch`,
and its MIT notice is retained in `third_party/licenses/NFR-MIT.txt`.

This is an original browser adaptation of the geometric stage, **not pretrained NFR
inference**. NFR's neural identity/expression encoders, learned FACS-like codes, datasets
and pretrained weights are not installed. The published runtime imports CUDA/CuPy,
which does not suit this Mac. Existing punch fields provide the target deformation;
jaw/smile sliders and exported expression morphs remain their existing authored rig.
No Python environment or dependency was changed.

## Solver

1. Use the existing welded tissue topology so texture seam copies share a solution.
2. Precompute each triangle's orthonormal rest frame and piecewise-linear gradient
   operator. Degenerate faces and extreme slivers are excluded from this solve but
   still checked by the downstream full-mesh validity guard.
3. Differentiate the requested displacement and add the rest tangents. Project the
   resulting 3-by-2 Jacobian's principal stretches into `[0.60, 1.45]`, preserving its
   rotation. The range is an animation setting, not a tissue measurement.
4. Integrate the adjusted gradients with a screened Poisson solve:

   `min_u sum_t area_t ||G_t u - (F_t - I)_tangent||² + sum_i mass_i / length² ||u_i - target_i||²`

   The attachment length is 12 mm. Area weighting avoids letting dense tessellation
   dominate; positive attachments fix translation and independently anchor every
   component, including isolated invisible cage landmarks. Corrections spread along
   the connected surface without coupling nearby disconnected layers.
5. Solve in Float64 with diagonally preconditioned conjugate gradients (relative
   residual target `1e-3`, maximum 100 iterations). A nonfinite result or residual above
   `0.02` retains the original authored field. Existing safety projection then runs in
   either case. Singular-value bounds apply to the requested local transforms; the
   reconstructed field is a least-squares compromise, not a hard guarantee on every
   final triangle's stretch.

The matrix is built lazily once per rig. The solve happens on a new impact; playback
uses the existing precomputed fields, temporal envelope and plastic commitments.
No per-frame global solve is added. Zero input and rigid transforms are unchanged;
rest positions, texture coordinates and identity geometry are not smoothed. A collapsed
Jacobian has no unique rotation; the existing area guard remains necessary. This is
surface animation, not a volumetric fascia or collision model, and it does not establish
biomechanical accuracy.

## Review and verification

Start the usual `npm run dev`, then open
[the local comparison](http://127.0.0.1:5173/rig-review.html).
It loads the latest saved photo model through the version-checked head bundle. Both
sides now keep the gradient stage, with the pain reaction disabled on the left.
Select either hook, uppercut or frontal contact; inspect the timeline, replay recovery,
orbit, and switch photo/wireframe views. The comparison
isolates surface correctives; hair, glasses and remote Newton offsets are visible in
the full app instead. This page is a development entry, not included in the default
production build.

Gradient-only validation on 2026-09-19, before the pain-expression layer:

- `FACE_IMPACT_CAPTURE=.local/face-captures/<local-id> npm test`: **99 passed**, zero
  skipped, including the actual captured-head repeated-dent regression.
- `npm run build`: passed.
- Added mathematical checks for rotation-preserving principal stretch limits,
  unchanged neutral/rigid motion, redistribution of a sharp local pull, seam
  coincidence, isolated cage points and no transfer to a close disconnected layer.
- Rendered comparison on the 19,789-position / 38,630-triangle capture: left hook
  adjusted 1,921 local transforms; right hook adjusted 1,894. Maximum correction before
  final safety projection was 2.12 / 2.18 mm. Both held comparison poses had zero
  reversed triangles. The visual difference is subtle at normal scale.
- Full app with Newton: held left hook at magnitude 0.85 reached **41.9 mm** peak,
  with finite positions and **zero reversed triangles** (minimum area ratio 0.2002 in
  the inspected held frame). The added solve converged in 57 iterations.
- Final rendered right hook reached **37.5 mm** peak with zero reversed triangles.
  After release, residual live motion fell to 0.0055 mm with zero permanent damage;
  Reset head reduced it below 0.000005 mm (Newton continues stepping). Uppercut review
  corrected 470 local transforms by at most 0.49 mm, with zero reversed triangles.
- Observed browser solve cost was approximately **50–72 ms per impact** on this capture,
  excluding first-time matrix construction. It is synchronous and can cause a short
  contact-time hitch; no 60-fps claim is made for impact preparation.

Captured geometry and photos remain local. No new capture, uploads, model weights or
external inference were required. Existing reconstruction/texture defects are outside
this deformation change.
