# Video, face rig and hair reconstruction

The current capture uses 51 saved video frames and 40 registered cameras. A
cached `gpt-6-astra` pass annotates seven front, profile and rear views with hair
regions, broad flow paths and style parameters. This is the configured vision
model, not a benchmarked claim that it measures individual hairs best.

## Processing order

1. **Extract and align.** The browser saves 540 × 960 masked frames, timestamps
   and face landmarks. Camera recovery establishes their relative angles.
   `photo_detail.py` finds the corresponding 1080 × 1920 original video frames,
   checks their alignment, and retains their additional fine detail. Browser
   decoded colour is preserved because native phone-video decoding can change
   HDR colour and exposure.
2. **Fit the face and rig.** Triangulated landmarks fit the head template. The
   existing face rig and observed facial vertices remain fixed during a hair-only
   rebuild. Camera calibration and unobserved head anatomy remain estimates.
3. **Fit hair shape.** Registered silhouettes initialize the outer volume.
   Version 5 treats the broad crown-height bound as an initialization, then fits
   visible contour vertices to local photographed outlines. A coupled XYZ solve
   preserves quiff asymmetry and visible peaks, with smooth, bounded corrections
   where the crown is unobserved. Face and ear vertices remain pinned. Triangle
   area, orientation and intersection checks back off unsafe shape proposals.
4. **Recover visible detail.** Hair masks, visibility and pixel orientation guide
   visible lock sections. Version 5 adds no unsupported crown roots and no
   generic sinusoidal waves. Each station samples photographed colour. Individual
   fibers remain modeled detail, while hidden crown texture uses a labeled
   photographic continuation.
5. **Bind and bake.** Every hair-curve station is projected onto its local mesh
   triangle before sampling color. The same barycentric weights drive deformation,
   preventing the old approximate tangent planes from burying fibers in the shell.
   Fine fibers have a small, editable relief above the surface. Glasses cleanup uses
   reviewed opaque-frame ribbons; a registered generated reference can replace
   the obscured region and is labeled estimated. Unreviewed or tinted source
   views cannot overwrite the cleanup. Broad colour is
   blended; fine hair, eyebrow, beard and moustache texture comes from one visible
   camera. Captured colour and lighting are retained without studio relighting
   at rest. Generated eye detail remains independently labeled and preserved.
6. **Compare.** The local review page shows photos and the model from the same
   recovered cameras, with photo overlay, fiber and shape-only toggles. Passing
   geometry tests is not evidence of photographic likeness.

## Rebuild a completed capture

```sh
.venv/bin/python scripts/rebuild_hair.py .local/face-captures/CAPTURE_ID --refit-envelope --rebake
.venv/bin/python scripts/prepare_hair_review.py .local/face-captures/CAPTURE_ID
```

Then open `http://localhost:5173/hair-review.html` while the dev server runs.
Review assets remain in ignored `public/generated/hair-review/`; the source video
and working outputs remain in `.local/`. Existing snapshots are preserved in
`before-hair-v3/`, `before-hair-v4/` and `before-hair-v5/`. A hair-only v5 rebuild
starts from its own compatible saved surface and refines local contours rather
than inflating the envelope again. The rebuild merges hair
updates into the latest mesh and refuses to overwrite concurrent surface edits.

`--recognize` refreshes the cached vision analysis only if needed and sends the
selected cropped images to the configured API. The changes above reuse the
existing analysis and installed local NumPy, SciPy, OpenCV and Three.js stack.
No additional ML weights, CUDA dependency or generated replacement hairstyle is
required. Earlier sessions remain readable.

## Verification

```sh
npm test
npm run build
.venv/bin/python -m unittest tests.hair_silhouette_test tests.hair_flow_test tests.hair_baseline_test tests.hair_eyewear_test
```

Checks cover asymmetric silhouette recovery on a synthetic capture, exact
protection of fixed vertices, safe unobserved-region fallback, image-flow lifting,
surface-conformed curve binding/deformation, protection of
brow pixels, saved grooms, export, and existing face/eye behavior. Inspect front,
profiles, rear and overhead after rebuilding. Hidden roots, covered eyebrow
hairs and unobserved crown concavities cannot be verified from this video.

Ear fitting, material ownership, glasses cleanup and camera-cache safeguards are
documented in [HEAD_RECONSTRUCTION_PLAN.md](HEAD_RECONSTRUCTION_PLAN.md).

## Rear hair texture support

The texture bake separates camera blending from photographic support. Its
`facing**8` blend preference selects clear overlapping photographs; using that
same score to trigger completion was replacing usable oblique rear views with
repeated frontal-hair material. Annotated hair now uses `facing**2` support while
retaining the depth, silhouette, alpha and accessory masks. The newly retained
color comes from those same hair observations, not a competing neck-skin view.
Feathered polygon edges and a per-view robust hair-color range prevent coarse
annotations from extending skin into the nape. Light hair uses its own sampled
color range. Hair annotations participate even when no glasses are present.

This changes appearance, not the reconstructed surface or rig. Unseen hair
still uses labeled material continuation. Regression checks are in
`tests/hair_appearance_test.py`; inspect the rebuilt result from both rear angles
with `hair-review.html` as well as running the tests.

## Version 5 capture validation

On capture `7a2bc070892642999d3357c2c5838390`, a full cached pipeline rebuild took
48.16 seconds. The accepted hair-only rebuild on the latest head took 33.17 seconds.
In the seven annotated review views, mismatched silhouette pixels within the hair
region fell from 19,046 to 15,366 (19.3%). Every view improved.
These views are reconstruction inputs, so this is a fit check, not an independent
likeness benchmark. Median local contour residual fell from 3.503 to 1.856 px.

The accepted face and ear vertices, mesh topology and physics cage are unchanged.
The latest nape transition is preserved by restricting refits to the hair core.
There are no reversed triangles or new nonadjacent crossings. All 21,600 rendered
fibers reproduce exactly after a JSON session round trip and follow a rigid
surface translation and return with less than 0.00004 mm numerical error.

Local evidence is in `.local/hair-rework-validation.json` and
`.local/hair-rework-runtime-validation.json`. The accepted generation and previous
generations remain in the capture's transactional artifact store. Compare
`hair-review.html` with `hair-review.html?review=hair-rework-before` while the dev
server runs. Unseen crown shape, the inner structure of hair clumps and the
photographic transition behind the ears remain approximate.
