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
3. **Fit hair shape.** Registered silhouettes constrain the outer envelope.
   The measured crown-height bound is applied after radial expansion, preventing
   an unseen cone from growing above the recorded height. A broad crown taper is
   still an estimate; this is not direct strand-level 3D reconstruction.
4. **Recover visible detail.** Hair masks, visibility and pixel orientation guide
   short visible lock sections. Version 4 adds no unsupported crown roots and no
   generic sinusoidal waves. Each station samples photographed colour. Individual
   fibers remain modeled detail, while hidden crown texture uses a labeled
   photographic continuation.
5. **Bind and bake.** Every hair-curve station has barycentric weights on its
   local rig triangle. Eyebrow landmarks protect brow pixels during opaque rim
   cleanup. Clean views take precedence over inpainted areas. Broad colour is
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
`before-hair-v3/` and this capture's `before-hair-v4/`. The rebuild merges hair
updates into the latest mesh and refuses to overwrite concurrent surface edits.

`--recognize` refreshes the cached vision analysis only if needed and sends the
selected cropped images to the configured API. The changes above reuse the
existing analysis and installed local NumPy, SciPy, OpenCV and Three.js stack.
No additional ML weights, CUDA dependency or generated replacement hairstyle is
required. Version 2 and 3 sessions remain readable.

## Verification

```sh
npm test
npm run build
.venv/bin/python -m unittest tests.hair_flow_test tests.hair_eyewear_test tests.eye_detail_test
```

Checks cover image-flow lifting, local curve binding/deformation, protection of
brow pixels, saved grooms, export, and existing face/eye behavior. Inspect front,
profiles, rear and overhead after rebuilding. Hidden roots, covered eyebrow
hairs and unobserved crown concavities cannot be verified from this video.
