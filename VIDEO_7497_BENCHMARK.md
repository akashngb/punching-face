# IMG_7497 video-to-face benchmark — 2026-09-19

The separate local demo is **http://127.0.0.1:5183/**. Its captures are in
`.local/video-captures/`, separate from the original demo on 5173.

## Measured result

The 15.4-second supplied video produced a usable, textured, editable head in
**108.639 seconds (1 minute 48.6 seconds)** with the fast local preset.

| Processing stage | Fast local, 2048px | AI detail, 3072px |
| --- | ---: | ---: |
| Import, decode, detect and save frames | 36.897 s | 41.046 s |
| Reconstruction, including worker startup | 70.492 s | 185.145 s |
| Load texture, mesh and Newton | 1.250 s | 4.069 s |
| **Video to interactive model** | **108.639 s** | **230.260 s** |

These were fresh capture folders: camera recovery, geometry and textures were
built for each run. The AI run used live requests, not replayed responses. The
reported total sums the application's measured import, reconstruction and load
intervals; manual waiting and the browser's status-poll interval are excluded.
The benchmark button starts reconstruction automatically after import. Installed
models and libraries were already present, and the machine was also running
other apps. This is one successful timing per preset, not a latency guarantee.

The fast preset keeps the fitted head, photographic hair envelope, recorded eye
detail and Newton physics. It omits AI ear refinements, generated missing detail,
separate hairstyle strands and estimated accessories. The full AI path remains
available but **does not meet the two-minute target** in this measurement.

## Changes

- Seed head tracking from a frontal frame elsewhere in the clip before scanning
  it chronologically. This video starts in profile; the old importer saved only
  13 frames and could not start reconstruction.
- Decode the source locally in one sequential pass, with a browser fallback.
  Keep sample timestamps and full-resolution source video for detail baking.
- Overlap one ordered frame save with the next detection, and use faster lossless
  PNG compression. Pause scene simulation/rendering while the capture dialog is
  open or the page is hidden.
- Skip back-facing texture samples, batch camera projections on four workers,
  and reduce contributions in camera order. Preserve the original visibility,
  ear, accessory and fine-detail ownership rules.
- Fit the template during the AI wait; overlap texture baking with independent
  hair/physics work. Publish the mesh once after its appearance is ready.
- Batch shared-vertex checks in the ear collision detector instead of calling
  `np.isin` for every candidate pair. Keep the intersection checks themselves.
- Add configurable loopback ports, independent capture storage and a 2K local
  preset. The original application's defaults remain 5173 and 3072px with AI.

## Verification

The fast result registered 44 views and contains 38,570 triangles. Its geometry
is finite and watertight and passed the pipeline's withheld-view surface gates.
The captured orbit still has gaps; those surfaces remain estimates.

In the browser, checked the textured face, wireframe, a held left-hook impact,
and Reset head. The scene used Newton with 4,518 tetrahedra. The full AI model is
also retained in Saved scans.

Checks passed: web tests (81 passed, one skipped), production build, native video
sampling (2), capture storage (14), head/ear semantics (25), and accelerators
(12 passed, one optional real-video check skipped). Twenty randomized comparisons
matched the old and batched triangle-crossing implementations exactly.

Local evidence:

- `.local/video-optimization/results.json`
- Fast capture: `.local/video-captures/dc12350198eb43198155055b9870acb5/`
- AI capture: `.local/video-captures/653b82df28eb4513b7e875f3e78cdb56/`
- Tested app snapshot: `artifacts/video-lab/`

## Run again

```sh
npm run dev:video -- .local/video-optimization/IMG_7497.mov
```

This starts web/API/physics on 5183/5184/5185 and adds a button that times a new
capture from the supplied file. Stop the current isolated instance before
restarting it. Without a video argument, normal file upload is available.
`CONTACT_TEXTURE_SIZE=3072 VITE_CONTACT_FAST_CAPTURE=0 npm run dev:video` selects
the full-detail defaults. Each launch snapshots the current application code;
changes being made in another task require restarting this isolated lab.
