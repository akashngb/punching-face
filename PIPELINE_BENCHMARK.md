# IMG_7496.MOV: measured video-to-model run

Measured locally on 2026-09-19 on the Apple M5 Pro Mac, with the browser importer, local camera/mesh/texture processing, OpenAI hair and eyewear analysis, and Newton model loading.

This baseline predates the dedicated eye-detail stage described below.

**Successful processing time: 180.6 seconds (3m 0.6s).**

| Step | Measured time |
| --- | ---: |
| Decode, select, mask and save video frames | 15.6 s |
| Recover cameras | 12.7 s |
| AI hair and glasses analysis | 66.6 s |
| Bake photographic texture | 82.1 s |
| Worker startup, face/head/hair fitting, accessory geometry, rig and output | 3.0 s |
| Load mesh, texture and Newton physics into the viewer | 0.6 s |
| **Total processing** | **180.6 s** |

The recording itself is **24.3 seconds**. Recording duration, time spent inspecting code or waiting to press Create, and the later copy of the original recording into the replay store are excluded. New imports include that local video copy in their import measurement. The model-file build alone took **164.4 seconds**. Browser polling can add up to about two seconds before automatic loading begins; that idle interval is excluded from the processing total.

This is one measured run, not a fixed service-level promise. Both camera reconstruction and AI analysis were fresh. Existing model, camera and AI results were not reused; successful camera matching was slower than the first attempt. Rebuilds with cached inputs are a different workload.

## Output

- 51 accepted frames at 540 × 960 pixels; 40 recovered cameras.
- 233.3° recovered coverage and seven rear cameras. This is not a complete 360° reconstruction.
- 19,792 mesh vertices and 38,636 triangles, plus independent eyewear and hair detail.
- Both glasses arms used recorded profile contours; lateral depth and hidden ear fit remain estimates.
- The original video is retained privately in the scan and decodes as a 24.3-second recording after a page reload. Byte-range seeking returned the requested bytes.

Local evidence: `.local/face-captures/7a2bc070892642999d3357c2c5838390/` contains `source.json`, `timing.json`, `status.json`, `mesh.json`, the recording (`source-video`), camera results, and AI annotations. Previous scans remain intact.

## Failed first attempt

The first build stopped after **258.0 seconds (4m 18.0s)** because the AI response reached its output limit; no completed model was reported. Its AI stage consumed 249.5 seconds. The retry used bounded polygon/description lengths, three-decimal coordinates and a larger response ceiling. AI analysis then completed in 66.6 seconds. The failed attempt is retained in the timing panel and under `attempt-1/`.

Including that failed attempt, active processing consumed **7m 18.6s**. Development and inspection time between attempts are excluded.

## Validation

Inspected the rendered glasses from the front and both profiles. Checked clear-lens appearance, bevelled/tapered arms and hinge details. Automated checks cover eyewear save/restore and GLB geometry/material round trips, profile fitting, source-video persistence and range requests, interrupted uploads, timing persistence, and totals that exclude recording/idle time. The JavaScript suite, focused Python tests and Vite production build passed.

## Eye-detail follow-up

Native-resolution eye scanning and a fresh Astra eye assessment/material specification took **10.845 seconds** for this recording. Both eyes failed the detail gate: the iris texture and pupil were not separately resolved through blur, eyelid coverage and glasses. Astra supplied matching dark-brown iris parameters, rendered locally as explicitly estimated eye materials. The first full rebuild with these cached eye parameters took **125.23 seconds**; it reused recovered cameras and head/accessory analysis and is not a fresh end-to-end timing. The final eye-only maintenance rebuild took **142.583 seconds** including output writes, retained the existing hair analysis, and incorporated lid-depth and shading corrections. A canceled wait for the separate hair-analysis update is recorded as an earlier interrupted attempt rather than worker startup. These maintenance runs do not establish a new fresh-capture end-to-end benchmark.

The original baseline outputs and timing are preserved under `before-eye-detail/` in the scan directory. Eye crops, the quality decision and generated parameter provenance are under `eye-detail/` and `eye-detail.json`. The full-pipeline eye stage is included in the timing panel for future builds.

Eye validation: native-detail rejection and the measured-iris branch, safe local-only/API-failure behavior, lid-depth clearance, and skin isolation passed. The browser retained eye color and gloss through GLB export/reimport and editable-session restore, with Newton reconnected. The combined head/strand GLB was 150.0 MB; local save and import now share a bounded 180 MB ceiling. Focused Python tests (29), JavaScript tests (54), and the production build passed.
