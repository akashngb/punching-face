# Performance budget

Guardrail from OMNI.md §6.3: OMNI adds features on top of the existing pipeline. Numbers must not
regress between "OMNI off" and "OMNI on". Snapshots recorded here.

## What we measure and how

| Metric | Source | How captured |
|---|---|---|
| Reconstruction time (photo → mesh) | `server.py`, `/api/face-captures` progress log | Time between `queue` → `complete` for the most recent capture; from `sponsor_obs` / `pipeline_timing`. |
| Rigging time (mesh → rig) | `face_pipeline.py::FaceStore.train()` subprocess | Wall time from `mesh.ready` to `rig.ready` in the progress feed. |
| Render FPS | `window.__punchingFace.state` + on-screen FPS meter (`#fps`) | `scripts/perf_snapshot.mjs` samples 10 s idle + 10 s under punches. |
| Contact → deformation latency (ms) | `#hit-latency` (webcam tracked) + `#cv-latency` | Same script; the tag `hit-latency` covers `MediaPipe frame timestamp → impulse applied`. |
| First model response (ms) | `omni/latency` overlay (`Ctrl-L`) | Toggles the overlay; median across 5 turns. |
| Contact → cached audio (ms) | `omni/latency` overlay | 50 ms budget (OMNI.md §3.2). |

Run `scripts/perf_snapshot.mjs` (Node 18+, DevTools protocol) against the running dev app to log a
JSON row. Pass `--label baseline` or `--label omni-on`. Row is appended below.

## Baseline (OMNI off)

Recorded before the OMNI integration lands. Any row here is authoritative for "existing pipeline".

<!-- APPEND ROWS BELOW; keep the header -->

| Date | Machine | Recon (s) | Rig (s) | Idle FPS | Under-punch FPS | Contact→deform (ms) |
|---|---|---|---|---|---|---|
| _pending_ | Apple M5 Pro, macOS 26.5 | | | | | |

## With OMNI on (Plan A/B/C)

Recorded after each engine milestone. Regression = block on it before shipping.

| Date | Plan | Recon (s) | Rig (s) | Idle FPS | Under-punch FPS | Contact→deform (ms) | First response (ms) | Cache audio (ms) |
|---|---|---|---|---|---|---|---|---|
| _pending_ | | | | | | | | |

## Notes

- Idle FPS should track the baseline within ±2 fps. If it drops, OMNI is on the render thread
  somewhere — every OMNI worker should be a `Worker` or `AudioWorkletNode`, never `setInterval` on
  the main thread doing heavy work.
- Contact → deformation must be **identical** to baseline: OMNI runs beside the physics loop.
  The classifier reads from existing collision events; it never adds a physics pass.
- First response and cache audio are additive numbers, not regressions. They're what judges score.
