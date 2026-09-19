# Optional learned-depth feasibility check

This experiment did **not** change the reconstruction pipeline or any published head. It tests whether an available multiview depth model can provide useful geometry on this Mac before adopting it. Five synthetic views of one head are not an identity benchmark or evidence of performance on arbitrary videos.

## Model and inputs

- [Meta MapAnything Apache](https://huggingface.co/facebook/map-anything-apache), pinned to `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`; [official implementation](https://github.com/facebookresearch/map-anything). The public Apache model supports images and optional geometric inputs. Supplied cameras condition its predictions; they do not force the output rays and poses to equal those cameras.
- Separate Python 3.13 environment `.local/head-depth-env`, PyTorch 2.8.0, Apple MPS, float32 inference. The application environment and dependencies are unchanged. Public weights and pinned DINOv2 source are cached under `.local/head-depth-cache`; no user photographs were uploaded.
- Existing Lee Perry-Smith CC BY 3.0 synthetic fixture, normalized to 0.28 m. Five views at -40, -20, 0, 20 and 40 degrees, known 1250-pixel focal length at 768 pixels, known camera poses. Official preprocessing resizes to 518 pixels and updates the intrinsics.
- Second run supplied 270–315 exact synthetic depth pixels per view as sparse anchors. Those are **oracle inputs**, not measurements available from an ordinary video.

## Results

Errors below are per-view ranges of median absolute local depth error against the known mesh. Background and invalid predictions are excluded. Oracle calibration uses only anchor pixels; evaluation excludes anchors plus a four-pixel margin.

| Method | Median error across the five views |
| --- | ---: |
| Images plus known cameras, raw output | 1,282–1,393 mm |
| Images, cameras and oracle sparse depth, raw output | 20.8–34.0 mm |
| Unanchored output, then one global oracle depth-scale correction | 8.7–19.0 mm |
| Anchor-conditioned output, then one global oracle scale correction | 12.3–21.0 mm |

For the calibrated unanchored result, per-view 95th-percentile errors remain 21.2–38.3 mm. Fitting scale from only views 0, 2 and 4 leaves 11.9 and 19.7 mm median errors in views 1 and 3. This calibration uses synthetic truth and is not unassisted reconstruction accuracy.

Astra independently checked the saved outputs, projection conventions, preprocessing and inference code. No demonstrated API or coordinate-gauge error explains the raw failure. Camera-only similarity alignment does not repair it: the unanchored camera trajectory suggests scale 1.418, while oracle depth suggests 0.312. Independent per-view scale corrections therefore do not establish a consistent 3D surface. A CPU/backend comparison has not been performed.

The anchor-conditioned run retains valid depth on roughly 82–86% of the evaluated foreground. Missing pixels are reported as missing, not successful predictions. Cached five-view inference took about 15–19 seconds; downloading and loading weights, and the concurrent full reconstruction jobs, are separate costs.

## Decision and evidence

Keep this model outside the production head pipeline. These outputs may contain coarse shape information, but the remaining centimetre-scale errors cannot safely replace measured facial geometry or establish accurate ear, eyelid or strand structure. Any later integration needs independently measured sparse anchors, fixed-camera consistency checks, uncertainty masks and validation on additional identities and capture conditions.

Private reproducible experiment files are `.local/benchmark_mapanything_fixture.py`, `.local/evaluate_mapanything_fixture.py`, and `.local/analyze_mapanything_calibration.py`. Their saved inputs/predictions and reports are under `.local/mapanything-fixture`, `.local/mapanything-fixture-anchored`, and `.local/mapanything-calibration-analysis.json`. The pinned environment inventory is `.local/head-depth-installed.txt`. These large local research assets are intentionally not application dependencies or Git assets.
