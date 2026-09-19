# Notes for coding agents working in this repo

Written 2026-09-19 by Claude at the owner's request, to hand over research results. Edit or delete freely.
Everything below was measured or read directly on this machine unless it says otherwise. The full reasoning,
evidence tags and recipes are in [OPEN_SOURCE_STACK.md](OPEN_SOURCE_STACK.md). Read that before choosing a library.

## Where things are

- `OPEN_SOURCE_STACK.md`: which open-source projects fit each failing stage, what was verified, what is still a guess.
- `.local/third_party/`: 17 pinned, permissively licensed sources (reference code and ungated model data).
  Recreate with `.venv/bin/python scripts/setup_third_party.py`; pins are in `scripts/third_party_manifest.json`.
- Convention already used here: third-party **sources go in `.local/`**, prepared **browser assets go in `public/`**
  via a `scripts/prepare_*.py` step. `.local/` is excluded from the Vite watcher, the dependency scan, `fs` serving
  and git, so writing there never reloads a camera session.

## Hard constraints of this machine

- Apple M5 Pro, 64 GB, macOS 26.5. **No CUDA.** Reject anything needing `nvdiffrast`, `pytorch3d` CUDA ops,
  `tiny-cuda-nn`, `diff-gaussian-rasterization` or `gsplat`.
- **No full Xcode** (Command Line Tools only): `xcrun -f metal` fails and `xcodebuild` will not run. Anything that
  compiles Metal at *build* time cannot be installed. Run-time Metal (torch-MPS, MLX wheels) is fine.
- `.venv` is Python 3.9.6, and that is a ceiling: `open3d==0.18.0` and `pycolmap==3.13.0` are the last releases with
  3.9 wheels. Put new ML/geometry tools in a separate Python 3.12/3.13 env under `.local/` and call them by
  subprocess, as `.local/newton-env` already does. Do not upgrade `.venv` in place.
- Never import `pycolmap` and `open3d` in one process on macOS (conflicting OpenMP runtimes; native crash).

## Verified traps

1. **Self-calibrated intrinsics are wrong and the gates do not notice.** On the synthetic fixture the true focal is
   1250 px (`public/generated/face-fixture/source.json`); the pipeline recovered 1701 px (+36 %) with radial
   `k = -2.48` on a distortion-free render, while reporting 0.41 px reprojection error and 40/40 views. Supply a known
   camera instead of estimating one. No test checks recovered focal against the stored ground truth yet.
2. **A supplied FOV is silently ignored on existing captures.** In `scripts/photo_cameras.py::recover()` the
   `dataset/sparse/0` legacy-reuse branch returns before the `calibrated` path is reached.
3. **Spark 2.2.0: `covObjectModifiers` is not a constructor option.** The installed build only reads the property and
   never assigns it from options. Set `mesh.covObjectModifiers = [...]` after construction, then
   `mesh.updateGenerator()`. Requires `SparkRenderer({covSplats:true, accumExtSplats:true})` and
   `SplatMesh({extSplats:true, covSplats:true})`.
4. **Do not mesh from Gaussian centres.** Median splat thinness here is 0.2–0.4 (blobs, not surfels). Fuse depth
   rendered at the registered cameras into a TSDF; Open3D 0.18 already has `ScalableTSDFVolume`.

## Licence rule for `.local/third_party`

Everything fetched there is MIT, Apache-2.0 or CC0, so its code may be adapted with attribution kept. Projects under
non-commercial, GPL/AGPL or custom licences (2DGS, RaDe-GS, GaussianAvatars, PhysGaussian, Spirula, OpenMVS, Sapiens2,
FLAME, MANO, SMPL-X) are cited by URL in `OPEN_SOURCE_STACK.md` on purpose: read them for the algorithm, do not paste
from them. Two caveats travel with fetched code: `anny` must stay on `topology="anny"` (its `smplx` option pulls
non-commercial assets), and `face-parsing`'s released weights inherit CelebAMask-HQ's non-commercial terms.

## Not done, on purpose

No model weights, prebuilt binaries or gated assets were downloaded; nothing was installed; no licence was accepted and
no account was used on the owner's behalf. Those choices are listed in section 6 of `OPEN_SOURCE_STACK.md` for the owner.
