# Open-source stack for CONTACT: what to use on this Mac, and why results are missing

Researched and fetched 2026-09-19. Companion to [README.md](README.md) and [RESEARCH.md](RESEARCH.md).
Sources live in `.local/third_party/` (hidden folder; in Finder press Cmd+Shift+. to show it).
Re-create them with `.venv/bin/python scripts/setup_third_party.py` (pins in [scripts/third_party_manifest.json](scripts/third_party_manifest.json)).

**Evidence tags used throughout.** Every claim carries one:

- **[measured]** I ran it on this machine, against this repo's own data.
- **[verified]** I read the actual source, licence text, package metadata or API response myself.
- **[reported]** Found by a research agent from live GitHub / PyPI / Hugging Face data on 2026-09-19. Not re-run by me.
- **[unverified]** Nobody confirmed it. Treat as a hypothesis.

Nothing in this document was installed, built or executed except where tagged **[measured]**. No model weights,
binaries or gated assets were downloaded.

---

## 0. Read this first

### Why the app shows no real result

1. **The surface step rejects every splat.** COLMAP and Brush succeed, then Poisson-on-Gaussian-centres lands 31–33 mm
   from the landmarks and the gate refuses it (`status.json` of jobs `38f42aa3…`, `cc7e8283…`). **[measured]**
2. **The fallback is a 468-point mask.** MediaPipe landmarks + Loop subdivision + a gray invented cranium. Everything
   between landmarks is smoothing, not the subject. No ears, hair, neck or back of head.
3. **The splat renderer is gone.** `src/main.js` no longer imports `@sparkjsdev/spark`; jobs report `"usesSplats": false`. **[verified]**
4. **The camera model is wrong and nothing notices** (section 2.1). On the repo's own synthetic fixture the recovered
   focal length is off by **+36 %** while reprojection error reads an excellent 0.41 px. **[measured]**
5. **The arm never trains.** SIFT on masked, low-texture skin registered 24/59 views over 11°. **[measured, from RESEARCH.md + job logs]**
6. **Hardware rules out the famous repos.** No CUDA, and no full Xcode, so no Metal *compiler* either (section 1). **[measured]**

### The five moves, ordered by payoff ÷ effort

| # | Move | New dependencies | Section |
|---|---|---|---|
| 1 | Give the pipeline a **known camera** instead of self-calibrating | none (OpenCV optional) | 4.1 |
| 2 | Mesh from **rendered / predicted depth via TSDF**, not from Gaussian centres | none (Open3D 0.18 already has it) | 4.2, 4.4 |
| 3 | Replace the landmark mask with a **fitted full-head template** + non-rigid wrap | none for the wrap (trimesh has it) | 4.3 |
| 4 | Re-register the arm with **learned features + part masks** | `brew install colmap` *or* hloc | 4.5 |
| 5 | Run **physics in the browser** and bring splats back with covariance-aware deformation | none (Spark 2.2.0 already installed) | 4.7, 4.8 |

Do move 1 first. Moves 2–4 all consume camera poses, and a dense photo-consistency stage
(`scripts/photo_dense.py`, in progress) is the most camera-sensitive stage of all.

---

## 1. This machine

| Fact | Value | Tag |
|---|---|---|
| Chip / RAM / OS | Apple M5 Pro, 64 GB unified, macOS 26.5 | [measured] |
| GPU compute | No NVIDIA, **no CUDA**. PyTorch-MPS, MLX, CoreML, ONNX Runtime, WebGPU are the options | [measured] |
| Xcode | **Not installed.** Command Line Tools only. `xcrun -f metal` fails; `xcodebuild` refuses to run | [measured] |
| Consequence | Anything that compiles Metal shaders **at build time** is blocked (OpenSplat-Metal, `trellis-mac`, Pixal3D's Mac port). Metal compiled **at run time** is fine (torch-MPS, MLX wheels, `metal-gauss`) | [measured] + [reported] |
| Swift | 6.3.3 present, so Apple's `PhotogrammetrySession` API is reachable from a SwiftPM CLI | [measured] + [reported] |
| Build tools | Homebrew yes. `cmake`, `ninja`, `cargo`, `uv`, `colmap` absent | [measured] |
| Pythons | `.venv` = 3.9.6 (pip 21.2.4, no `--dry-run`). `.local/newton-env` = 3.13.14. Homebrew `python3.13` | [measured] |

### Python 3.9 is a ceiling, not a choice

The pins `open3d==0.18.0` and `pycolmap==3.13.0` are exactly the **last versions that ship Python 3.9 wheels**. **[measured]**

| package | newest for py3.9 | newest for py3.13 |
|---|---|---|
| open3d | 0.18.0 | **0.20.0** |
| pycolmap | 3.13.0 | **4.2.0** |
| torch | 2.8.0 | **2.14.0** |
| mujoco | 3.3.7 | 3.13.0 |
| genesis-world | 0.2.1 | **1.4.1** |
| pytetwild | none | 0.4.2 |
| pymeshlab | none | 2025.7.post1 |
| ipctk | 1.4.0 | 1.6.0 |
| mlx | 0.29.3 | 0.32.2 |
| eos-py | 1.5.0 | none (sdist only) |
| opencv-python-headless | 5.0.0.93 | (not checked) |

Method: `pip install --dry-run --no-deps --only-binary=:all:`. This proves a matching wheel **exists**; it does not
prove dependencies resolve or that the package runs. The 3.9 column was cross-resolved from pip 26 against platform
tags `macosx_11…15_0_arm64` + `universal2`, so "none" means none among those tags.

**Recommendation:** put every new ML/geometry tool in a **separate Python 3.12 or 3.13 env** under `.local/`, the way
`.local/newton-env` already works, and talk to it by subprocess. Do not upgrade `.venv` in place.

> Keep `pycolmap` and `open3d` in **separate processes**. They ship different OpenMP runtimes on macOS and crash when
> imported together. The repo already knows this (`scripts/train_arm.py`, `scripts/photo_cameras.py::export_cameras`). **[verified]**

---

## 2. Root causes found in this repo's own evidence

### 2.1 Self-calibrated intrinsics are wrong, and the quality gates cannot see it

`scripts/render_face_fixture.py` renders the synthetic test head with an ideal pinhole: `size=768`, `focal=1250`, zero
distortion, camera 0.65 m away, yaw −50°…+50°. It even stores the truth in `public/generated/face-fixture/source.json`.
Job `38f42aa3…` ran the pipeline on those renders with no FOV supplied, so COLMAP self-calibrated. **[measured]**

| | focal | horizontal FOV | radial `k` | camera sweep |
|---|---|---|---|---|
| Ground truth | 1250 px | 34.2° | 0 | 100° |
| Pipeline recovered | **1701 px** | 25.4° | **−2.48** | 108.97° |

That is **+36.1 % focal error**, with absurd distortion on a distortion-free render, while the same job reported
reprojection error **0.409 px** and **40/40** views registered. Reprojection error, registration count and camera span
all look healthy for a camera that is badly wrong. No test compares the recovered focal to `source.json["focal"]`.

Likely mechanism (my interpretation): a head turning in front of a fixed camera is an orbit at constant radius around
one small object. Focal length and subject distance then trade off almost freely, and `SIMPLE_RADIAL`'s `k` soaks up
the remainder. The real capture has the same geometry.

Real capture `cc7e8283…`: recovered `f = 580.4 px`, `k = −0.137`, i.e. **95.6° horizontal FOV**, and
`horizontalFovDegrees` was never supplied. **[measured]** The true focal is unknown, so I cannot state the size or sign
of the error, only that it came from a procedure shown unreliable above. 95.6° is unusually wide for a laptop webcam.
**[unverified]** hypothesis worth a 10-second check: if Center Stage / auto-framing is on, the crop window moves between
frames, which breaks the pipeline's `camera_mode=SINGLE` assumption outright.

**What this does and does not explain.** It does *not* by itself explain the 31–33 mm rejection: landmarks and splats
share the same cameras, and the README already notes the chin failed the gate, which points to boundary coverage
(section 2.2). It *does* distort metric shape (likeness), and it undermines anything that fuses depth across views or
scores photo-consistency along camera rays.

**A trap in the current code.** `scripts/photo_cameras.py::recover()` has a correct "known FOV" path
(`SIMPLE_PINHOLE`, refinement disabled). But the `legacy=folder/'dataset/sparse/0'` branch runs **first** and returns
the old self-calibrated cameras whenever they exist. On every existing capture, a supplied FOV is **silently ignored**. **[verified]**

### 2.2 Poisson on Gaussian centres

`reconstruction.py::reconstruct()` treats splat centres as surface samples and the smallest covariance axis as the
normal. Median thinness is 0.38 (real) and 0.20 (fixture): these are volumetric blobs, not surfels. **[measured]** The
literature's answer is to mesh from **depth rendered at the training cameras**, fused into a TSDF (2DGS, RaDe-GS, PGSR
all do this). **[reported]** The repo already owns the hard part: `scripts/bake_texture.py::render()` is a CPU Gaussian
rasterizer that accumulates alpha-weighted depth.

### 2.3 SIFT on masked skin

Arm job `879025fd…`: 24/59 registered, 427 sparse points, 11.31° spread. Bare forearm skin has almost no SIFT-stable
texture, and the mask removed the textured background and clothing that could have carried registration.

### 2.4 Physics runs in the slowest possible place

`newton_face.py` builds with `device='cpu'`. Warp on macOS is CPU-only, and its FAQ says CPU kernel launches run
serially. **[reported]** On top of that sits an HTTP JSON round trip at 30 Hz. Indentation is a scripted sinusoid capped
at 3.5 mm, not a collision response. **[verified in `newton_face.py::step`]**

### 2.5 The splat path, and a Spark trap waiting for whoever restores it

Spark 2.2.0 (already in `node_modules`) has an experimental **covariance-splat** path that lets splats rotate, stretch
and shear with a deforming mesh, which is exactly the README's stated limitation. All the APIs exist in the installed
build. **[verified]** But in `node_modules/@sparkjsdev/spark/dist/spark.module.js`, `covObjectModifiers` /
`covWorldModifiers` are only ever **read** (lines 12537–12549) and never assigned from constructor options, unlike
`objectModifiers` (lines 12239–12245). **Passing `covObjectModifiers` to the constructor is a silent no-op.** **[verified]**
One research agent wrote it as a constructor option; that form does not work. See 4.8.

---

## 3. What is in `.local/third_party/`

724 MB, 17 pinned entries. Permissive licences only, each checked by reading the LICENSE text or file headers. **[verified]**

| Path | Licence | Fixes | Why it is here |
|---|---|---|---|
| `ICT-FaceKit/` (sparse: `FaceXModel`, `Scripts`) | MIT | face | Ungated full-head template: 26,719 verts with ears, neck, scalp, teeth, eyeballs; 100 identity modes; 53 ARKit-named expressions matching MediaPipe's blendshape names |
| `GNM/` (sparse: `gnm`) | Apache-2.0 | face | Google GNM Head (2026). 17,821 verts, ~5,000 training subjects. Stronger identity prior. Needs Python ≥ 3.10 |
| `gnm-webcam-puppet/` | Apache-2.0 | face | `webcam_puppet/assets/correspondence.npz`: 473 MediaPipe-landmark → GNM-vertex pairs. Load with `allow_pickle=False` |
| `spark/` (sparse: `src`, `docs/docs`, `examples/interactive-deform`) | MIT | splats | Source + docs + deform example for the renderer you already ship as `dist`-only |
| `map-anything/` | Apache-2.0 code | arm, face | Multi-view poses + dense depth, MPS support merged, **accepts known intrinsics** |
| `Depth-Anything-3/` | Apache-2.0 code | face | Pose-conditioned multi-view depth in the COLMAP frame and scale |
| `MoGe/` **(pinned pre-v3)** | MIT (+ Apache-2.0 section) | face, arm | Metric depth + normals. `main` is v3 and says macOS is unsupported, hence the pin `07444410` |
| `Hierarchical-Localization/`, `LightGlue/` | Apache-2.0 | arm | ALIKED + LightGlue into a COLMAP database, compatible with your current pycolmap 3.13. Submodules deliberately not fetched |
| `anny/` | Apache-2.0 code, CC0 data | arm, face | Rigged MakeHuman-family body, 104 bones incl. fingers, plus CC0 facial action units. No MANO / SMPL-X needed |
| `MHR/` | Apache-2.0 | arm | Meta's Momentum Human Rig: alternative ungated body + hand rig |
| `face-parsing/` | MIT code | masks | BiSeNet 19-class parsing incl. **eyeglasses**, hair, ears, lips, neck |
| `JoltPhysics.js/` | MIT | physics | Maintained WASM engine fallback: XPBD tets, kinematic-collider contact, skinned constraints with back-stop |
| `single-files/ten-minute-physics/` | MIT (per-file header) | physics | `10-softBodies.html`, `12-softBodySkinning.html`, `BlenderTetPlugin.py`. Repo is 473 MB with no LICENSE file; only header-bearing files taken |
| `single-files/dn-splatter/export_mesh.py` | Apache-2.0 | splat→mesh | Permissive reference for meshing from rendered depth/normals |
| `single-files/nerfstudio/tsdf_utils.py` | Apache-2.0 | splat→mesh | Compact pure-torch TSDF integration |
| `single-files/QtMeshEditor/` | MIT | face | Worked example: parse ICT OBJs by hand, NRICP wrap, blendshape transfer |

Checked after fetching **[verified]**: all 16 licence files read and matching the table; ICT-FaceKit holds 158 OBJs
(100 identity modes + 57 expression/other + neutral) and `generic_neutral_mesh.obj` has 26,719 vertices; GNM's
`gnm/shape/data/versions/v3_0/gnm_head.npz` is present (53.3 MB); `correspondence.npz` loads with `allow_pickle=False`
and holds 473 landmark→vertex pairs with `rigid` flags. One Git-LFS file was deliberately not downloaded: a README
teaser image in Depth-Anything-3.

Already on disk, no fetch needed: Newton's XPBD tet constraint at
`.local/newton-env/lib/python3.13/site-packages/newton/_src/solvers/xpbd/kernels.py::solve_tetrahedra` (Apache-2.0). **[verified]**

**Licence caveats that travel with the code.**

- `face-parsing`: code is MIT, but the released ONNX **weights** were trained on CelebAMask-HQ (non-commercial research terms). **[reported]**
- `map-anything` weights: `facebook/map-anything-apache` is Apache-2.0; `facebook/map-anything` is CC-BY-NC. **[reported]**
- `Depth-Anything-3` weights: BASE / SMALL / METRIC-LARGE Apache-2.0; GIANT-1.1 and NESTED CC-BY-NC; LARGE-1.1 disputed (README vs Hugging Face, issue #259 unanswered). **[reported]**
- `anny`: stay on `topology="anny"`. The optional `smplx` topology downloads **non-commercial** assets, and the README warns the free install "may download non-commercial only assets when needed". **[verified]**
- `LightGlue`: use ALIKED or DISK extractors. SuperPoint weights carry a restrictive Magic Leap licence. **[reported]**

---

## 4. Recipes

### 4.1 Give the pipeline a known camera (do this first)

Cheapest route, no new code: **fill in the existing FOV field** with a real number and make sure it is honoured.

1. In `scripts/photo_cameras.py::recover()`, check `calibrated` **before** the `legacy` reuse branch, or skip legacy
   reuse when `horizontalFovDegrees` is set. Otherwise existing captures ignore the FOV (section 2.1).
2. Turn off Center Stage / auto-framing / any "portrait" zoom for capture, and keep 1280×720 fixed.
3. Get the number properly once, with OpenCV (`opencv-python-headless` 5.0.0.93 has a py3.9 arm64 wheel **[measured]**;
   `cv2` is not installed yet): show a checkerboard on a phone or tablet, grab ~20 frames at varied tilt across the
   whole field, run `cv2.findChessboardCorners` → `cv2.calibrateCamera`. Keep `fx, fy, cx, cy` **and** the distortion
   terms. The current known-FOV path assumes zero distortion and a centred principal point; `OPENCV` or `RADIAL`
   camera models in COLMAP can hold the measured values with refinement disabled.
4. Reuse the same `K` for the arm. It is the same webcam.
5. Add the missing regression test: recovered fixture focal within a few percent of `source.json["focal"]`. It will
   fail today (+36 %), which is the point.

Free cross-check already in your stack: MediaPipe FaceLandmarker's `outputFacialTransformationMatrixes` gives a
per-frame 4×4 canonical-face → camera matrix, a metric pose prior for the face. **[reported]**

### 4.2 Splat → mesh by TSDF fusion (keeps Brush, zero new dependencies)

Replace `reconstruction.py::reconstruct()` (called from `scripts/convert.py`, invoked by `scripts/train_face.py` at the
`surface` stage).

1. Make a **perspective** twin of `scripts/bake_texture.py::render()`: transform centres with COLMAP `R,t`; project with
   `K`; 2-D covariance `Σ₂ = J W Σ Wᵀ Jᵀ` with `J = [[f/z, 0, −f·x/z²], [0, f/z, −f·y/z²]]`. It already accumulates
   alpha-weighted (expected) depth. Median depth, where accumulated alpha crosses 0.5, is sharper on blobby splats.
2. For each registered view, zero depth where the capture mask is off or alpha < 0.5.
3. In a **separate process** from pycolmap: `o3d.pipelines.integration.ScalableTSDFVolume(voxel_length≈0.001,
   sdf_trunc≈0.004–0.005, color_type=RGB8)`, `integrate(rgbd, intrinsic, extrinsic)` with the world→camera matrix,
   then `extract_triangle_mesh()`. Build RGBD with `depth_scale=1.0`, `convert_rgb_to_intensity=False`.
   **[verified: `ScalableTSDFVolume`, `TSDFVolumeColorType.RGB8` and `o3d.t.geometry.VoxelBlockGrid` all exist in the installed Open3D 0.18.0; parameter values are reported suggestions]**
4. Keep the largest connected cluster, then run the existing landmark gate.

Read, do not copy (non-commercial licences, cited only): 2DGS `utils/mesh_utils.py::GaussianExtractor.extract_mesh_bounded`;
RaDe-GS `mesh_extract.py` and the ray-plane depth in `render_forward.cu`; PGSR `render.py` (`sdf_trunc = 4 × voxel`).
Permissive references are in `single-files/dn-splatter/` and `single-files/nerfstudio/`.

Expect soft depth from 22k splats at 3,000 steps; more steps / splats should help **[unverified]**. TSDF meshes are
open surfaces: fine for the gate, not for tetrahedral meshing until closed.

### 4.3 A fitted full-head template instead of the landmark mask

Integration point: the surface-construction step in `scripts/build_photo_face.py::run()` / `scripts/photo_geometry.py`,
and `physics_binding()`. Codex has just started down this road with a **CC0 MakeHuman head**
(`public/head-template/`, `scripts/prepare_head_template.py`), which is a sound, licence-clean choice.

Three templates, all ungated, and they combine:

| Template | Gives you | Lacks |
|---|---|---|
| **MakeHuman head** (in progress) / `anny/` | Clean quads, edge loops, CC0. `anny` adds a differentiable shape model and **CC0 facial action units** on the same mesh family | No statistical *identity* prior learned from scans |
| **ICT-FaceKit** | 100 identity modes + 53 ARKit expressions whose names match MediaPipe's blendshape scores, so the rig is `morphTargetInfluences[name] = score` | Identity space reported weak (issues #18, #19); dormant since 2020 **[reported]** |
| **GNM** | Best identity prior (253 params, ~5k subjects, claims 0.75 mm scan fit vs 0.97 mm FLAME) **[reported]** | 2 months old, API moving, expression components unnamed, no jaw joint |

Recipe:

1. **Landmark fit.** Optimise similarity transform + identity weights (start 30–50, L2 prior) + a few nuisance
   expressions per frame; residual = Huber reprojection of embedded landmarks through the registered cameras. Drop
   face-oval points, which slide with yaw. scipy on CPU is enough.
2. **Non-rigid wrap, already installed:** `trimesh.registration.nricp_amberg(source_mesh, target_geometry,
   source_landmarks, target_positions, steps=[[ws, wl, wn, max_iter], …])`. Landmarks may be vertex indices or
   `(triangle_ids, barycentric)`; target may be a point cloud. `nricp_sumner` preserves shape better.
   **[verified: both functions, with these signatures, are in the project venv's trimesh 4.12.2]** Tuning advice is
   reported: normalise to ~unit scale, gate dense points to within ~8–10 mm of the landmark fit,
   **mask the glasses region**, and blend displacement to zero over ears / scalp / neck, which ±27° of yaw never saw.
3. **Rig.** `B_k = N_personal + (expr_k − generic_neutral)`; sum left/right pairs where MediaPipe has one channel;
   export GLB morph targets. This replaces the 4 heuristic morphs.

ICT-FaceKit gotchas **[reported]**: parse OBJs **by hand** (trimesh reorders and duplicates vertices on these
multi-material files; see `single-files/QtMeshEditor/`); no MediaPipe→ICT embedding ships (build from the 68 iBUG
indices in its README, then densify by render → detect → back-project); `eyeLook*` shapes do not move the eyeballs;
left/right mirroring against MediaPipe is unchecked.

If you are willing to register: **FLAME 2023 Open** is CC-BY-4.0 with commercial use allowed, but the download still
needs a sign-up, and regular FLAME 2020/2023 remain non-commercial. **[reported]** The official 105-point MediaPipe
embedding is public. I did not sign up on your behalf.

### 4.4 Dense geometry from learned multi-view depth

The pipeline has no dense depth source today; `scripts/photo_dense.py` is hand-building one from photo-consistency.
Before investing further there, note that these models take your **existing COLMAP solve as conditioning** and return
depth already in that frame, ready for the TSDF of 4.2:

- **Depth Anything 3**: `da3 colmap <scene> --device mps --export-format mini_npz`. MPS float64 fix merged (PR #26).
  Upstream lists `xformers` but guards the import, so install with `--no-deps`. **[reported]**
  `awesome-depth-anything-3` on PyPI is a **third-party fork**, not ByteDance; prefer upstream.
- **MapAnything**: `scripts/demo_inference_on_colmap_outputs.py --colmap_path … --apache --save_colmap`; MPS support
  merged (PR #131). Accepts known `K`. Maintainer caveat: training data has no static multi-view human captures. **[reported]**
- **MoGe-2** (pinned pre-v3): metric depth **and normals**, accepts known `fov_x`. Monocular, so fit a per-view scale
  against COLMAP sparse points before fusing. The pinned commit needs only Python ≥ 3.9. **[reported]**

Memory: torch-MPS attention is a plain math kernel. One agent's arithmetic puts VGGT-class global attention over
~59 frames above 64 GB, so plan ~24–30 frames per pass. With known poses, chunking is free. **[unverified: arithmetic, not measured]**

### 4.5 Arm registration and masks

**Matching.** The macOS `pycolmap` wheels are built with ONNX **off**, but Homebrew's `colmap` 4.2.0 depends on
`onnxruntime`, has an `arm64_tahoe` bottle, is BSD-3, and is not installed. **[verified via `brew info`]** COLMAP 4.x
adds `ALIKED_N16ROT` + `ALIKED_LIGHTGLUE`, LoMa, and merges GLOMAP as `global_mapper`. **[reported]** That ALIKED
matching works end-to-end on your arm photos is **[unverified]**.

```bash
brew install colmap
```

Then: `feature_extractor --FeatureExtraction.type ALIKED_N16ROT --ImageReader.mask_path <eroded masks>
--ImageReader.camera_params <locked K>`; `exhaustive_matcher --FeatureMatching.type ALIKED_LIGHTGLUE`; `mapper` with
focal refinement off, or `global_mapper`. If you would rather not install it, `Hierarchical-Localization/` +
`LightGlue/` do the same from Python against your current pycolmap 3.13 (device is hard-coded cuda-else-cpu; CPU is
fine for 59 images; no native masks, so filter keypoints in `features.h5`).

One benchmark warns that classical SfM **and** VGGT both struggle on handheld objects filmed by a static camera,
which is this exact setup (arXiv 2602.05822). **[reported]** Hence the mask trick below.

**Masks.** Two-mask trick **[reported]**: register with a *wide rigid* mask (torso + upper clothing + arm: clothing has
texture and rotates rigidly with the chair), then **train** with the arm-only mask. Part labels: Sapiens2 has
Hand / Lower_Arm / Upper_Arm and an **Eyeglass** class, runnable via `mlx-vlm` with ungated `mlx-community` weights.
Its custom licence prohibits "biometric processing", so it is cited, not fetched; whether that reaches reconstructing
your own face is your call. SAM 2.1 (Apache-2.0) runs on MPS in fp32 and gives temporally stable edges when prompted
with a part mask. **The `sam2` package on PyPI is a third-party fork, not Meta's**; install from the GitHub repo.

**Glasses.** Union of `face-parsing` class 6, a dilated eyeglass mask, and optionally `glasses-detector` (MIT).
Eyeglass-*removal* GANs hallucinate per frame and are not multi-view consistent: avoid. A glasses-off pass is cleanest.

### 4.6 A personal arm without MANO or SMPL-X

Every monocular hand-mesh method found (HaMeR, WiLoR, Hamba, …) requires MANO: registration, non-commercial. **[reported]**
Ungated route: fit `anny/` (or `MHR/`) bone lengths from MediaPipe world landmarks and girths from silhouettes or the
point cloud, cut the arm + hand submesh, bake texture from registered views, export glTF `SkinnedMesh`.
Ship something today: the WebXR generic-hand GLB (`@webxr-input-profiles/assets`, MIT, ~94 KB per hand, 25 bones),
with MediaPipe 21 → WebXR 25: landmark 0 → wrist; 1–4 → thumb chain; per finger MCP/PIP/DIP/TIP → proximal /
intermediate / distal / tip; the four finger metacarpals stay in bind pose. It is generic and ends at the wrist. **[reported]**

### 4.7 Physics in the browser

No released face-physics code runs on this Mac (the SCA 2026 wrinkling paper, the Generalized Physical Face Model,
Phace, SoftDECA: no code found). **[reported]**

Adopt: an **XPBD tetrahedral solver in a Web Worker**, seeded from `single-files/ten-minute-physics/`
(demo 10 = solver; demo 12 = `computeSkinningInfo` / `updateVisMesh`, the same cage → render-mesh binding this project
already uses). One agent benchmarked that solver class **on this M5 Pro** (Node 26.5, one thread, 10 substeps,
edge + volume constraints, one sphere collider) **[reported, measured by the agent, not by me]**:

| tets | 5,000 | 10,240 | 20,000 | 40,960 |
|---|---|---|---|---|
| ms / frame | 3.9 | 9.4 | 20.1 | 34 |

So 5k–10k tets at 60 Hz is within budget. Caveat: a cache-friendly grid mesh in Node; a real mesh in a browser Worker
may be 1.5–2× slower, and neither a neo-Hookean constraint nor skull queries were timed.

- Tets: `pytetwild` 0.4.2 (MPL-2.0, py ≥ 3.10) or `wildmeshing` 0.4.1 (has a py3.9 wheel) from a **closed** head mesh;
  repair with `manifold3d` (Apache-2.0). **[measured: wheels resolve]** TetGen is AGPL; MeshLib is non-commercial.
- Material: swap edge springs for the stable neo-Hookean tet constraint, ported from Newton's `solve_tetrahedra`
  (already on disk, Apache-2.0).
- Skull: a static inner surface queried with `three-mesh-bvh` (MIT); inner nodes constrained to the offset surface,
  free tangentially, plus a weak ligament spring. That yields **skin sliding over bone**.
- Fist: kinematic sphere/capsule swept per substep; project penetrating nodes out, Coulomb friction on the tangential part.
- Wrinkles are not resolvable at this tet count; they need a bonded thin-shell layer or a tension-map shader.

Keep the adapter seam Codex just added (`tests/newton-adapter.test.mjs`) and swap the backend behind it.
**Fallback:** `JoltPhysics.js/` (`npm i jolt-physics`). Its multithread build needs COOP/COEP headers, which can break
CDN-loaded MediaPipe/Spark assets. No self-collision; soft-body speed unmeasured.
**Offline validation:** `genesis-world` 1.4.1 (Apache-2.0; wheel resolves on py3.13; CI runs macOS CPU + Metal). Its
self-contact coupler needs float64, which Metal refuses, so that runs on the CPU backend. **[reported]**
**Newton-on-CPU:** drop it from the runtime loop; keep it, at most, as an occasional cross-check.

### 4.8 Splats that deform with the mesh

```js
const spark = new SparkRenderer({ renderer, covSplats: true, accumExtSplats: true });
const gs = new SplatMesh({ url, extSplats: true, covSplats: true });   // else: "CovSplats requires ExtSplats"
await gs.initialized;
gs.covObjectModifiers = [myCovModifier];   // NOT a constructor option: silently ignored there  [verified]
gs.updateGenerator();
// per frame: fill the RGBA32F DataTexture; tex.needsUpdate = true; gs.updateVersion();
```

`myCovModifier` is a per-splat-indexed clone of `spark/src/SplatSkinning.ts::applyCovSplatLBSkinning`: read a 3×3 `F`
and offset `t` from a float `DataTexture` (3 texels per splat), then `center = F·center + t`, `cov = F·cov·Fᵀ`.
Per frame on the CPU: `F_t = [e1′ e2′ n′]·invRest_t` per triangle, area-weighted onto vertices, `t = v′ − F·v`, singular
values clamped to ~0.5–2.5 so splats do not become needles. About 240 KB per frame for 20k splats. **[reported]**
Start from `spark/examples/interactive-deform/`.

Fallback if `covSplats` misbehaves (it is experimental, with no example exercising it): polar-decompose `F = R·U`,
rotate the splat quaternion by `R`, scale axes by `|U·axis_k|`, through a plain `objectModifier`. Loses shear.
Note Spark's raycast and SDF edits use **undeformed** splats, and LAM's own renderer also moves centres only, so the
project already matches official behaviour; this goes beyond it.

### 4.9 Hair, ears, back of head

±27° of yaw cannot recover these classically. **Cheapest fix, no ML: have someone take three phone photos (left,
right, back).** That converts hallucinated regions into measured ones.

A learned prior is second choice, and only as a prior for unseen regions: render it frontally, run MediaPipe on the
render, Umeyama-align to your measured landmarks, refine with `nricp_amberg`, keep **measured** geometry inside the
face mask, feather 10–15 mm. Status on this Mac **[reported unless tagged]**:

| Option | Licence | Status here |
|---|---|---|
| TRELLIS.2 via `trellis-mac` | MIT code + weights | **Blocked: installer needs `xcodebuild`** [measured]. DINOv3 backbone is gated |
| Pixal3D (multi-view via `transforms.json`) | MIT | **Blocked: same reason** [measured]; Mac port is an unmerged PR; 46 GB |
| Hunyuan3D-2 shape on MPS | Tencent licence: **not granted in EU / UK / South Korea** | Runs; output is watertight. Whether the territory term applies to you is your determination |
| SAM 3D Objects via `mlx-spatial` | SAM License | pip route, not blocked; outputs a Gaussian PLY; 13.7 GB weights; tiny port |
| SPAR3D / SF3D | Stability Community | Documented MPS support; low detail; gated weights |
| FaceLift | code Apache-2.0, **weights academic-only** | Grey for hobby use; needs patching for Mac |

### 4.10 Texturing

`colmap mesh_texturer` (COLMAP 4.2, Waechter et al. 2014, CPU) and OpenMVS `TextureMesh --mesh-file` both replace the
hand-made bake with view selection and seam levelling. Open3D 0.20 has
`o3d.t.geometry.TriangleMesh.project_images_to_albedo(...)` (py ≥ 3.10 only; 0.19 installs in neither env). **[reported]**

---

## 5. Deliberately not placed in the folder

Restricted code is cited by URL so it cannot be pasted into the project by accident.

| Project | Why cited only |
|---|---|
| 2DGS, RaDe-GS, PGSR, SuGaR, Gaussian Frosting, GaMeS | Inria-style non-commercial licences. Read for the algorithm |
| GaussianAvatars, SplattingAvatar | CC BY-NC-SA. SplattingAvatar's `master` deleted all files (use commit `6fe2554`) [reported] |
| PhysGaussian, Mani-GS | No licence. `mpm_utils.py::compute_cov_from_F` is the `cov = F·cov₀·Fᵀ` reference |
| Spirula Studio (GPL-3.0) | Most promising surface-aware splat mesher for Apple GPUs, but its macOS build is an **ad-hoc-signed, non-notarized DMG**. I did not download it |
| OpenMVS v2.4.0 (AGPL-3.0) | Official CPU `OpenMVS_macOS_arm64.zip` exists (65 MB) [verified]. A binary: your decision |
| Apple Object Capture | Proprietary (free) engine. The best-ranked OSS wrapper had **0 stars and was two weeks old**, so I did not fetch or build it. Use Apple's documented `PhotogrammetrySession` API and Apple's own sample instead |
| Sapiens2 | Custom licence with a biometric-processing clause |
| `affromero/splattie` | 1.1 GB, 8 stars, Spark dependency pinned to a personal fork. Reference only |
| `meshmonk` | Moved from its original org to a personal account [verified]; trimesh NRICP suffices |
| pip / npm / brew tools | `pytetwild`, `manifold3d`, `genesis-world`, `ipctk`, `mlx-vlm`, `uniface`, `metal-gauss`, `jolt-physics`, `three-mesh-bvh`: install when adopted, nothing to vendor |

Two agents disagreed about OpenMVS ("runs CPU-only" vs "needs CUDA"). I settled it from the release assets: a CPU
macOS arm64 zip ships *separately* from a Windows CUDA build, so CUDA is optional. **[verified]**

---

## 6. Decisions that are yours

- **Install full Xcode?** Unblocks `trellis-mac`, Pixal3D and OpenSplat-Metal. Needs your Apple ID.
- **`brew install colmap`?** 22 dependencies. Cheapest test of the arm fix.
- **Run a non-notarized DMG (Spirula) or a prebuilt binary (OpenMVS)?** I will not make that call for you.
- **Download model weights / accept gated licences / log into Hugging Face?** Multi-GB each; several are non-commercial.
- **Register for FLAME 2023 Open?** CC-BY-4.0, still needs a sign-up.
- **Hosted demos** (ReconViaGen accepts many uncalibrated views, TRELLIS.2, Hunyuan3D, FaceLift): your face photos leave the machine.
- **Privacy warning.** A research agent, reading the studio's `app.py::upload2oss`, reports that the ModelScope LAM demo
  (`Damo_XR_Lab/LAM_Large_Avatar_Model`) uploads the exported avatar zip to a **public** bucket under a
  timestamp-guessable name. **[reported, not confirmed by me]** I would not put your face through it. LAM generation is
  CUDA-only in any case, and its weights are CC BY-NC.

---

## 7. Suggested order, with pass/fail gates

1. **Camera.** Calibrate once; fix the legacy-reuse trap. *Pass:* fixture focal within ~3 % of 1250.
2. **Re-solve the existing 83-frame capture** with locked `K`. *Pass:* withheld-landmark error does not get worse.
3. **TSDF from rendered depth** on the existing `face.ply`. *Pass:* the 18 mm landmark gate, first at the chin and forehead.
4. **Template wrap** onto the TSDF surface. *Pass:* watertight, recognisable, glasses region masked.
5. **Arm:** ALIKED + LightGlue with the wide rigid mask and locked `K`. *Pass:* ≥ ~50/59 registered, spread well above 11°.
6. **Physics:** Worker XPBD at 5k tets behind the adapter. *Pass:* existing `tests/physics.test.mjs` invariants (bounded, returns to rest).
7. **Splats:** restore Spark with `covSplats`. *Pass:* no needle artefacts at the README's 24–25 mm peak displacement.

---

## 8. How this was produced, and its limits

Seven parallel research agents (Claude Fable 5.1) each covered one gap, working from live GitHub / PyPI / Hugging Face
data. I then independently re-checked licence, size, activity and ownership for every candidate through the GitHub API,
read licence texts where the API said `NOASSERTION` (MoGe, Anny, Ten Minute Physics), and verified locally whatever
could be verified locally (Spark internals, wheel resolution, toolchain, intrinsics against ground truth).

Limits: nothing here has been integrated or run end to end. Star counts and "last pushed" dates are a snapshot.
Runtime and memory figures marked [reported] or [unverified] are estimates until measured on this machine.
While this was written, a Codex session was actively editing this repo (adding `public/head-template/`,
`scripts/fit_head_template.py`, `scripts/photo_dense.py`, `scripts/photo_hair.py`), so file and line references
reflect 2026-09-19 ~01:50 and may have moved since. `npm test` was 20/20 before this work and 23/23 after it
(checked 01:56); the additions here touch no file the tests import.
