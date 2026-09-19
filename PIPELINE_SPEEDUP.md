# Video to model in under a minute: what was measured, what is done, what is left

Written 2026-09-19 by Claude for the owner and for Codex. Everything below was measured on this machine
(Apple M5 Pro, 18 cores, `.venv` Python 3.9) on capture `7a2bc070…` (`IMG_7496.MOV`, 24.3 s, 51 views, 40 cameras).
No OpenAI request was made while measuring: AI calls were replayed from their cache files after a fixed delay
(`complete` 37 s, hair 30 s, eyes 10 s, semantics 45 s; the first two split the measured 66.6 s by output size, the
semantics figure is an assumption).

## The short answer

Parallelism alone cannot do it, because most of the three minutes was not waiting, it was one slow function.
But the pipeline can get under a minute in three layers. The first is done and verified. The other two need edits
inside `bake_photographs()` and `run()`, which Codex is actively changing, so they are written up as a recipe instead
of being forked.

| Fresh build of this capture, server side | Seconds |
| --- | ---: |
| Today's code, serial (`build_photo_face.py`) | **308.7** |
| Layer 1, shipped: exact accelerators + concurrent AI calls (`build_photo_face_fast.py`) | **122.4** |
| Layer 2, recipe below: restructure the bake loop, overlap local work with the AI wait | about 75 |
| Layer 3, recipe below: trim unused AI output, start AI with camera recovery | about 45 |

Both measured rows are fresh builds (every cache deleted) of the same frozen copy of the code as of 05:55, run back to
back, and **every output is identical**: `appearance.png` and `appearance-roughness.png` byte for byte, `mesh.json`
positions, normals, indices, rig anchors and accessories (glasses and the hair groom), `texture-atlas.json`,
`physics-*.json`, all 51 native detail frames, and all four AI cache files.

| Stage, seconds | Serial | Layer 1 |
| --- | ---: | ---: |
| cameras | 10.6 | 13.8 |
| astra (`complete`, then hair) | 67.1 | 37.1 |
| eyes | 10.5 | 0.0 |
| semantics (includes first native-frame decode when serial) | 100.9 | 7.6 |
| accessory-cleanup | 14.3 | 0.6 |
| surface, hair envelope, rig | 3.2 | 2.3 |
| accessories (glasses, hair groom) | 27.5 | 3.1 |
| texture | 74.7 | 57.9 |
| **total** | **308.7** | **122.4** |

In the layer 1 run the four AI calls all started at 14 s and the last finished at 59 s; serially they ran until 179 s.
Cameras are slower in the layer 1 run because native frames decode alongside COLMAP. The serial run reused the
layer 1 run's recovered cameras and its 10.6 s camera time comes from a separate serial run; see the COLMAP note
under "Measuring" for why.

Browser frame extraction (15.6 s) and model load (about 1 s) come on top of every row. `PIPELINE_BENCHMARK.md`'s
180.6 s predates eye detail, native detail frames, groom v4 and the ear/semantics stage; today's fresh build is
far slower than that because of what those stages added, mostly one hidden cost (see finding 2).

## Findings

1. **`photo_geometry.zbuffer` is a per-triangle Python loop, called 30 times per build.** 1.5 to 2.5 s per call:
   16 calls in the bake and 14 in `hair_flow.photo_field`, which rasterizes the same mesh from the same cameras twice
   more. 74 s of a 198 s cached rebuild. Vectorized it is 0.05 s and bit-identical; shared between groom and bake it
   is computed once per camera. The groom drops from 42.4 s to 2.8 s from this alone.
2. **`prepare_detail_frames` costs 58 s on a fresh capture and no benchmark has ever shown it.** It seeks the HEVC
   recording 255 times (5 candidates for each of 51 views), and every seek decodes forward from a keyframe. Every
   rebuild so far found `photo-detail.json` already cached. Streaming the file once takes 6.0 s and writes
   byte-identical PNGs and an identical audit.
3. **Four AI calls run one after another and none needs another's answer.** `recognize_hair` and
   `head_semantics.analyze` read only the *filenames and crops* that `complete()` chose; `scan_eyes` reads only the
   capture. Started together they cost the slowest one, not the sum (about 122 s serial with the delays above, 45 s
   overlapped).
4. **`world@R.T` on the 6.3M-texel array takes 0.66 to 0.92 s, and Accelerate serializes concurrent callers.**
   `(R@world.T).T` is bit-identical and takes 0.027 s. This one line is about 14 s of the bake, and it is also why
   threading the camera loop gives no speedup until it is fixed (8 concurrent products took 8.1 s).
5. **The bake does all per-camera work on every texel, though a camera sees about a quarter of them.** A texel with
   `quality == 0` adds exactly 0 to every accumulator and can never win `quality>detail_best`, so skipping it is exact.
6. **About 46 % of `complete()`'s output is never used.** `eyewearRegions` (1,444 of 6,020 characters) has no reader
   anywhere in the repo. `complete()`'s `hairRegions` (1,315) are always overwritten by `recognize_hair`'s in
   `hair_completion()` whenever a groom is built. These calls are bound by output tokens, so this is latency.

## Measuring: what invalidates a comparison here

- **COLMAP is not deterministic run to run.** The same 51 frames registered 40 cameras in one run and 49 in another.
  Two fresh builds therefore never match, and `complete()` may choose different views, so replayed AI answers fail
  its filename check. An A/B needs one shared `photo-cameras/`.
- **A fresh build and a rebuild from cached cameras visit views in different orders**, because `rec.images` is ordered
  differently when mapped in memory than when loaded from disk. `eyewear-mask-audit.json` lists the same 17 records in
  another order, and the bake's `take=quality>detail_best` ties and float sums depend on that order. The texture was
  byte-identical here, but sorting `train` by name in `run()` would make it so by construction.
- **The code changes underneath a long benchmark.** `ear_fit.py`, `photo_geometry.py` and `build_photo_face.py` were
  all edited between two runs eight minutes apart, which moved 1,053 ear vertices by up to 4.9 cm and looked like an
  accelerator bug until the sources' mtimes were checked. Freeze a copy of `*.py` and `scripts/` and run from that.
- Machine load matters too: the reference bake took 82 s on a quiet machine and 146 s while a hair rebuild was running.

## Layer 1: shipped, exact, no edits to Codex's functions

- `scripts/pipeline_accel.py`: drop-in `zbuffer` (vectorized, cached per camera), `raster_atlas`,
  `map_coordinates` (sample list split across threads), `prepare_detail_frames` (streamed, one decode per folder
  even with concurrent callers), and `install_prefetch()`, which starts hair, semantics, eyes and native-frame
  decoding early. Prefetch calls the pipeline's own stage functions; each writes its usual cache file, and the serial
  call that follows in `run()` reads it. A failed prefetch is ignored and the serial call runs as before.
- `scripts/build_photo_face_fast.py`: installs the above, then calls `build_photo_face.run()` unchanged.
- `face_pipeline.py` launches the fast script. **Restart `npm run dev` to pick this up.**
  `CONTACT_SERIAL_PIPELINE=1` restores the plain script; `CONTACT_PREFETCH=0` keeps AI calls serial.
- `tests/pipeline_accel_test.py`: bit-identity against the reference implementations on synthetic meshes
  (clipped, off-screen and degenerate triangles included), pin behaviour, and prefetch: four calls overlap, each is
  requested exactly once, a reordered model answer still hits the prefetched cache, a failed prefetch falls back.
  Pass a capture folder to also compare streamed native frames byte for byte with the seeking decoder.

**Drift guard.** Each replacement is pinned to the SHA-256 of the reference source it replaces. If
`photo_geometry.zbuffer`, `photo_geometry.raster_atlas` or `photo_detail.prepare_detail_frames` is edited, that
replacement steps aside, the edited reference runs, and the build log says `STALE`. Port the edit, then
`.venv/bin/python scripts/pipeline_accel.py --pin`. `--status` shows what is active. The references stay reachable
as `photo_geometry.reference_zbuffer` and so on. Standalone scripts (`rebuild_hair.py`, `rebuild_head_details.py`)
can opt in with `import scripts.pipeline_accel as accel; accel.install_leaves()` before their other imports.

Not verified: behaviour against the live API. Four concurrent vision requests could meet a rate limit that serial
requests would not; a 429 on a prefetch falls back to the serial request a few seconds later. A subject with no hair
costs one wasted hair request, because hair recognition starts before `complete()` says whether hair is present.

## Layer 2: inside `bake_photographs()` and `run()` (Codex)

Measured on the pre-ear version of the bake with layer 1 installed: **54 s to 17.5 s, `appearance.png`,
`appearance-roughness.png` and the mask audit byte-identical.** The ear logic added since is per camera and per texel,
so the same split applies.

1. Replace `cp=world@R.T+translation` with `cp=(R@world.T).T+translation`, and build `world`/`world_n` the same way
   (`accel.rows(X,M)` is `X@M`). About 14 s, one line, bit-identical.
2. Move the loop body into `project_view(im)` that returns its contributions, and cull before sampling:
   compute `facing`, then `near=np.flatnonzero((facing>0)&(cp[:,2]>0)&(observed|head_capture))`; project only
   `cp[near]`; after `quality` is formed (including `quality[bottom]=0` and the ear `mismatch`), keep
   `index=near[quality>0]` and run `map_coordinates`, `fringe`, the colour correction and `masked` on that subset.
3. `ThreadPoolExecutor(8).map(project_view, views)`, then reduce **in the original view order** with indexed
   updates: `total[index]+=quality`, `accum[index]+=low_rgb*quality[:,None]`,
   `take=quality>detail_best[index]; fine_detail[index[take]]=(rgb-low_rgb)[take]`, and so on. Order matters for
   `take` ties and for float sums; with it preserved the PNG is byte-identical. The 16-camera phase took 3.3 s.
4. In `run()`, after `fix_normals()` the groom, `physics_binding` and the bake are independent of each other.
   Run them together; write `mesh.json` once (it is 35 MB and is written twice, about 0.8 s each).
5. `fit_template` and `fit_template_hair` (about 6 s) need only cameras, not AI. With three or more rear views
   `apply_shape_prior` is a no-op, so the mesh is final before the AI answers and xatlas (5.4 s), `raster_atlas`
   and the z-buffers can also run during the AI wait.

## Layer 3: the AI wait itself

After layer 2 the critical path is cameras (13 to 15 s), then the slowest AI call, then about 12 s.

1. Remove `eyewearRegions` from `complete()`'s schema and prompt, and `hairRegions` when a head capture will run
   `recognize_hair` anyway. No `VERSION` bump is needed: old caches only carry extra unused keys, and both readers use
   `.get()`.
2. Start the AI calls with camera recovery instead of after it. `complete()` chooses views by `cameraYaw`; MediaPipe
   `yaw` is already its fallback for facial frames. Only the two rear views need another rule before cameras exist,
   for example the middle of the longest run of frames without landmarks.
3. `head_semantics.analyze` allows 24,000 output tokens and is now the slowest call. It could run one request per
   view in parallel, since no view's annotation depends on another's.
4. Browser extraction (15.6 s) awaits each frame's upload before seeking the next. A promise chain keeps order while
   letting seek and detection run ahead.
