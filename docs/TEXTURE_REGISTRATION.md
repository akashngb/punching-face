# Photograph registration around ears and the posterior head

Research review: 2026-09-19. Implementation: original NumPy/SciPy/OpenCV code
using this project's existing calibrated cameras, xatlas layout and photographs.
No research model, checkpoints, external inference service or new dependency was
installed for this change.

## Defects addressed

- Ear color coordinates received an affine correction, while the neighboring
  scalp abruptly reverted to the original projection. Hair/skin class masks
  were also read at the original coordinates instead of the corrected RGB ones.
- A 6 mm depth tolerance admitted surfaces just behind an ear. A raw depth
  comparison needed that tolerance because texel and raster samples used
  different rays on sloped triangles.
- Photo-mask disagreement removed some anatomical ear labels, permitting hair
  material on the back of the ear. Material identity and photographic visibility
  must remain separate.
- Any nonzero ear observation could disable completion, even when its physical
  support was extremely weak. Unseen rear folds lacked usable same-facing donors.
- Missing short side/rear hair reused the long front-quiff swatch.
- Sharp local surface normals could change scalp camera ownership abruptly.

## Implementation

`scripts/texture_registration.py` extends the existing ear affine photo warp
over actual mesh edges using geodesic distance and a smooth falloff. Measured
facial vertices stay pinned. Disconnected or spatially close surface sheets
do not share graph edges. Alpha, hair, skin and occluder masks use the same
corrected sample coordinates as RGB.

The extra confidence granted to oblique hair projections fades with image-warp
displacement and proximity along the surface to the ear. A strong original
photographic sample keeps its confidence. Inpainted source-ear pixels supply
no skull-color votes; otherwise a synthesized dark ear shadow can incorrectly
override the missing-surface estimate.

Posterior visibility compares the texel's triangle plane with the first depth
hit at the rasterizer's pixel-center ray, with a 0.5 mm tolerance in nominal
model units. It applies to the ears and lower posterior scalp; central
measured-face and crown visibility retain their existing behavior.
The tolerance is an engineering parameter, not a measurement-accuracy claim.

The bake retains anatomical ear material labels even when a photograph cannot
see them. Unsupported ear color continues over the connected ear mesh using
the existing positive-weight harmonic solver. These fills remain estimates;
they do not increase measured coverage. Scalp view preference uses a broad
head-relative direction, while actual visibility and masks remain authoritative.

`scripts/rear_hair.py` selects bounded, fully masked short-hair swatches from
side/rear photographs and excludes annotated ear and opaque-frame pixels.
They are used only in missing lower-hair appearance; available crown detail
retains its previous source. A missing side swatch can use the available rear
swatch, so unobserved sides are not claimed as reconstructed detail.

`scripts/rebuild_head_texture.py` rebuilds only appearance from cached inputs.
For older releases it recovers template ear correspondence after an exact
topology check, without adopting any fitted geometry. It checks unchanged
positions, normals, triangle indices, accessories, transform and physics files.
Publication uses the existing validated, atomic artifact transaction.

```sh
PYTHONPATH=. .venv/bin/python scripts/rebuild_head_texture.py CAPTURE_FOLDER
```

The normal full reconstruction also uses the new bake. No zbuffer,
atlas-rasterizer or detail-frame accelerator leaf was changed.

## Research and repositories reviewed

| Work | Relevant idea | What this implementation uses |
| --- | --- | --- |
| [Im2SurfTex (2025)](https://ygeorg01.github.io/Im2SurfTex/), [official repository](https://github.com/ygeorg01/Im2SurfTex) | Learned backprojection combines 3D geometry and geodesic neighborhoods to improve texture coherence. | Surface-connected registration falloff as an engineering design principle. Its attention network and generative pipeline were not ported or run. |
| [GOATex (NeurIPS 2025)](https://goatex3d.github.io/) | Ray-hit visibility layers distinguish exterior and occluded surfaces. | Explicit first-hit rejection for photographic projection. This is not GOATex's layered diffusion algorithm; no implementation repository was linked on the reviewed project page. |
| [MVPaint (CVPR 2025), official repository](https://github.com/3DTopia/MVPaint) | Separates multiview generation, spatial 3D completion and seam refinement. | Reviewed for separating projection from missing-surface completion. No MVPaint code or model is integrated. |
| [UniTEX (CVPR 2026), official repository](https://github.com/YixunLiang/UniTEX) | Continuous 3D texture functions avoid relying exclusively on UV-space operations. | Reviewed as a current research direction. Not integrated: the published setup requires CUDA, Kaolin and nvdiffrast, incompatible with this Mac's supported runtime. |
| [Zhou and Koltun, Color Map Optimization (SIGGRAPH 2014)](https://vladlen.info/publications/color-map-optimization-for-3d-reconstruction-with-consumer-depth-cameras/) | Joint photo/camera optimization with nonrigid image corrections. | Foundational motivation for correcting image registration instead of distorting accepted geometry. No implementation copied; camera poses are unchanged here. |

These papers mostly address generative asset texturing. Their published quality
claims do not transfer to this captured-person pipeline. This change is a
research-informed repair, not a reproduction of those systems or a claim of
state-of-the-art reconstruction quality.

## Validation and limitations

Regression tests exercise occluded surfaces separated by 2 mm, oblique
first-hit planes, missing depth, geodesic falloff, measured-face pins,
disconnected surfaces, camera selection with rejected samples, connected ear
completion, unchanged photographed donors and bounded short-hair sampling.
The focused Python suite passes 68 tests; viewer bundle, sponsor hooks and
Meshy hooks pass 9 Node tests. These are software checks, not a perceptual
quality benchmark of the cited research systems.

Visual review uses the saved capture with an identical-geometry before/after
render, including both profiles and the back. Low-resolution iterations are
for inspection only; the final bake uses the original 3072 resolution.
The scan was loaded through the normal viewer with Newton ready. Exact
geometry/accessory equality and unchanged physics files were verified against
the saved baseline. The bake's fallback-preservation audit also verifies that
supported prepared colors are not altered by missing-region completion; this
is distinct from comparing colors between different registration versions.

Unseen ear backs and hair gaps remain appearance estimates. Coarse masks,
camera registration errors, directional baked lighting and residual geometric
dents can still affect the result. A texture repair cannot establish geometry
or identity accuracy for regions absent from the video.

The reviewed result still has a visible transition behind the left ear. Ear
material assignment and the large misplaced hair patch improved, but this is
not a claim that the head now looks fully natural. The retained comparison is
at `http://localhost:5173/generated/rear-head-review.html?after=rear-texture-after`;
the machine-readable capture validation is in
`.local/rear-head-repair/texture-validation.json`.
