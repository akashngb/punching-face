# Impact validation — 2026-09-19

Simulator/engineering validation, using the fitted photo head already on this machine (19,789 positions, 38,630 indexed triangles; one rest-degenerate triangle is excluded from area diagnostics). No new capture, camera use, external image upload, or asset download was needed.

## Automated results

- Final combined run with the actual local capture enabled: **80 tests passed, zero failed, zero skipped**. The parallel cumulative-deformation task independently ran the full suite with `FACE_IMPACT_CAPTURE` and confirmed these totals after the solver changes were finalized.
- Production build passed. `git diff --check` passed.
- Actual-capture regression: four same-patch clay strikes deepen the dent; finite positions, no reversed triangles, reset, mode preservation, and saved-dent restoration pass.
- Repeated rear hits retain welded seams and stay within the 65 mm cap. A nearby unsaturated patch and the opposite side still accept new impacts.
- Sliver-triangle regression checks area/orientation throughout rest-to-impact interpolation.

## Rendered checks

The production snapshot is available at `http://localhost:5173/impact-preview/index.html`. This avoids development hot reloads during observation, while using the same reconstruction and Newton services. Source remains available at the usual `http://localhost:5173/`.

| Contact | Magnitude | Measured peak / retained dent | Inspected geometry |
| --- | ---: | --- | --- |
| Left cheek, oblique | 0.85 | 40.96 mm peak, then recovery to zero | 0 reversed triangles |
| Mouth, oblique | 0.75 | 28.23 mm peak, then recovery to zero | 0 reversed triangles |
| Chin / uppercut | 0.80 | 20.50 mm peak, then recovery to zero | 0 reversed triangles |
| Forehead | 0.90 | 16.42 mm peak; **0 permanent** | 0 reversed at peak and recovery |
| Forehead | 1.00 | 18.76 mm peak; **14.63 mm retained** | 0 reversed at peak and settled |
| Left temple | 0.85 | 17.63 mm recorded peak; 0 permanent | 0 reversed in inspected frames |
| Right temple | 0.80 | 16.48 mm recorded peak; 0 permanent | 0 reversed in inspected frames |
| Crown, clay | 0.70 | 12.64 mm retained | 0 reversed triangles |
| Back, clay | 0.80 | 16.11 mm retained | 0 reversed triangles |

Timing is simulation time. The remote Newton offsets can recover later than the browser corrective when server throughput is limited; temple snapshots still had submillimetre transient motion at the first recovery check. These are inspected poses, not a continuous self-collision proof.

The saved-session roundtrip on the photo head produced **maximum vertex difference 0**, with the 16.11 mm back dent preserved. A final held cheek pose with wireframe and rig markers had a 40.68 mm recorded peak, **zero reversed triangles**, minimum area ratio 0.3004, and finite positions. Its local screenshot is `.local/impact-evidence/wireframe-impact.png`.

The cumulative-deformation task also verified the visible same-cheek sequence **36.2 → 54.4 → 62.2 → 65.0 mm**, full-power clay deformation, live recovery, and Reset head.

## Interpretation

The reference photos guided cheek compression, broad skin transport, asymmetric lip/jaw motion and head recoil. This is a heuristic facial animation rig layered over the existing solver. It does not establish hyperrealism, biomechanical accuracy, fracture prediction, or clinical validity. Internal fascia, bones and ligaments are not subject-fitted. Fable was unavailable in the connected tools and plugin lookup; validation was performed with the local rendered app, geometry diagnostics and automated tests. See `IMPACT_RIG.md` for the integration contract and primary research links.
