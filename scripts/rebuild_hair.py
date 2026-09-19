"""Rebuild an existing capture's hair without repeating camera/face fitting.

Usage: PYTHONPATH=. .venv/bin/python scripts/rebuild_hair.py CAPTURE_FOLDER
       --recognize --refit-envelope --rebake
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pycolmap
import trimesh
from face_pipeline import atomic
from head_artifacts import HeadArtifactTransaction, published_folder
from scripts.hair_groom import build_hair_groom, scalp_weight
from scripts.hair_recognition import recognize_hair, hair_completion
from scripts.photo_hair import ear_hair_allowance


def refit_domain(p, faces, face_count, hair):
    """Keep an accepted nape transition outside the hair-only refit domain."""
    weight = scalp_weight(p, p, hair)
    angle = np.abs(np.arctan2(p[:, 0], p[:, 2] + 0.09))
    threshold = np.interp(angle, [0, 1, 1.7, np.pi], [p[10, 1], 0.065, 0.06, -0.035])
    weight[np.clip((p[:, 1] - threshold) / 0.018, 0, 1) < 0.98] = 0
    weight[:468] = 0
    weight[np.unique(faces[:face_count])] = 0
    return weight


def update_refit_baseline(stage, before, after):
    """Carry persisted hair displacement onto the independent pre-ear surface.

    Preserve its unfitted ears and all NPZ metadata. Applying the displacement
    rather than replacing positions avoids baking fitted ears into their own
    baseline and makes a subsequent ear rebuild retain the accepted scalp.
    """
    from scripts.surface_evidence import validate_refinement_baseline

    path = stage / 'pre-ear-surface.npz'
    if not path.is_file():
        raise ValueError(
            'Hair refit requires a compatible pre-ear baseline; run the complete pipeline.'
        )
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key].copy() for key in saved.files}
    rest = np.asarray(payload['positions'])
    baseline = {
        **before,
        'positions': rest.ravel().tolist(),
        'indices': np.asarray(payload['indices']).ravel().tolist(),
    }
    cage = json.loads((stage / 'physics-cage.json').read_text())
    validate_refinement_baseline(before, baseline, cage)
    regularization = before['stats']['templateFit'].get('regularization', 0.025)
    if float(payload['regularization']) != regularization:
        raise ValueError(
            'Hair baseline uses a different identity fit; run the complete pipeline.'
        )
    old = np.asarray(before['positions'], dtype=np.float32).reshape(-1, 3)
    new = np.asarray(after['positions'], dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(before['indices'], int).reshape(-1, 3)
    if (
        rest.shape != old.shape
        or new.shape != old.shape
        or before['indices'] != after['indices']
        or not np.isfinite(new).all()
    ):
        raise ValueError(
            'Hair refit changed baseline topology or has invalid positions.'
        )
    regions = before['stats']['templateFit'].get('earRegions')
    if not regions:
        raise ValueError(
            'Hair refit needs fitted ear regions to preserve the pre-ear baseline.'
        )
    protected = np.unique(
        np.concatenate(
            [
                np.arange(468),
                faces[: before['stats']['observedFaceTriangles']].ravel(),
                *[np.asarray(region['vertices'], int) for region in regions.values()],
            ]
        )
    )
    if not np.array_equal(new[protected], old[protected]):
        raise ValueError('Hair refit moved protected face, cage or ear vertices.')
    # Float64 subtraction preserves the exact difference between persisted
    # float32 endpoints, including sub-ULP details in the original baseline.
    delta = new.astype(np.float64) - old.astype(np.float64)
    moved = np.any(delta != 0, axis=1)
    updated = rest.astype(np.float64).copy()
    updated[moved] += delta[moved]
    if not np.isfinite(updated).all():
        raise ValueError('Hair refit produced an invalid pre-ear baseline.')
    payload['positions'] = updated
    temporary = stage / '.pending-hair-baseline.npz'
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def rebuild(folder, recognize=False, refit=False, rebake=False):
    if refit and not rebake:
        raise ValueError(
            'An envelope refit changes the textured surface; include --rebake.'
        )
    with HeadArtifactTransaction(folder, seed=True) as publication:
        return _rebuild(folder, recognize, refit, rebake, publication)


def _rebuild(folder, recognize, refit, rebake, publication):
    accepted = published_folder(folder)
    start = time.perf_counter()
    data = json.loads((accepted / 'mesh.json').read_text())
    original_positions = data['positions']
    if refit and not (publication.stage / 'pre-ear-surface.npz').is_file():
        raise ValueError(
            'Hair refit requires a compatible pre-ear baseline; run the complete pipeline.'
        )
    advice = json.loads((folder / 'astra-head-completion.json').read_text())
    analysis = folder / 'hair-recognition.json'
    if recognize:
        advice = hair_completion(advice, recognize_hair(folder, advice))
    elif analysis.exists():
        advice = hair_completion(advice, json.loads(analysis.read_text()))
    frames = {
        v['filename']: v
        for v in json.loads((folder / 'capture.json').read_text())['frames']
    }
    rec = pycolmap.Reconstruction(str(folder / 'photo-cameras'))
    points = np.array(
        json.loads((accepted / 'surface-validation.json').read_text())['landmarksWorld']
    )
    center = (points[10] + points[152]) / 2
    right = points[263] - points[33]
    right /= np.linalg.norm(right)
    up = points[10] - points[152]
    up -= right * np.dot(up, right)
    up /= np.linalg.norm(up)
    B = np.stack([right, up, np.cross(right, up)])
    for im in rec.images.values():
        c = (im.projection_center() - center) @ B.T
        frames[im.name]['cameraYaw'] = float(np.degrees(np.arctan2(c[0], c[2])))
    # A versioned rest surface lets repeated contour fits remain deterministic.
    # Earlier backups may predate a complete head/template topology rebuild.
    backup = folder / 'before-hair-v5'
    backup.mkdir(exist_ok=True)
    for name in ('mesh.json', 'appearance.png', 'texture-atlas.json'):
        if not (backup / name).exists():
            shutil.copy2(accepted / name, backup / name)
    # Always refit from the same rest surface, never grow an already fitted cap.
    p = np.array(data['positions']).reshape(-1, 3)
    f = np.array(data['indices']).reshape(-1, 3)
    face_count = data['stats']['observedFaceTriangles']
    if refit:
        base = json.loads((backup / 'mesh.json').read_text())
        if (
            len(base['positions']) != len(data['positions'])
            or base['indices'] != data['indices']
        ):
            raise ValueError(
                'Head topology changed since the hair backup; rebuild from the capture pipeline.'
            )
        base_p = np.array(base['positions']).reshape(-1, 3)
        ears = data['stats'].get('templateFit', {}).get('earRegions')
        allowance = ear_hair_allowance(p, f, ears)
        hair_only = (refit_domain(base_p, f, face_count, advice['hair']) > 0) & (
            allowance > 0.99
        )
        p[hair_only] = base_p[hair_only]
        from scripts.hair_silhouette import refine_hair_contours
        from scripts.frame_evidence import assess_frames, choose_views

        selected = choose_views(
            list(rec.images.values()),
            frames,
            np.arange(-180, 180, 30),
            assess_frames(folder),
        )
        weights = refit_domain(p, f, face_count, advice['hair'])
        p, contour_fit = refine_hair_contours(
            folder,
            p,
            f,
            weights,
            allowance,
            [im for im in rec.images.values() if im.name in selected],
            rec,
            center,
            B,
            data['transform'],
        )
        data['stats']['hair'] = {
            **data['stats'].get('hair', {}),
            'method': 'Locally refined multiview hair contours; face, ears and nape preserved',
            'contourRefinement': contour_fit,
            'actualCrownAboveHairlineMm': round(
                float(p[:, 1].max() - p[10, 1]) * 1000, 2
            ),
        }
    mesh = trimesh.Trimesh(p, f, process=False)
    groom = build_hair_groom(
        p,
        f,
        advice,
        rec,
        center,
        B,
        data['transform'],
        folder,
        data['stats'].get('templateFit', {}).get('earRegions'),
    )
    if not groom or not groom['rootCount']:
        raise ValueError(
            'No photograph-supported hair could be reconstructed; saved model unchanged.'
        )
    data['positions'] = p.astype(np.float32).ravel().tolist()
    data['normals'] = np.array(mesh.vertex_normals).astype(np.float32).ravel().tolist()
    data['accessories']['hair'] = groom
    data['stats']['hairGroom'] = {
        'available': True,
        'version': groom['version'],
        'type': advice['hair']['type'],
        'roots': groom['rootCount'],
        'estimatedFibers': True,
        'recognition': groom['recognition'],
        **groom['evidence'],
    }
    if refit:
        update_refit_baseline(
            publication.stage, {**data, 'positions': original_positions}, data
        )
    if rebake:
        from scripts.photo_geometry import bake_photographs

        eye_path = folder / 'eye-detail.json'
        eyes = json.loads(eye_path.read_text()) if eye_path.exists() else None
        semantic_path = folder / 'head-semantics.json'
        semantics = (
            json.loads(semantic_path.read_text()) if semantic_path.exists() else None
        )
        ears = data['stats'].get('templateFit', {}).get('earRegions')
        data['stats']['appearance'] = bake_photographs(
            folder,
            p,
            f,
            face_count,
            rec,
            frames,
            list(rec.images.values()),
            center,
            B,
            data['transform'],
            advice,
            eyes,
            semantics,
            ears,
            output_folder=publication.stage,
        )
    # Other local tasks may update independent eye/accessory metadata while
    # the depth masks build. Preserve those updates; never overwrite a changed
    # surface with curves bound to an older rest pose.
    current = json.loads((accepted / 'mesh.json').read_text())
    if (
        current['positions'] != original_positions
        or current['indices'] != data['indices']
    ):
        raise ValueError(
            'The head surface changed during this rebuild. Rerun to bind hair to the current surface.'
        )
    current['accessories']['hair'] = groom
    current['stats']['hairGroom'] = data['stats']['hairGroom']
    if refit:
        current.update(positions=data['positions'], normals=data['normals'])
        current['stats']['hair'] = data['stats']['hair']
    if rebake:
        current['stats']['appearance'] = data['stats']['appearance']
    atomic(publication.stage / 'mesh.json', current)
    publication.commit()
    audit = {
        'seconds': round(time.perf_counter() - start, 2),
        'mode': groom['mode'],
        'roots': groom['rootCount'],
        'silhouetteRefit': refit,
        'textureRebaked': rebake,
        'recognition': groom['recognition'],
        **groom['evidence'],
        'backup': str(backup),
    }
    atomic(folder / 'hair-reconstruction.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--recognize', action='store_true')
    parser.add_argument('--refit-envelope', action='store_true')
    parser.add_argument('--rebake', action='store_true')
    args = parser.parse_args()
    rebuild(args.folder.resolve(), args.recognize, args.refit_envelope, args.rebake)
