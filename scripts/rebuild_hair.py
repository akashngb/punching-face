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
from scripts.hair_groom import build_hair_groom, scalp_weight
from scripts.hair_recognition import recognize_hair, hair_completion
from scripts.photo_hair import fit_template_hair


def rebuild(folder, recognize=False, refit=False, rebake=False):
    start = time.perf_counter()
    data = json.loads((folder/'mesh.json').read_text())
    original_positions = data['positions']
    advice = json.loads((folder/'astra-head-completion.json').read_text())
    analysis = folder/'hair-recognition.json'
    if recognize:
        advice = hair_completion(advice, recognize_hair(folder, advice))
    elif analysis.exists():
        advice = hair_completion(advice, json.loads(analysis.read_text()))
    frames = {v['filename']: v for v in json.loads((folder/'capture.json').read_text())['frames']}
    rec = pycolmap.Reconstruction(str(folder/'photo-cameras'))
    points = np.array(json.loads((folder/'surface-validation.json').read_text())['landmarksWorld'])
    center = (points[10]+points[152])/2
    right = points[263]-points[33]; right /= np.linalg.norm(right)
    up = points[10]-points[152]; up -= right*np.dot(up, right); up /= np.linalg.norm(up)
    B = np.stack([right, up, np.cross(right, up)])
    for im in rec.images.values():
        c = (im.projection_center()-center) @ B.T
        frames[im.name]['cameraYaw'] = float(np.degrees(np.arctan2(c[0], c[2])))
    backup = folder/'before-hair-v3'
    backup.mkdir(exist_ok=True)
    for name in ('mesh.json', 'appearance.png', 'texture-atlas.json'):
        if not (backup/name).exists():
            shutil.copy2(folder/name, backup/name)
    # Always refit from the same rest surface, never grow an already fitted cap.
    p = np.array(data['positions']).reshape(-1, 3); f = np.array(data['indices']).reshape(-1, 3)
    face_count = data['stats']['observedFaceTriangles']
    if refit:
        base = json.loads((backup/'mesh.json').read_text())
        if len(base['positions']) != len(data['positions']) or base['indices'] != data['indices']:
            raise ValueError('Head topology changed since the hair backup; rebuild from the capture pipeline.')
        base_p = np.array(base['positions']).reshape(-1, 3)
        hair_only = scalp_weight(base_p, base_p, advice['hair']) > 0
        hair_only[:468] = False; hair_only[np.unique(f[:face_count])] = False
        p[hair_only] = base_p[hair_only]
        p, evidence = fit_template_hair(folder, p, f, face_count, rec, frames,
            list(rec.images.values()), center, B, data['transform'])
        data['stats']['hair'] = evidence
    mesh = trimesh.Trimesh(p, f, process=False)
    groom = build_hair_groom(p, f, advice, rec, center, B, data['transform'], folder)
    if not groom or not groom['rootCount']:
        raise ValueError('No photograph-supported hair could be reconstructed; saved model unchanged.')
    data['positions'] = p.astype(np.float32).ravel().tolist()
    data['normals'] = np.array(mesh.vertex_normals).astype(np.float32).ravel().tolist()
    data['accessories']['hair'] = groom
    data['stats']['hairGroom'] = {'available': True, 'version': groom['version'],
        'type': advice['hair']['type'], 'roots': groom['rootCount'], 'estimatedFibers': True,
        'recognition': groom['recognition'], **groom['evidence']}
    if rebake:
        from scripts.photo_geometry import bake_photographs
        eye_path = folder/'eye-detail.json'
        eyes = json.loads(eye_path.read_text()) if eye_path.exists() else None
        data['stats']['appearance'] = bake_photographs(folder, p, f, face_count, rec,
            frames, list(rec.images.values()), center, B, data['transform'], advice, eyes)
    # Other local tasks may update independent eye/accessory metadata while
    # the depth masks build. Preserve those updates; never overwrite a changed
    # surface with curves bound to an older rest pose.
    current = json.loads((folder/'mesh.json').read_text())
    if current['positions'] != original_positions or current['indices'] != data['indices']:
        raise ValueError('The head surface changed during this rebuild. Rerun to bind hair to the current surface.')
    current['accessories']['hair'] = groom
    current['stats']['hairGroom'] = data['stats']['hairGroom']
    if refit:
        current.update(positions=data['positions'], normals=data['normals'])
        current['stats']['hair'] = data['stats']['hair']
    if rebake:
        current['stats']['appearance'] = data['stats']['appearance']
    atomic(folder/'mesh.json', current)
    audit = {'seconds': round(time.perf_counter()-start, 2), 'mode': groom['mode'],
        'roots': groom['rootCount'], 'silhouetteRefit': refit, 'textureRebaked': rebake,
        'recognition': groom['recognition'], **groom['evidence'], 'backup': str(backup)}
    atomic(folder/'hair-reconstruction.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--recognize', action='store_true')
    parser.add_argument('--refit-envelope', action='store_true')
    parser.add_argument('--rebake', action='store_true')
    args = parser.parse_args()
    rebuild(args.folder.resolve(), args.recognize, args.refit_envelope, args.rebake)
