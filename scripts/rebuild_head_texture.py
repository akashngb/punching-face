"""Rebake cached photographs while preserving the accepted head and physics.

Usage: PYTHONPATH=. .venv/bin/python scripts/rebuild_head_texture.py CAPTURE
No model inference or remote API requests. Publication validates the complete
bundle, and refuses a concurrent change to the accepted generation.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from face_pipeline import atomic
from head_artifacts import HeadArtifactTransaction, published_folder


def rebuild(folder):
    import pycolmap
    from scripts.hair_recognition import hair_completion
    from scripts.photo_geometry import bake_photographs
    from scripts.template_selection import normalized_frame

    folder = Path(folder).resolve()
    with HeadArtifactTransaction(folder, seed=True) as publication:
        accepted = published_folder(folder)
        data = json.loads((publication.stage / 'mesh.json').read_text())
        unchanged = deepcopy(
            {
                key: data.get(key)
                for key in (
                    'positions',
                    'normals',
                    'indices',
                    'accessories',
                    'transform',
                )
            }
        )
        physics = {
            name: (publication.stage / name).read_bytes()
            for name in ('physics-cage.json', 'physics-binding.json')
        }
        advice = json.loads((folder / 'astra-head-completion.json').read_text())
        if (folder / 'hair-recognition.json').exists():
            advice = hair_completion(
                advice, json.loads((folder / 'hair-recognition.json').read_text())
            )
        semantics = json.loads((folder / 'head-semantics.json').read_text())
        eyes = (
            json.loads((folder / 'eye-detail.json').read_text())
            if (folder / 'eye-detail.json').exists()
            else None
        )
        points = np.asarray(
            json.loads((accepted / 'surface-validation.json').read_text())[
                'landmarksWorld'
            ]
        )
        normalized, center, basis, _ = normalized_frame(points)
        # Earlier releases narrowed anatomical ear labels using unregistered
        # photo masks. Recover correspondence only; never adopt fitted positions.
        from scripts.fit_head_template import fit_template

        _, topology, _, template = fit_template(
            normalized, data['stats']['templateFit'].get('regularization', 0.025)
        )
        if not np.array_equal(topology.ravel(), data['indices']):
            raise ValueError(
                'Cannot recover anatomical material labels for this topology.'
            )
        for sign, region in data['stats']['templateFit']['earRegions'].items():
            region['coreVertices'] = template['earRegions'][sign]['coreVertices']
            region['anatomicalCoreVertices'] = region['coreVertices'].copy()
        data['stats']['earOwnership'] = {
            sign: {
                'earVertices': len(region['coreVertices']),
                'method': 'Anatomical material identity retained independently of photographic visibility',
                'geometryChanged': False,
            }
            for sign, region in data['stats']['templateFit']['earRegions'].items()
        }
        rec = pycolmap.Reconstruction(str(folder / 'photo-cameras'))
        frames = {
            frame['filename']: frame
            for frame in json.loads((folder / 'capture.json').read_text())['frames']
        }
        for im in rec.images.values():
            origin = (im.projection_center() - center) @ basis.T
            frames[im.name]['cameraYaw'] = float(
                np.degrees(np.arctan2(origin[0], origin[2]))
            )
        p = np.asarray(data['positions']).reshape(-1, 3)
        f = np.asarray(data['indices']).reshape(-1, 3)
        print(
            'Rebuilding photo registration on the accepted, unchanged mesh.', flush=True
        )
        data['stats']['appearance'] = bake_photographs(
            folder,
            p,
            f,
            data['stats']['observedFaceTriangles'],
            rec,
            frames,
            list(rec.images.values()),
            center,
            basis,
            data['transform'],
            advice,
            eyes,
            semantics,
            data['stats']['templateFit']['earRegions'],
            output_folder=publication.stage,
        )
        if any(data.get(key) != value for key, value in unchanged.items()):
            raise ValueError(
                'Texture rebuild attempted to change the accepted geometry or accessories.'
            )
        if any(
            (publication.stage / name).read_bytes() != value
            for name, value in physics.items()
        ):
            raise ValueError(
                'Texture rebuild attempted to change the physics bindings.'
            )
        atomic(publication.stage / 'mesh.json', data)
        publication.commit()
        print(
            'Published validated textures; geometry, hair groom and physics unchanged.',
            flush=True,
        )
        return data['stats']['appearance']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    import scripts.pipeline_accel as accel

    accel.install_leaves()
    rebuild(args.folder)
