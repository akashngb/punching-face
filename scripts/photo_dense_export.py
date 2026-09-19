"""COLMAP-side export for the dense depth stage.

pycolmap and Open3D each bundle an OpenMP runtime and abort when loaded into one
process, so this file (pycolmap only) writes plain arrays that photo_dense.py
(Open3D only) reads. Usage: photo_dense_export.py <capture folder> <out.npz>
"""

from pathlib import Path
import sys
import numpy as np
import pycolmap


def export_cameras(rec, path):
    """Camera arrays plus bundle-adjusted sparse points with their quality."""
    images = sorted(rec.images.values(), key=lambda im: im.name)
    cam = rec.cameras[images[0].camera_id]
    if cam.model.name not in ('SIMPLE_RADIAL', 'SIMPLE_PINHOLE'):
        raise ValueError(
            'Dense depth expects the single recovered pinhole or radial camera.'
        )
    centers = {i: im.projection_center() for i, im in rec.images.items()}
    xyz = []
    angle = []
    error = []
    track = []
    for point in rec.points3D.values():
        # Widest angle between observing rays: narrow tracks have poor depth.
        rays = np.array([centers[e.image_id] for e in point.track.elements]) - point.xyz
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        xyz.append(point.xyz)
        angle.append(np.degrees(np.arccos(np.clip(rays @ rays.T, -1, 1).min())))
        error.append(point.error)
        track.append(len(rays))
    np.savez(
        path,
        names=np.array([im.name for im in images]),
        R=np.stack([im.cam_from_world().rotation.matrix() for im in images]),
        t=np.stack([im.cam_from_world().translation for im in images]),
        centers=np.stack([im.projection_center() for im in images]),
        params=np.asarray(cam.params, float),
        width=cam.width,
        height=cam.height,
        points=np.array(xyz).reshape(-1, 3),
        pointAngle=np.array(angle),
        pointError=np.array(error),
        pointTrack=np.array(track),
    )


if __name__ == '__main__':
    export_cameras(
        pycolmap.Reconstruction(str(Path(sys.argv[1]) / 'photo-cameras')), sys.argv[2]
    )
