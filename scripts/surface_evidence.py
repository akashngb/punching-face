"""Measure semantic landmarks on the rendered surface, not only its rig cage."""

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def bind_landmarks(points, vertices, faces):
    triangles = vertices[faces]
    k = min(32, len(faces))
    _, near = cKDTree(triangles.mean(axis=1)).query(points, k=k)
    near = np.asarray(near).reshape(len(points), k)
    closest = trimesh.triangles.closest_point(
        triangles[near].reshape(-1, 3, 3), np.repeat(points, k, axis=0)
    ).reshape(-1, k, 3)
    best = np.argmin(np.linalg.norm(closest - points[:, None], axis=2), axis=1)
    face_ids = near[np.arange(len(points)), best]
    surface = closest[np.arange(len(points)), best]
    weights = trimesh.triangles.points_to_barycentric(triangles[face_ids], surface)
    return faces[face_ids], weights


def landmark_positions(vertices, binding):
    return np.sum(
        vertices[np.asarray(binding['vertices'], int)]
        * np.asarray(binding['weights'])[:, :, None],
        axis=1,
    )


def validate_refinement_baseline(current, baseline, cage):
    """A saved ear rest surface must belong to the currently installed rig."""
    faces = np.asarray(current['indices'], int).reshape(-1, 3)
    now = np.asarray(current['positions'], dtype=np.float32).reshape(-1, 3)
    rest = np.asarray(baseline['positions'], dtype=np.float32).reshape(-1, 3)
    if rest.shape != now.shape or not np.array_equal(
        faces.ravel(), baseline['indices']
    ):
        raise ValueError('Ear baseline topology is stale; run the complete pipeline.')
    protected = np.unique(
        np.r_[
            np.arange(468), faces[: current['stats']['observedFaceTriangles']].ravel()
        ]
    )
    if not np.isfinite(rest).all() or not np.array_equal(
        rest[protected], now[protected]
    ):
        raise ValueError(
            'Ear baseline does not match the current facial surface; run the complete pipeline.'
        )
    if not np.array_equal(
        np.asarray(cage['positions'], dtype=np.float32), now[:468].ravel()
    ):
        raise ValueError(
            'The saved physics cage does not match the current face; run the complete pipeline.'
        )


def surface_projection_error(
    vertices, binding, rec, frames, images, center, B, transform
):
    positions = landmark_positions(vertices, binding) / transform['scale'] @ B + center
    indices = np.asarray(binding['landmarkIds'], int)
    errors = []
    for im in images:
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        predicted = cam.img_from_cam(
            positions @ pose.rotation.matrix().T + pose.translation
        )
        measured = np.array(
            [
                [q['x'] * cam.width, q['y'] * cam.height]
                for q in frames[im.name]['landmarks']
            ]
        )[indices]
        errors.extend(np.linalg.norm(predicted - measured, axis=1))
    values = np.asarray(errors)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Rendered surface has invalid landmark projections.')
    return {
        'medianPx': float(np.median(values)),
        'p95Px': float(np.quantile(values, 0.95)),
        'landmarksPerView': len(indices),
        'views': len(images),
        'usesRenderedSurface': True,
    }
