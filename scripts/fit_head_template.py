"""Fit a complete CC0 head template with a smooth, regularized identity warp.

Measured face landmarks constrain broad shape. Template ears, eyelids, mouth,
neck and skull keep their topology. This is not a learned anatomical identity
model; unobserved posterior geometry is a prior until an orbit constrains it.
"""

from pathlib import Path
import json, numpy as np, trimesh
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scripts.prepare_head_template import load

ROOT = Path(__file__).resolve().parents[1]


def catmull_clark(p, q):
    fp = p[q].mean(1)
    edge_map = {}
    adj = [[] for _ in p]
    ve = [[] for _ in p]
    for fi, face in enumerate(q):
        for v in face:
            adj[v].append(fi)
        for i in range(4):
            key = tuple(sorted((face[i], face[(i + 1) % 4])))
            edge_map.setdefault(key, []).append(fi)
    keys = list(edge_map)
    edge_id = {k: i for i, k in enumerate(keys)}
    ep = []
    for a, b in keys:
        fs = edge_map[(a, b)]
        ep.append(
            (p[a] + p[b] + fp[fs].sum(0)) / 4 if len(fs) == 2 else (p[a] + p[b]) / 2
        )
        ve[a].append((a, b))
        ve[b].append((a, b))
    new = p.copy()
    for i in range(len(p)):
        boundary = [b if a == i else a for a, b in ve[i] if len(edge_map[(a, b)]) == 1]
        if len(boundary) == 2:
            new[i] = p[i] * 0.75 + p[boundary].sum(0) * 0.125
        elif adj[i]:
            n = len(adj[i])
            F = fp[adj[i]].mean(0)
            R = np.array([(p[a] + p[b]) / 2 for a, b in ve[i]]).mean(0)
            new[i] = (F + 2 * R + (n - 3) * p[i]) / n
    out = np.vstack([new, ep, fp])
    faces = []
    offset = len(p) + len(ep)
    for fi, face in enumerate(q):
        for i, v in enumerate(face):
            faces.append(
                [
                    v,
                    len(p) + edge_id[tuple(sorted((v, face[(i + 1) % 4])))],
                    offset + fi,
                    len(p) + edge_id[tuple(sorted((face[(i - 1) % 4], v)))],
                ]
            )
    return out, np.asarray(faces)


def project_cage(points, p, f):
    tri = p[f]
    _, candidates = cKDTree(tri.mean(1)).query(points, k=24)
    closest = trimesh.triangles.closest_point(
        tri[candidates].reshape(-1, 3, 3), np.repeat(points, 24, axis=0)
    ).reshape(-1, 24, 3)
    which = np.argmin(np.linalg.norm(closest - points[:, None], axis=2), axis=1)
    projected = closest[np.arange(len(points)), which]
    # Only snap depth; semantic ordering of the 468 frontal landmarks is
    # retained so the existing well-conditioned Newton cage stays valid.
    result = points.copy()
    result[:, 2] = projected[:, 2]
    return result


def fit_template(points, smoothing=0.025, check_regularization=False):
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError('Template regularization must be positive and finite.')
    p, _, q, _ = load()
    p, q = catmull_clark(p, q)
    canonical = p.copy()
    anchors = json.loads((ROOT / 'public/head-template/anchors.json').read_text())
    ids = np.array(list(map(int, anchors)))
    a = np.array(list(anchors.values()))
    # Symmetrize the neutral template registration, not the scanned identity.
    pairs = [
        (21, 251),
        (33, 263),
        (50, 280),
        (61, 291),
        (70, 300),
        (93, 323),
        (98, 327),
        (103, 332),
        (127, 356),
        (133, 362),
        (148, 377),
        (172, 397),
        (234, 454),
        (159, 386),
        (145, 374),
        (158, 385),
        (153, 380),
        (160, 387),
        (144, 373),
    ]
    for left, right in pairs:
        il = np.where(ids == left)[0][0]
        ir = np.where(ids == right)[0][0]
        avg = (a[ir] + a[il] * [-1, 1, 1]) / 2
        a[ir] = avg
        a[il] = avg * [-1, 1, 1]
    for i, idx in enumerate(ids):
        if idx in [0, 1, 4, 6, 9, 10, 17, 152, 168]:
            a[i, 0] = 0
    source_top = a[np.where(ids == 10)[0][0]]
    source_chin = a[np.where(ids == 152)[0][0]]
    sy = (points[10, 1] - points[152, 1]) / (source_top[1] - source_chin[1])
    sx = (points[454, 0] - points[234, 0]) / (
        a[np.where(ids == 454)[0][0], 0] - a[np.where(ids == 234)[0][0], 0]
    )
    sz = (sx + sy) / 2
    scale = np.array([sx, sy, sz])
    translation = points[1] - a[np.where(ids == 1)[0][0]] * scale
    p = p * scale + translation
    a = a * scale + translation
    from scripts.surface_evidence import bind_landmarks

    landmark_vertices, landmark_weights = bind_landmarks(
        a, p, np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    )
    # Posterior and neck support pins stop facial fitting from stretching the
    # back into a face-shaped cap. The warp is global and continuous, with no
    # piecewise nearest-neighbor kernels or exact noisy landmark spikes.
    support = np.where(
        (p[:, 2] < -0.095)
        | (p[:, 1] < points[152, 1] - 0.02)
        | (p[:, 1] > points[10, 1] + 0.045)
    )[0][::60]
    origin = np.vstack([a, p[support]])
    delta = np.vstack([points[ids] - a, np.zeros((len(support), 3))])
    warp = RBFInterpolator(
        origin / 0.1, delta, kernel='thin_plate_spline', smoothing=smoothing
    )
    fitted = p + warp(p / 0.1)
    regularization_quality = None
    if check_regularization and smoothing != 0.025:
        from scripts.ear_deformation import surface_quality
        from scripts.surface_intersections import new_crossings

        baseline = p + RBFInterpolator(
            origin / 0.1, delta, kernel='thin_plate_spline', smoothing=0.025
        )(p / 0.1)
        triangles = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        regularization_quality = surface_quality(baseline, fitted, triangles)
        regularization_quality['maximumChangeMm'] = float(
            np.linalg.norm(fitted - baseline, axis=1).max() * 1000
        )
        if (
            regularization_quality['minimumNormalAgreement'] < 0.5
            or regularization_quality['minimumAreaRatio'] < 0.5
            or regularization_quality['maximumAreaRatio'] > 2
            or regularization_quality['maximumChangeMm'] > 10
        ):
            raise ValueError(
                'Stronger identity fitting would excessively distort the template.'
            )
        regularization_quality.update(new_crossings(baseline, fitted, triangles))
        if regularization_quality['newCrossings']:
            raise ValueError('Stronger identity fitting introduced surface crossings.')
    # Keep semantic correspondence through the neck slice, which reindexes
    # vertices. Ear ownership must not depend on the later hairstyle envelope.
    pre_slice = fitted.copy()
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    m = trimesh.Trimesh(fitted, faces, process=False)
    m.fix_normals()
    faces = np.asarray(m.faces)
    # A planar neck cut avoids the jagged row left by selecting whole quads.
    edges, counts = np.unique(np.sort(m.edges, axis=1), axis=0, return_counts=True)
    boundary = np.unique(edges[counts == 1])
    cut = points[152, 1] - 0.024
    # Identity warping can lift part of the template's raw neck boundary above
    # a fixed cut. Cut above its highest point so no tiny open notches survive.
    if len(boundary):
        cut = max(cut, float(fitted[boundary, 1].max()) + 0.0005)
    if cut > points[152, 1] - 0.010:
        raise ValueError('The neck template warp is too distorted to close reliably.')
    m = trimesh.Trimesh(fitted, faces, process=False).slice_plane(
        [0, cut, 0], [0, 1, 0], cap=True
    )
    fitted = np.asarray(m.vertices)
    faces = np.asarray(m.faces)
    distance, landmark_vertices = cKDTree(fitted).query(pre_slice[landmark_vertices])
    if distance.max() > 1e-7:
        raise ValueError('The neck slice removed a facial landmark binding.')
    landmark_binding = {
        'landmarkIds': ids.tolist(),
        'vertices': (landmark_vertices + 468).tolist(),
        'weights': landmark_weights.tolist(),
    }
    from scripts.ear_fit import template_ear_regions

    ear_regions = template_ear_regions(canonical, pre_slice, fitted)
    # Keep connected spherical eye surfaces inside the template sockets. Their
    # appearance comes from the photographs; their rigid geometry is a prior.
    for sign in [-1, 1]:
        eye = trimesh.creation.uv_sphere(radius=0.0142, count=[24, 24])
        center = np.array([sign * 0.0308, 0.0184, -0.0105]) * scale + translation
        center += warp(center[None] / 0.1)[0]
        ev = np.array(eye.vertices) * scale + center
        offset = len(fitted)
        fitted = np.vstack([fitted, ev])
        faces = np.vstack([faces, np.array(eye.faces) + offset])
    m = trimesh.Trimesh(fitted, faces, process=False)
    m.fix_normals()
    faces = np.array(m.faces)
    tri = fitted[faces]
    normal = np.asarray(m.face_normals)
    front = (
        (tri[:, :, 2].mean(1) > -0.060)
        & (normal[:, 2] > -0.15)
        & (tri[:, :, 1].mean(1) < points[10, 1] + 0.006)
        & (tri[:, :, 1].mean(1) > points[152, 1] - 0.006)
        & (np.abs(tri[:, :, 0].mean(1)) < np.ptp(points[:, 0]) * 0.51)
    )
    face_count = int(front.sum())
    faces = np.vstack([faces[front], faces[~front]])
    coarse = project_cage(points, fitted, faces[:face_count])
    p = np.vstack([coarse, fitted])
    f = faces + 468
    return (
        p,
        f,
        face_count,
        {
            'template': 'MakeHuman hm08, CC0',
            'method': 'Full head template fitted with a regularized global landmark warp and Catmull-Clark subdivision.',
            'landmarkAnchors': len(ids),
            'regularization': smoothing,
            'regularizationQuality': regularization_quality,
            'landmarkBindings': landmark_binding,
            'templateVertices': len(fitted),
            'depthMm': float(np.ptp(fitted[:, 2]) * 1000),
            'earRegions': ear_regions,
            'limitation': (
                'Eyes and unseen skull retain template priors; captured '
                'landmarks constrain face and visible ear shape.'
            ),
        },
    )
