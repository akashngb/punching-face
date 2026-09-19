"""Photo-classified, editable strand groom on the measured hair envelope.

Photo flow is fused into visible lock paths. Browser fibers are deterministic, with
triangle/barycentric binding to preserve edits, deformation and session saves.
"""

import numpy as np
import cv2


def scalp_weight(x, p, hair):
    height = p[10, 1] - p[152, 1]
    azimuth = np.abs(np.arctan2(x[:, 0], x[:, 2] + 0.09))
    threshold = np.interp(
        azimuth,
        [0, 0.6, 1.45, 2.15, np.pi],
        [
            p[10, 1] + 0.002,
            p[10, 1] - 0.012,
            p[10, 1] - height * hair.get('sideHairlineFraction', 0.3),
            p[10, 1] - height * hair.get('rearHairlineFraction', 0.85),
            p[10, 1] - height * hair.get('rearHairlineFraction', 0.85),
        ],
    )
    w = np.clip((x[:, 1] - threshold) / 0.008, 0, 1)
    return w * w * (3 - 2 * w)


def build_hair_groom(
    p,
    faces,
    spec,
    rec=None,
    center=None,
    B=None,
    transform=None,
    folder=None,
    ear_regions=None,
):
    hair = (spec or {}).get('hair', {})
    if (
        not hair.get('present')
        or hair.get('type') == 'bald'
        or hair.get('density', 0.8) <= 0
    ):
        return None
    from scripts.ear_fit import ear_vertex_labels

    excluded_faces = (ear_vertex_labels(len(p), ear_regions)[faces] > 0).any(1)
    if folder is not None and rec is not None:
        return build_photo_strands(
            p, faces, spec, rec, center, B, transform, folder, excluded_faces
        )
    tri = p[faces]
    centroids = tri.mean(1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) * 0.5
    normals = cross / np.maximum(2 * area[:, None], 1e-12)
    weight = scalp_weight(centroids, p, hair)
    # Internal anatomy and the neck cut never grow hair.
    radial = centroids - np.array([0, 0.025, -0.09])
    weight *= np.sum(normals * radial, axis=1) > 0
    weight[excluded_faces] = 0
    if rec is not None:
        world = (centroids / transform['scale']) @ B + center
        for view in spec.get('views', []):
            regions = view.get('hairRegions', [])
            if not regions:
                continue
            im = next(
                (im for im in rec.images.values() if im.name == view['filename']), None
            )
            if im is None:
                continue
            cam = rec.cameras[im.camera_id]
            pose = im.cam_from_world()
            xy = cam.img_from_cam(world @ pose.rotation.matrix().T + pose.translation)
            crop = np.asarray(spec['crops'][im.name])
            mask = np.zeros((cam.height, cam.width), np.uint8)
            for polygon in regions:
                if len(polygon) >= 3:
                    poly = np.rint(
                        np.asarray(polygon) * (crop[2:] - crop[:2]) + crop[:2]
                    ).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 255)
            ij = np.rint(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(int)
            inside = (
                (ij[:, 0] >= 0)
                & (ij[:, 0] < cam.width)
                & (ij[:, 1] >= 0)
                & (ij[:, 1] < cam.height)
            )
            toward = im.projection_center() - world
            toward /= np.maximum(np.linalg.norm(toward, axis=1)[:, None], 1e-9)
            facing = np.sum((normals @ B) * toward, axis=1)
            visible = inside & (facing > 0.55)
            # Apply only facing evidence: a front polygon cannot erase rear hair.
            hit = (
                mask[
                    np.clip(ij[:, 1], 0, cam.height - 1),
                    np.clip(ij[:, 0], 0, cam.width - 1),
                ]
                > 0
            )
            weight[visible & ~hit] = 0
    probability = area * weight
    if probability.sum() <= 1e-9:
        return None
    count = int(2500 + hair.get('density', 0.8) * 8500)
    rng = np.random.default_rng(20260919)
    chosen = rng.choice(len(faces), count, p=probability / probability.sum())
    a = np.sqrt(rng.random(count))
    b = rng.random(count)
    bary = np.c_[1 - a, a * (1 - b), a * b]
    return {
        'version': 2,
        'type': 'strand-hair',
        'mode': 'editable-prior',
        'photoGuides': None,
        'source': 'Captured local hair flow and color; Astra classifies style without replacing its silhouette',
        'estimated': True,
        'parameters': hair,
        'seed': 20260919,
        'rootTriangles': faces[chosen].ravel().tolist(),
        'rootWeights': bary.astype(np.float32).ravel().tolist(),
        'sourceVertexCount': len(p),
        'hairlineY': float(p[10, 1]),
        'rootCount': count,
        'binding': 'Barycentric scalp roots; rigid skull motion and local surface edits',
        'limitation': (
            'Strand detail is a hairstyle-conditioned model, not measured '
            'individual hairs. Hidden hair and skin remain estimates.'
        ),
    }


def build_photo_strands(
    p, faces, spec, rec, center, B, transform, folder, excluded_faces=None
):
    from scipy.spatial import cKDTree
    from scripts.hair_flow import photo_field, trace_locks, unit, conform_curve_points
    from scripts.photo_detail import prepare_detail_frames

    prepare_detail_frames(folder)
    hair = spec['hair']
    tri = p[faces]
    centroids = tri.mean(1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) * 0.5
    normals = unit(cross)
    weight = scalp_weight(centroids, p, hair)
    weight *= np.sum(normals * (centroids - [0, 0.025, -0.09]), axis=1) > 0
    if excluded_faces is not None:
        weight[excluded_faces] = 0
    probability = area * weight
    if probability.sum() <= 1e-9:
        return None
    rng = np.random.default_rng(20260919)
    chosen = rng.choice(len(faces), 32000, p=probability / probability.sum())
    a = np.sqrt(rng.random(len(chosen)))
    b = rng.random(len(chosen))
    bary = np.c_[1 - a, a * (1 - b), a * b]
    roots = np.einsum('ij,ijk->ik', bary, tri[chosen])
    rn = normals[chosen]
    field = photo_field(folder, roots, rn, spec, rec, center, B, transform, (p, faces))
    supported = (
        (field['coverage'] > 0.65)
        & (field['observed'] > 0.08)
        & (field['confidence'] > 0.06)
    )
    if not supported.any():
        return None
    # Missing crown evidence does not authorize a carpet of long invented
    # locks. Its underlying photographic continuation remains labeled estimated.
    unseen = np.zeros(len(roots), bool)
    eligible = np.where(supported | unseen)[0]
    if not len(eligible):
        return None
    keep = rng.choice(eligible, min(8000, len(eligible)), replace=False)
    roots, rn, chosen, bary = roots[keep], rn[keep], chosen[keep], bary[keep]
    directions = field['directions'][keep]
    confidence = field['confidence'][keep]
    # Smooth signs locally without averaging opposing representations of the
    # same unoriented line. Low-contrast regions inherit nearby observed flow.
    tree = cKDTree(roots)
    _, near = tree.query(roots, k=min(16, len(roots)))
    if near.ndim == 1:
        near = near[:, None]
    for _ in range(3):
        local = directions[near].copy()
        local *= np.where(np.sum(local * directions[:, None], axis=2) < 0, -1, 1)[
            :, :, None
        ]
        weights = np.maximum(confidence[near], 0.015)
        smooth = unit(np.sum(local * weights[:, :, None], axis=1))
        directions = unit(directions * 0.6 + smooth * 0.4)
        directions = unit(directions - rn * np.sum(directions * rn, axis=1)[:, None])
    offsets, curve_normals, lengths = trace_locks(
        roots,
        rn,
        directions,
        confidence,
        {**hair, 'visibleSectionFraction': 0.38},
        p[10, 1],
    )
    import trimesh

    traced = (roots[:, None] + offsets).reshape(-1, 3)
    surface, station_normals, bindings = conform_curve_points(
        p,
        faces,
        traced,
        np.asarray(trimesh.Trimesh(p, faces, process=False).vertex_normals),
    )
    snap_distance = np.linalg.norm(surface - traced, axis=1)
    offsets = surface.reshape(offsets.shape) - roots[:, None]
    curve_normals = station_normals.reshape(curve_normals.shape)
    # Visible sections only. Global style length does not imply that a lock is
    # continuously observed over 5 cm, particularly on the short tapered sides.
    curve_field = photo_field(
        folder,
        (roots[:, None] + offsets).reshape(-1, 3),
        curve_normals.reshape(-1, 3),
        spec,
        rec,
        center,
        B,
        transform,
        (p, faces),
        colors_only=True,
    )
    curve_colors = curve_field['colors'].reshape(len(roots), offsets.shape[1], 3)
    curve_support = curve_field['coverage'].reshape(len(roots), offsets.shape[1])
    curve_colors = np.where(
        (curve_support > 0.5)[:, :, None], curve_colors, field['colors'][keep, None]
    )
    guide = {
        'directions': np.round(directions, 6).ravel().tolist(),
        'colors': np.round(np.clip(field['colors'][keep], 0, 1), 5).ravel().tolist(),
        'confidence': np.round(confidence, 5).tolist(),
        'curveOffsets': np.round(offsets, 6).ravel().tolist(),
        'curveNormals': np.round(curve_normals, 6).ravel().tolist(),
        'curveColors': np.round(curve_colors, 5).ravel().tolist(),
        'observedOnly': True,
        'curveBindings': bindings,
        'surfaceConformed': True,
        'segments': offsets.shape[1] - 1,
        'referenceLengthMm': hair['lengthMm'],
    }
    return {
        'version': 5,
        'type': 'strand-hair',
        'mode': 'photo-strands',
        'photoGuides': guide,
        'source': 'Multiview hair recognition, pixel orientation fusion and surface-traced locks',
        'estimated': True,
        'parameters': hair,
        'seed': 20260919,
        'rootTriangles': faces[chosen].ravel().tolist(),
        'rootWeights': bary.astype(np.float32).ravel().tolist(),
        'sourceVertexCount': len(p),
        'hairlineY': float(p[10, 1]),
        'rootCount': len(keep),
        'binding': 'Every curve station follows its local face-rig triangle with barycentric weights',
        'recognition': spec.get(
            'hairRecognition',
            {'model': spec.get('model'), 'framesSent': len(field['views'])},
        ),
        'evidence': {
            'views': field['views'],
            'photoSupportedRoots': int(supported[keep].sum()),
            'inferredCrownRoots': int(unseen[keep].sum()),
            'strongOrientationRoots': int((confidence > 0.08).sum()),
            'medianVisibleLockLengthMm': round(float(np.median(lengths) * 1000), 2),
            'stationSurfaceCorrectionP95Mm': round(
                float(np.percentile(snap_distance, 95) * 1000), 3
            ),
        },
        'limitation': (
            'Exterior lock flow and color follow the photographs. Individual '
            'fibers, hidden roots and subpixel detail remain estimates.'
        ),
    }
