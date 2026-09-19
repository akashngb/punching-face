"""Fit template ears to validated multi-view ear observations."""

import json, itertools
import numpy as np
import cv2
from scipy.spatial import cKDTree

KEYS = ('top', 'bottom', 'tragus', 'upperAttachment', 'lowerAttachment')


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def template_ear_regions(canonical, warped, sliced):
    """CC0 template correspondence, retained across slicing and identity fit."""
    _, nearest = cKDTree(warped).query(sliced)
    c = canonical[nearest]
    x, y, z = c.T
    weight = (
        smooth((abs(x) - 0.058) / 0.012)
        * smooth((y + 0.052) / 0.020)
        * smooth((0.045 - y) / 0.018)
        * smooth((z + 0.140) / 0.025)
        * smooth((-0.048 - z) / 0.014)
    )
    targets = {
        'top': [0.080045, 0.024035, -0.08341625],
        'bottom': [0.06856, -0.0234525, -0.083705],
        'tragus': [0.07206625, 0.002765, -0.07199063],
    }
    regions = {}
    for sign in [-1, 1]:
        ids = np.where((weight > 0) & (x * sign > 0))[0]
        anchors = {
            name: int(np.argmin(np.linalg.norm(c - np.array(v) * [sign, 1, 1], axis=1)))
            + 468
            for name, v in targets.items()
        }
        # Ownership follows the auricle, not the rectangular deformation
        # support. Labeling that whole support erases adjacent temple skin.
        outline = ((y[ids] + 0.001) / 0.030) ** 2 + ((z[ids] + 0.085) / 0.021) ** 2
        core = (abs(x[ids]) > 0.064) & (outline < 1.05)
        regions[str(sign)] = {
            'vertices': (ids + 468).tolist(),
            'weights': weight[ids].tolist(),
            'coreVertices': (ids[core] + 468).tolist(),
            'anchors': anchors,
        }
    return regions


def fit_ears(p, f, face_count, regions, measurements, protected_vertices=None):
    """Bounded local affine fit preserves ear folds and pins the facial rig."""
    result = p.copy()
    audit = {}
    protected = np.zeros(len(p), bool)
    protected[:468] = True
    protected[np.unique(f[:face_count])] = True
    if protected_vertices is not None:
        protected[np.asarray(protected_vertices, int)] = True
    distance = cKDTree(p[protected]).query(p)[0]
    for sign, region in regions.items():
        observed = measurements.get(sign, {}).get('landmarks', {})
        keys = [k for k in ('top', 'bottom', 'tragus') if k in observed]
        if len(keys) < 3:
            audit[sign] = {
                'accepted': False,
                'reason': 'Three ear landmarks need agreement from independent views.',
            }
            continue
        a = np.array([p[region['anchors'][k]] for k in keys])
        b = np.array([observed[k] for k in keys])
        if not np.isfinite(b).all() or any(
            np.linalg.norm(np.cross(v[0] - v[1], v[2] - v[1])) < 1e-5 for v in (a, b)
        ):
            audit[sign] = {
                'accepted': False,
                'reason': 'Ear landmarks do not establish a stable, non-collinear surface.',
            }
            continue
        ac = a - a.mean(0)
        bc = b - b.mean(0)
        u, s, vt = np.linalg.svd(ac.T @ bc)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        scale = float(np.sum(s * np.diag(correction)) / np.sum(ac * ac))
        if not 0.70 < scale < 1.65:
            audit[sign] = {
                'accepted': False,
                'reason': 'Ear scale inconsistent with the head prior.',
                'scale': scale,
            }
            continue
        ids = np.array(region['vertices'])
        weight = np.array(region['weights']) * smooth(distance[ids] / 0.006)
        weight[protected[ids]] = 0

        def basis(v):
            u = v[0] - v[1]
            t = v[2] - v[1]
            n = np.cross(u, t)
            n /= np.linalg.norm(n)
            return np.stack([u, t, n * 0.03])

        affine = np.linalg.solve(basis(a), basis(b))
        stretches = np.linalg.svd(affine, compute_uv=False)
        if stretches.min() < 0.5 or stretches.max() > 2:
            audit[sign] = {
                'accepted': False,
                'reason': 'Ear observations require excessive local distortion.',
            }
            continue
        # Preserve a little of the shape prior rather than exactly interpolating
        # approximate image annotations. Normal thickness remains the prior.
        fitted = 0.90 * ((p[ids] - a[1]) @ affine + b[1]) + 0.10 * (
            (p[ids] - a.mean(0)) @ rotation * scale + b.mean(0)
        )
        anchor_fit = 0.90 * ((a - a[1]) @ affine + b[1]) + 0.10 * (
            ac @ rotation * scale + b.mean(0)
        )
        residual = float(np.max(np.linalg.norm(anchor_fit - b, axis=1)))
        delta = (fitted - p[ids]) * weight[:, None]
        if residual > 0.012 or np.max(np.linalg.norm(delta, axis=1)) > 0.065:
            audit[sign] = {
                'accepted': False,
                'reason': 'Ear alignment exceeded bounded fit tolerance.',
                'residualMm': residual * 1000,
            }
            continue
        result[ids] += delta
        region['observedLandmarks'] = observed
        region['coreVertices'] = [i for i in region['coreVertices'] if not protected[i]]
        actual_residual = max(
            float(np.linalg.norm(result[region['anchors'][k]] - observed[k]))
            for k in keys
        )
        audit[sign] = {
            'accepted': True,
            'scale': scale,
            'residualMm': residual * 1000,
            'movedVertices': int(np.sum(weight > 0)),
            'surfaceAnchorResidualMm': actual_residual * 1000,
            'protectedVertices': int(protected.sum()),
            'observations': measurements[sign]['audit'],
            'method': 'Multiview triangulation and bounded local affine fit; inner ear folds remain a template prior.',
        }
    if not np.array_equal(result[protected], p[protected]):
        raise ValueError('Ear fitting changed protected facial rig vertices.')
    accepted = {
        sign: regions[sign] for sign, entry in audit.items() if entry['accepted']
    }
    if accepted:
        from scripts.ear_deformation import regularize

        result, quality = regularize(p, result, f, accepted, protected)
        for sign, region in accepted.items():
            audit[sign]['surfaceQuality'] = quality
            audit[sign]['surfaceAnchorResidualMm'] = max(
                float(
                    np.linalg.norm(result[index] - measurements[sign]['landmarks'][key])
                )
                * 1000
                for key, index in region['anchors'].items()
            )
            audit[sign]['limitation'] = (
                'Ear position is constrained by source observations but limited '
                'to avoid folding the scalp; inner folds remain a template '
                'estimate.'
            )
    return result, audit


def ear_vertex_labels(count, regions):
    labels = np.zeros(count, np.uint8)
    for sign, region in (regions or {}).items():
        labels[np.asarray(region['coreVertices'], int)] = 3 if int(sign) < 0 else 4
    return labels


def refine_ear_ownership(
    folder, p, faces, regions, semantics, rec, center, B, transform
):
    """Remove adjacent skull vertices from ear ownership using source views."""
    import trimesh
    from scripts.photo_geometry import zbuffer

    detail_path = folder / 'photo-detail.json'
    detail = (
        {
            v['filename']: v
            for v in json.loads(detail_path.read_text()).get('frames', [])
        }
        if detail_path.exists()
        else {}
    )
    labels = ear_vertex_labels(len(p), regions)
    votes = np.zeros(len(p))
    seen = np.zeros(len(p))
    world = p / transform['scale'] @ B + center
    normal = np.asarray(trimesh.Trimesh(p, faces, process=False).vertex_normals) @ B
    images = {im.name: im for im in rec.images.values()}
    for view in semantics['views']:
        im = images.get(view['filename'])
        if im is None:
            continue
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        cp = world @ pose.rotation.matrix().T + pose.translation
        xy = cam.img_from_cam(cp)
        ij = np.rint(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(int)
        inside = (
            (ij[:, 0] >= 0)
            & (ij[:, 0] < cam.width)
            & (ij[:, 1] >= 0)
            & (ij[:, 1] < cam.height)
            & (cp[:, 2] > 0)
        )
        x = np.clip(ij[:, 0], 0, cam.width - 1)
        y = np.clip(ij[:, 1], 0, cam.height - 1)
        depth = zbuffer(xy, cp[:, 2], faces, cam.width, cam.height)
        toward = im.projection_center() - world
        toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-9)
        facing = np.maximum(np.sum(normal * toward, axis=1), 0)
        weight = (
            facing**2
            * inside
            * (np.abs(cp[:, 2] - depth[y, x]) * transform['scale'] < 0.006)
            * (labels > 0)
        )
        scale = detail.get(im.name, {}).get('size', [cam.width])[0] / cam.width
        mask, _ = source_ear_mask(
            (cam.height, cam.width),
            im.name,
            semantics,
            (im.projection_center() - center) @ B.T,
            scale,
        )
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        seen += weight
        votes += weight * (mask[y, x] == labels)
    supported = (seen < 0.15) | (votes >= seen * 0.4)
    audit = {}
    for sign, region in regions.items():
        before = region['coreVertices']
        region.setdefault('anatomicalCoreVertices', before.copy())
        region['coreVertices'] = [i for i in before if supported[i]]
        audit[sign] = {
            'templateVertices': len(before),
            'earVertices': len(region['coreVertices']),
            'reclassifiedAdjacentSurface': len(before) - len(region['coreVertices']),
            'method': 'Visibility-weighted agreement with photographed ear outlines; hidden vertices retain the prior.',
        }
    return audit


def source_ear_mask(shape, filename, semantics, origin, pixel_scale):
    mask = np.zeros(shape[:2], np.uint8)
    view = next(
        (v for v in semantics.get('views', []) if v['filename'] == filename), None
    )
    if not view:
        return mask, False
    crop = np.array(semantics['crops'][filename])
    extent = crop[2:] - crop[:2]
    for key in ('imageLeftEar', 'imageRightEar'):
        ear = view[key]
        outline = np.asarray(ear['outline'], float)
        if not ear['visible'] or ear['confidence'] < 0.45 or len(outline) < 3:
            continue
        sign = (
            int(np.sign(origin[0]))
            if abs(origin[0]) > abs(origin[2]) * 0.2
            else (-1 if key == 'imageLeftEar' else 1)
        )
        points = np.rint((outline * extent + crop[:2]) / pixel_scale).astype(np.int32)
        cv2.fillPoly(mask, [points], 3 if sign < 0 else 4)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8)), True


def ear_sample_coordinates(
    xy,
    parts,
    p,
    regions,
    semantics,
    im,
    cam,
    center,
    B,
    transform,
    pixel_scale,
    registration_weights=None,
):
    """Register ear photographs with optional surface-continuous scalp falloff.

    The surface fit can stop short of noisy observations to avoid folds. A
    bounded per-ear texture registration keeps its photograph on that surface.
    The same correction fades over connected scalp when weights are supplied.
    """
    output = xy.copy()
    view = next(
        (v for v in semantics.get('views', []) if v['filename'] == im.name), None
    )
    if not view:
        return output
    origin = (im.projection_center() - center) @ B.T
    pose = im.cam_from_world()
    crop = np.array(semantics['crops'][im.name])
    for key in ('imageLeftEar', 'imageRightEar'):
        ear = view[key]
        if not ear['visible'] or ear['confidence'] < 0.65:
            continue
        sign = (
            int(np.sign(origin[0]))
            if abs(origin[0]) > abs(origin[2]) * 0.2
            else (-1 if key == 'imageLeftEar' else 1)
        )
        region = regions.get(str(sign))
        if not region:
            continue
        keys = ('top', 'bottom', 'tragus')
        ids = [region['anchors'][k] for k in keys]
        world = p[ids] / transform['scale'] @ B + center
        model = cam.img_from_cam(world @ pose.rotation.matrix().T + pose.translation)
        source = (
            np.array([ear[k] for k in keys]) * (crop[2:] - crop[:2]) + crop[:2]
        ) / pixel_scale
        matrix = cv2.getAffineTransform(np.float32(model), np.float32(source))
        stretch = np.linalg.svd(matrix[:, :2], compute_uv=False)
        if (
            not np.isfinite(matrix).all()
            or np.linalg.det(matrix[:, :2]) <= 0
            or stretch.min() < 0.35
            or stretch.max() > 2.5
        ):
            continue
        owned = parts == (3 if sign < 0 else 4)
        if registration_weights is None:
            output[owned] = xy[owned] @ matrix[:, :2].T + matrix[:, 2]
        else:
            weight = registration_weights.get(str(sign), np.zeros(len(xy))).copy()
            weight[owned] = 1
            active = weight > 1e-6
            corrected = xy[active] @ matrix[:, :2].T + matrix[:, 2]
            output[active] += (corrected - xy[active]) * weight[active, None]
    return output


def predicted_ear_mask(shape, regions, im, cam, center, B, transform):
    """Conservative fallback contour for views without an ear annotation."""
    mask = np.zeros(shape[:2], np.uint8)
    origin = (im.projection_center() - center) @ B.T
    pose = im.cam_from_world()
    if abs(origin[0]) < abs(origin[2]) * 0.25:
        return mask
    sign = int(np.sign(origin[0]))
    landmarks = (regions.get(str(sign)) or {}).get('observedLandmarks', {})
    if not all(k in landmarks for k in ('top', 'bottom', 'tragus')):
        return mask
    top, bottom, tragus = [np.array(landmarks[k]) for k in ('top', 'bottom', 'tragus')]
    middle = (top + bottom) / 2
    up = top - bottom
    length = np.linalg.norm(up)
    up /= length
    forward = tragus - middle
    forward -= up * np.dot(forward, up)
    width = np.linalg.norm(forward)
    forward /= max(width, 1e-9)
    angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = (
        middle
        + np.sin(angle)[:, None] * up * length * 0.55
        + np.cos(angle)[:, None] * forward * max(width, length * 0.27)
    )
    cp = (
        ring / transform['scale'] @ B + center
    ) @ pose.rotation.matrix().T + pose.translation
    if (cp[:, 2] <= 0).any():
        return mask
    xy = cam.img_from_cam(cp)
    if not np.isfinite(xy).all():
        return mask
    cv2.fillPoly(mask, [np.rint(xy).astype(np.int32)], 3 if sign < 0 else 4)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def triangulate_ears(folder, rec, center, B, transform, semantics):
    observations = {-1: {k: [] for k in KEYS}, 1: {k: [] for k in KEYS}}
    images = {im.name: im for im in rec.images.values()}
    detail_path = folder / 'photo-detail.json'
    detail_frames = (
        json.loads(detail_path.read_text()).get('frames', [])
        if detail_path.exists()
        else []
    )
    for view in semantics['views']:
        im = images.get(view['filename'])
        if im is None:
            continue
        camera = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        origin = (im.projection_center() - center) @ B.T * transform['scale']
        crop = np.array(semantics['crops'][im.name])
        detail = next((v for v in detail_frames if v['filename'] == im.name), None)
        scale = (detail['size'][0] / camera.width) if detail else 1.0
        for key in ('imageLeftEar', 'imageRightEar'):
            ear = view[key]
            if not ear['visible'] or ear['confidence'] < 0.7:
                continue
            sign = (
                int(np.sign(origin[0]))
                if abs(origin[0]) > abs(origin[2]) * 0.2
                else (-1 if key == 'imageLeftEar' else 1)
            )
            for name in KEYS:
                xy = (np.array(ear[name]) * (crop[2:] - crop[:2]) + crop[:2]) / scale
                ray = np.r_[camera.cam_from_img(xy), 1.0] @ pose.rotation.matrix() @ B.T
                ray /= np.linalg.norm(ray)
                observations[sign][name].append(
                    (origin, ray, ear['confidence'], im, xy)
                )
    output = {}
    for sign, landmarks in observations.items():
        fitted = {}
        audit = {}
        for name, items in landmarks.items():
            if len(items) < 2:
                continue
            origins = np.array([i[0] for i in items])
            rays = np.array([i[1] for i in items])
            weights = np.array([i[2] for i in items])
            projectors = np.eye(3) - rays[:, :, None] * rays[:, None, :]
            best = None
            for pair in itertools.combinations(range(len(items)), 2):
                if abs(np.dot(rays[pair[0]], rays[pair[1]])) > 0.985:
                    continue
                A = projectors[list(pair)].sum(0)
                rhs = np.einsum(
                    'nij,nj->i', projectors[list(pair)], origins[list(pair)]
                )
                if np.linalg.cond(A) > 1e4:
                    continue
                q = np.linalg.solve(A, rhs)
                distance = np.linalg.norm(np.cross(q - origins, rays), axis=1)
                inliers = distance < 0.005
                score = (int(inliers.sum()), -float(np.median(distance)))
                if best is None or score > best[0]:
                    best = (score, inliers, q)
            if best is None or best[1].sum() < 2:
                continue
            take = best[1]
            w = weights * take
            A = np.einsum('n,nij->ij', w, projectors)
            q = np.linalg.solve(A, np.einsum('n,nij,nj->i', w, projectors, origins))
            errors = []
            for item, used in zip(items, take):
                if not used:
                    continue
                _, _, _, im, xy = item
                cam = rec.cameras[im.camera_id]
                pose = im.cam_from_world()
                world = q / transform['scale'] @ B + center
                projected = cam.img_from_cam(
                    pose.rotation.matrix() @ world + pose.translation
                )
                errors.append(float(np.linalg.norm(projected - xy)))
            if (
                max(errors) > 5
                or q[0] * sign < 0.055
                or abs(q[0]) > 0.18
                or abs(q[1]) > 0.13
                or not -0.25 < q[2] < 0.02
            ):
                continue
            fitted[name] = q.tolist()
            audit[name] = {
                'views': int(take.sum()),
                'maxReprojectionPx': round(max(errors), 3),
            }
        output[str(sign)] = {'landmarks': fitted, 'audit': audit}
    return output
