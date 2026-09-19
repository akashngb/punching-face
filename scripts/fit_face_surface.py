"""A connected face fitted to captured camera rays, with an estimated cranium.

Only topology comes from the canonical model. Facial vertex coordinates are
triangulated from the user's images. Gaussian detail is tightly bounded and
unobserved completion is explicitly labeled, never counted as captured depth.
"""

from pathlib import Path
import argparse, json, sys, time, shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pycolmap, trimesh, xatlas
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.interpolate import RBFInterpolator
from scipy.sparse import coo_matrix
from scipy.ndimage import distance_transform_edt
from plyfile import PlyData
from face_pipeline import atomic
from scripts.train_face import triangulate

ROOT = Path(__file__).resolve().parents[1]
OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


def camera_rays(rec, frames, images):
    centers = []
    directions = []
    for im in images:
        cam = rec.cameras[im.camera_id]
        xy = np.array(
            [
                [p['x'] * cam.width, p['y'] * cam.height]
                for p in frames[im.name]['landmarks']
            ]
        )
        xy = cam.cam_from_img(xy)
        pose = im.cam_from_world()
        rays = np.column_stack([xy, np.ones(len(xy))]) @ pose.rotation.matrix()
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        centers.append(im.projection_center())
        directions.append(rays)
    return np.asarray(centers), np.asarray(directions)


def robust_landmarks(centers, directions):
    projectors = np.eye(3) - directions[:, :, :, None] * directions[:, :, None, :]
    weight = np.ones(directions.shape[:2])
    points = None
    for _ in range(8):
        A = np.einsum('vn,vnij->nij', weight, projectors)
        b = np.einsum('vn,vnij,vj->ni', weight, projectors, centers)
        if np.max(np.linalg.cond(A)) > 1e6:
            raise ValueError('Insufficient angular diversity for stable face depth.')
        points = np.linalg.solve(A, b)
        delta = points[None] - centers[:, None]
        errors = np.linalg.norm(np.cross(delta, directions), axis=-1)
        robust_scale = np.maximum(np.median(errors, axis=0) * 1.4826, 1e-7)
        weight = np.minimum(1, 1.5 * robust_scale / np.maximum(errors, 1e-9))
    return points, errors


def make_surface(points, back_ratio):
    faces = []
    for line in (
        (ROOT / 'public/models/canonical_face_model.obj').read_text().splitlines()
    ):
        if line.startswith('f '):
            faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:]])
    face_count = len(faces)
    verts = points.tolist()
    boundary = points[OVAL]
    previous = np.array(OVAL)
    width = np.ptp(boundary[:, 0])
    height = np.ptp(boundary[:, 1])
    rear_depth = width * back_ratio
    middle = np.array([np.mean(boundary[:, 0]), height * 0.06, 0.0])
    # This smooth cap is an explicit prior. It does not claim recovered ears,
    # hair, scalp or occipital shape, and receives a separate neutral texture.
    for theta in np.linspace(0, np.pi / 2, 20)[1:-1]:
        ring = middle + (boundary - middle) * np.cos(theta)
        ring[:, 2] -= rear_depth * np.sin(theta)
        ring[:, 1] += (
            height
            * 0.31
            * np.sin(2 * theta)
            * np.clip((boundary[:, 1] / height + 0.20) / 0.70, 0, 1)
        )
        current = np.arange(len(verts), len(verts) + len(OVAL))
        verts.extend(ring.tolist())
        for j in range(len(OVAL)):
            k = (j + 1) % len(OVAL)
            faces.extend(
                [
                    [int(previous[j]), int(current[j]), int(previous[k])],
                    [int(previous[k]), int(current[j]), int(current[k])],
                ]
            )
        previous = current
    pole = len(verts)
    verts.append([middle[0], middle[1], middle[2] - rear_depth])
    for j in range(len(OVAL)):
        faces.append([int(previous[j]), pole, int(previous[(j + 1) % len(OVAL)])])
    base = trimesh.Trimesh(verts, faces, process=False)
    base.fix_normals()
    if not base.is_watertight:
        raise ValueError('Head completion has an invalid boundary.')
    p, f = trimesh.remesh.subdivide_loop(base.vertices, base.faces, iterations=2)
    # Restore observed landmarks after smooth subdivision, without copying a
    # stock face shape. Anchor rear points to keep the correction local.
    pins = np.r_[np.arange(468), np.arange(468, len(base.vertices), 12)]
    target = np.asarray(base.vertices)[pins]
    correction = target - p[pins]
    warp = RBFInterpolator(
        p[pins], correction, kernel='thin_plate_spline', smoothing=1e-10, neighbors=24
    )
    p += warp(p)
    p[:468] = points
    return p, f, face_count * 16, rear_depth


def fit_gaussian_detail(folder, positions, faces, face_count, transform, bound_mm):
    m = trimesh.Trimesh(positions, faces, process=False)
    normals = np.array(m.vertex_normals)
    v = PlyData.read(folder / 'face.ply')['vertex'].data
    xyz = (np.column_stack([v[k] for k in 'xyz']) - transform['center']) * transform[
        'scale'
    ]
    sizes = (
        np.exp(np.column_stack([v[f'scale_{i}'] for i in range(3)]))
        * transform['scale']
    )
    alpha = 1 / (1 + np.exp(-np.clip(v['opacity'], -30, 30)))
    order = np.sort(sizes, axis=1)
    valid = (
        (alpha > 0.015)
        & (order[:, 0] / np.maximum(order[:, 1], 1e-9) < 0.35)
        & (order[:, 2] / np.maximum(order[:, 1], 1e-9) < 8)
        & np.isfinite(xyz).all(axis=1)
    )
    if np.count_nonzero(valid) < 100:
        return positions, 0.0
    rot = Rotation.from_quat(
        np.column_stack([v[f'rot_{i}'] for i in [1, 2, 3, 0]])
    ).as_matrix()
    gn = rot[np.arange(len(v)), :, np.argmin(sizes, axis=1)][valid]
    distance, ids = cKDTree(xyz[valid]).query(positions, k=16)
    delta = xyz[valid][ids] - positions[:, None]
    alignment = np.abs(np.einsum('nki,ni->nk', gn[ids], normals))
    weights = (
        alpha[valid][ids]
        * alignment**4
        * np.exp(-((distance / 0.004) ** 2))
        * (distance < 0.008)
        * (alignment > 0.7)
    )
    offset = (weights * np.einsum('nki,ni->nk', delta, normals)).sum(
        axis=1
    ) / np.maximum(weights.sum(axis=1), 1e-9)
    support = np.clip(weights.sum(axis=1), 0, 1)
    front = np.zeros(len(positions))
    front[np.unique(faces[:face_count])] = 1
    edge_distance = cKDTree(positions[OVAL]).query(positions)[0]
    support *= np.clip(edge_distance / 0.018, 0, 1) * front
    # Glasses and mobile eyelid/mouth regions do not drive skin geometry.
    eyes = np.minimum(
        np.linalg.norm((positions - positions[33]) / [0.03, 0.025, 0.035], axis=1),
        np.linalg.norm((positions - positions[263]) / [0.03, 0.025, 0.035], axis=1),
    )
    support *= np.clip(eyes - 0.6, 0, 1)
    offset = np.clip(offset, -bound_mm / 1000, bound_mm / 1000) * support
    edges = m.edges_unique
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    degree = np.bincount(rows, minlength=len(positions))
    adj = coo_matrix(
        (1 / np.maximum(degree[rows], 1), (rows, cols)),
        shape=(len(positions), len(positions)),
    ).tocsr()
    for _ in range(8):
        offset = (0.6 * offset + 0.4 * (adj @ offset)) * front
    offset[:468] = 0
    return positions + normals * offset[:, None], float(np.max(np.abs(offset)) * 1000)


def projection_error(points, rec, frames, images):
    errors = []
    for im in images:
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        predicted = cam.img_from_cam(
            points @ pose.rotation.matrix().T + pose.translation
        )
        observed = np.array(
            [
                [p['x'] * cam.width, p['y'] * cam.height]
                for p in frames[im.name]['landmarks']
            ]
        )
        errors.extend(np.linalg.norm(predicted - observed, axis=1).tolist())
    return {
        'medianPx': float(np.median(errors)),
        'p95Px': float(np.quantile(errors, 0.95)),
    }


def raster_atlas(p, n, mapping, indices, uv, is_face, size):
    world = np.zeros((size, size, 3), np.float32)
    normal = np.zeros_like(world)
    covered = np.zeros((size, size), bool)
    observed = np.zeros_like(covered)
    for i, triangle in enumerate(indices):
        t = uv[triangle] * size
        lo = np.maximum(np.floor(t.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(t.max(axis=0)).astype(int), size - 1)
        if np.any(lo > hi):
            continue
        xx, yy = np.meshgrid(
            np.arange(lo[0], hi[0] + 1) + 0.5, np.arange(lo[1], hi[1] + 1) + 0.5
        )
        d = (t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0]) + (t[2, 0] - t[1, 0]) * (
            t[0, 1] - t[2, 1]
        )
        if abs(d) < 1e-9:
            continue
        a = (
            (t[1, 1] - t[2, 1]) * (xx - t[2, 0]) + (t[2, 0] - t[1, 0]) * (yy - t[2, 1])
        ) / d
        b = (
            (t[2, 1] - t[0, 1]) * (xx - t[2, 0]) + (t[0, 0] - t[2, 0]) * (yy - t[2, 1])
        ) / d
        c = 1 - a - b
        inside = (a >= -1e-5) & (b >= -1e-5) & (c >= -1e-5)
        w = np.stack([a, b, c], axis=-1)
        ids = mapping[triangle]
        region = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        world[region][inside] = (w @ p[ids])[inside]
        normal[region][inside] = (w @ n[ids])[inside]
        covered[region] |= inside
        observed[region][inside] = is_face[i]
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-9)
    return world[covered], normal[covered], observed[covered], covered


def zbuffer(projected, depth, faces, width, height):
    result = np.full((height, width), np.inf)
    for face in faces:
        t = projected[face]
        z = depth[face]
        if np.min(z) <= 0:
            continue
        lo = np.maximum(np.floor(t.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(t.max(axis=0)).astype(int), [width - 1, height - 1])
        if np.any(lo > hi):
            continue
        xx, yy = np.meshgrid(
            np.arange(lo[0], hi[0] + 1) + 0.5, np.arange(lo[1], hi[1] + 1) + 0.5
        )
        d = (t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0]) + (t[2, 0] - t[1, 0]) * (
            t[0, 1] - t[2, 1]
        )
        if abs(d) < 1e-9:
            continue
        a = (
            (t[1, 1] - t[2, 1]) * (xx - t[2, 0]) + (t[2, 0] - t[1, 0]) * (yy - t[2, 1])
        ) / d
        b = (
            (t[2, 1] - t[0, 1]) * (xx - t[2, 0]) + (t[0, 0] - t[2, 0]) * (yy - t[2, 1])
        ) / d
        c = 1 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        interpolated = 1 / np.maximum(a / z[0] + b / z[1] + c / z[2], 1e-12)
        tile = result[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        tile[inside] = np.minimum(tile[inside], interpolated[inside])
    return result


def bake_photographs(
    folder, p, f, face_count, rec, frames, train, center, B, transform
):
    size = 1536
    m = trimesh.Trimesh(p, f, process=False)
    normal = np.array(m.vertex_normals)
    atlas = xatlas.Atlas()
    atlas.add_mesh(p.astype(np.float32), f.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = size
    pack.padding = 4
    atlas.generate(pack_options=pack)
    mapping, indices, uv = atlas[0]
    face_keys = {tuple(sorted(face)) for face in f[:face_count]}
    is_face = np.array([tuple(sorted(face)) in face_keys for face in mapping[indices]])
    texel, tn, observed, covered = raster_atlas(
        p, normal, mapping, indices, uv, is_face, size
    )
    world = (texel / transform['scale'] + transform['center']) @ B + center
    world_n = tn @ B
    verts_world = (p / transform['scale'] + transform['center']) @ B + center
    # Neutral shading identifies the inferred cranium instead of inventing a
    # hair/scalp texture for regions the camera never captured.
    light = np.array([-0.3, 0.7, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.6 + 0.4 * np.maximum(tn @ light, 0)
    color = np.array([0.31, 0.38, 0.36])[None] * shade[:, None]
    best = np.zeros(len(texel))
    second = np.zeros(len(texel))
    second_rgb = np.zeros_like(color)
    selected = {
        min(train, key=lambda im: abs(frames[im.name]['yaw'] - angle)).name
        for angle in np.linspace(-32, 32, 9)
    }
    for im in [im for im in train if im.name in selected]:
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        R = pose.rotation.matrix()
        translation = pose.translation
        cp = world @ R.T + translation
        xy = cam.img_from_cam(cp)
        ij = np.floor(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(int)
        pixels = np.asarray(Image.open(folder / 'images' / im.name).convert('RGBA'))
        h, w = pixels.shape[:2]
        inside = (
            (ij[:, 0] >= 0)
            & (ij[:, 0] < w)
            & (ij[:, 1] >= 0)
            & (ij[:, 1] < h)
            & (cp[:, 2] > 0)
            & observed
        )
        ij[:, 0] = np.clip(ij[:, 0], 0, w - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, h - 1)
        x, y = ij.T
        vertex_cp = verts_world @ R.T + translation
        depth = zbuffer(cam.img_from_cam(vertex_cp), vertex_cp[:, 2], f, w, h)
        visible = np.abs(cp[:, 2] - depth[y, x]) * transform['scale'] < 0.004
        toward = im.projection_center() - world
        toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-9)
        facing = np.maximum(np.sum(world_n * toward, axis=1), 0)
        preference = 1.5 if abs(frames[im.name]['yaw']) < 10 else 1.0
        quality = facing**10 * inside * visible * (pixels[y, x, 3] / 255) * preference
        rgb = pixels[y, x, :3] / 255.0
        wins = quality > best
        runner = (~wins) & (quality > second)
        second[wins] = best[wins]
        second_rgb[wins] = color[wins]
        second[runner] = quality[runner]
        second_rgb[runner] = rgb[runner]
        best[wins] = quality[wins]
        color[wins] = rgb[wins]
        print('Projected capture', im.name, flush=True)
    # Blend only compatible runner-up colors; disagreeing expressions and
    # glasses reflections retain one source rather than ghosting together.
    blend = (
        np.minimum(0.25, second / np.maximum(best + second, 1e-9))
        * (np.linalg.norm(color - second_rgb, axis=1) < 0.12)
        * (best > 0.001)
    )
    color = color * (1 - blend[:, None]) + second_rgb * blend[:, None]
    missing = observed & (best < 0.001)
    low = float(np.mean(missing[observed]))
    if low > 0.12:
        raise ValueError(
            f'Photographic coverage is insufficient for {low:.0%} of the face surface.'
        )
    if np.any(missing):
        supported = observed & ~missing
        _, closest = cKDTree(texel[supported]).query(texel[missing])
        color[missing] = color[supported][closest]
    texture = np.zeros((size, size, 3), np.uint8)
    texture[covered] = np.uint8(np.clip(color, 0, 1) * 255)
    _, nearest = distance_transform_edt(~covered, return_indices=True)
    texture[~covered] = texture[nearest[0][~covered], nearest[1][~covered]]
    Image.fromarray(texture[::-1]).save(folder / 'appearance.png')
    metadata = {
        'mapping': mapping.tolist(),
        'indices': indices.ravel().tolist(),
        'uv': uv.ravel().tolist(),
        'texture': f'/api/face-asset?id={folder.name}&asset=appearance.png',
        'stats': {
            'textureSize': size,
            'sourceViews': len(selected),
            'lowConfidenceFraction': low,
            'method': (
                'Captured photographs projected with recovered cameras and mesh '
                'depth visibility. Estimated cranium is neutral gray; no '
                'generated identity texture.'
            ),
        },
    }
    atomic(folder / 'texture-atlas.json', metadata)
    return metadata['stats']


def run(folder, astra=False):
    if (folder / 'model-release.json').exists():
        raise ValueError(
            'This capture uses validated model releases; use build_photo_face.py to rebuild it.'
        )

    started = time.perf_counter()
    old = json.loads((folder / 'status.json').read_text())
    evidence = old.get('evidence', {}).copy()

    def status(stage, message):
        atomic(
            folder / 'status.json',
            {
                'status': 'running',
                'stage': stage,
                'message': message,
                'evidence': evidence,
            },
        )

    try:
        if not (folder / 'poisson-status.json').exists():
            atomic(folder / 'poisson-status.json', old)
        if (folder / 'mesh.json').exists() and not (
            folder / 'poisson-mesh.json'
        ).exists():
            shutil.copy2(folder / 'mesh.json', folder / 'poisson-mesh.json')
        if astra and not (folder / 'astra-review.json').exists():
            status(
                'astra',
                'Astra is reviewing three captured face angles and the reconstruction evidence…',
            )
            from scripts.astra_face_review import run as review_with_astra

            review_with_astra(folder)
        advice = (
            json.loads((folder / 'astra-review.json').read_text())
            if (folder / 'astra-review.json').exists()
            else None
        )
        if 'maxLandmarkSurfaceDistanceMm' in evidence:
            evidence['poissonBaselineMaxLandmarkErrorMm'] = evidence.pop(
                'maxLandmarkSurfaceDistanceMm'
            )
        if advice:
            evidence['astraGuidance'] = {
                'model': advice['model'],
                'normalOffsetLimitMm': advice['gaussianNormalOffsetMm'],
                'estimatedBackDepthToFaceWidth': advice[
                    'estimatedBackDepthToFaceWidth'
                ],
                'unobservedParts': advice['unobservedParts'],
            }
        status(
            'guided-surface',
            'Fitting coherent face depth to the saved camera observations…',
        )
        frames = {
            x['filename']: x
            for x in json.loads((folder / 'capture.json').read_text())['frames']
        }
        rec = pycolmap.Reconstruction(str(folder / 'dataset/sparse/0'))
        ordered = sorted(rec.images.values(), key=lambda im: frames[im.name]['yaw'])
        held = ordered[2::5]
        held_names = {x.name for x in held}
        train = [im for im in ordered if im.name not in held_names]
        points, errors = robust_landmarks(*camera_rays(rec, frames, train))
        test = projection_error(points, rec, frames, held)
        evidence['withheldLandmarks'] = {
            **test,
            'views': len(held),
            'usedForGeometryFit': False,
        }
        if test['medianPx'] > 4 or test['p95Px'] > 12:
            raise ValueError(
                'The fitted surface does not explain withheld camera views. A sharper scan is needed.'
            )
        # Use the original coordinate frame of the trained splat, not a second
        # inconsistent pose normalization for the new mesh.
        lm = {i: triangulate(rec, frames, i)[0] for i in [33, 263, 10, 152, 1, 61, 291]}
        center = (lm[10] + lm[152]) / 2
        right = lm[263] - lm[33]
        right /= np.linalg.norm(right)
        up = lm[10] - lm[152]
        up -= right * np.dot(up, right)
        up /= np.linalg.norm(up)
        B = np.stack([right, up, np.cross(right, up)])
        aligned = (points - center) @ B.T
        mid = (aligned[10] + aligned[152]) / 2
        scale = 0.20 / (aligned[10, 1] - aligned[152, 1])
        transform = {'center': mid.tolist(), 'scale': float(scale)}
        normalized = (aligned - mid) * scale
        p, f, face_count, back_depth = make_surface(
            normalized, advice['estimatedBackDepthToFaceWidth'] if advice else 0.85
        )
        p, max_offset = fit_gaussian_detail(
            folder,
            p,
            f,
            face_count,
            transform,
            advice['gaussianNormalOffsetMm'] if advice else 0.5,
        )
        m = trimesh.Trimesh(p, f, process=False)
        m.fix_normals()
        p = np.asarray(m.vertices)
        f = np.asarray(m.faces)
        nose_depth = float(
            (normalized[1, 2] - (normalized[33, 2] + normalized[263, 2]) / 2) * 1000
        )
        if (
            not m.is_watertight
            or not np.isfinite(p).all()
            or nose_depth < 5
            or nose_depth > 65
        ):
            raise ValueError('The fitted head failed surface or depth validation.')
        evidence.update(
            surfaceMode='landmark-guided',
            noseProjectionFromOuterEyePlaneMm=nose_depth,
            gaussianNormalOffsetMm=max_offset,
            estimatedBackDepthMm=back_depth * 1000,
        )
        stats = {
            **evidence,
            'source': 'Recorded multiview face',
            'sourceSplats': evidence.get('gaussians', 0),
            'vertices': len(p),
            'triangles': len(f),
            'observedFaceTriangles': face_count,
            'estimatedCraniumTriangles': len(f) - face_count,
            'watertight': True,
            'method': (
                'Robust multiview landmark triangulation, connected topology, '
                'anchored Loop subdivision, bounded Gaussian detail and observed '
                'photo projection.'
            ),
            'scale': 'Visible hairline-to-chin height normalized to 20 cm. Metric dimensions are estimated.',
            'limitation': (
                'Observed face is a regularized reconstruction; rear cranium is '
                'an explicit gray modeling prior. Hair, ears, glasses geometry '
                'and hidden recesses are not recovered.'
            ),
            'rig': 'Heuristic surface control fields, not measured anatomy.',
        }
        data = {
            'positions': p.astype(np.float32).ravel().tolist(),
            'normals': np.asarray(m.vertex_normals).astype(np.float32).ravel().tolist(),
            'indices': f.ravel().tolist(),
            'colors': np.tile([0.55, 0.43, 0.36], (len(p), 1)).ravel().tolist(),
            'transform': transform,
            'stats': stats,
        }
        atomic(folder / 'mesh.json', data)
        atomic(
            folder / 'surface-validation.json',
            {
                'withheldFrames': [im.name for im in held],
                'trainingFrames': [im.name for im in train],
                'landmarksWorld': points.tolist(),
                'evidence': evidence,
            },
        )
        status(
            'texture',
            'Projecting your captured photographs onto the fitted 3D surface…',
        )
        texture = bake_photographs(
            folder, p, f, face_count, rec, frames, train, center, B, transform
        )
        data['stats']['appearance'] = texture
        data['stats']['seconds'] = round(time.perf_counter() - started, 2)
        atomic(folder / 'mesh.json', data)
        atomic(
            folder / 'status.json',
            {
                'status': 'complete',
                'stage': 'ready',
                'message': (
                    'Fitted face ready. The face uses captured views; the gray '
                    'rear head is estimated. Inspect Geometry and side views.'
                ),
                'evidence': evidence,
                'id': folder.name,
                'splatAvailable': True,
            },
        )
        print(json.dumps(data['stats'], indent=2))
    except Exception as e:
        atomic(
            folder / 'status.json',
            {
                'status': 'failed',
                'stage': 'failed',
                'message': str(e),
                'evidence': evidence,
                'splatAvailable': (folder / 'face.ply').exists(),
            },
        )
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--astra', action='store_true')
    args = parser.parse_args()
    run(args.folder.resolve(), args.astra)
