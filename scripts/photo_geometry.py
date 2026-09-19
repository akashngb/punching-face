"""Photo-derived face geometry and texture. No radiance-field reconstruction."""

from pathlib import Path
import json
import hashlib
import numpy as np
import trimesh, xatlas
from PIL import Image
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import distance_transform_edt, map_coordinates
from face_pipeline import atomic
from scripts.head_material import rear_reference, missing_head_material
from scripts.head_accessories import clean_view, build_glasses
from scripts.eye_detail import apply_eye_material
from scripts.photo_detail import prepare_detail_frames, detail_image
import cv2

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
    middle = np.array([np.mean(boundary[:, 0]), height * 0.20, 0.0])
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


def interpolate_part_labels(weights, labels):
    """Categorical surface ownership, independent of triangle vertex order.

    Add barycentric support for vertices sharing a part. A mixed ear/skull
    triangle then changes owner along the interpolated boundary rather than
    taking the arbitrary first vertex's label for the entire triangle.
    """
    result = np.zeros(weights.shape[:-1], np.uint8)
    best = np.full(result.shape, -np.inf)
    for label in np.unique(labels):
        score = np.sum(np.where(labels == label, weights, 0), axis=-1)
        take = score > best + 1e-12  # Deterministic lowest-label tie at a boundary.
        result[take] = label
        best[take] = score[take]
    return result


def raster_atlas(
    p, n, mapping, indices, uv, is_face, size, part_labels=None, return_binding=False
):
    world = np.zeros((size, size, 3), np.float32)
    normal = np.zeros_like(world)
    covered = np.zeros((size, size), bool)
    observed = np.zeros_like(covered)
    parts = np.zeros((size, size), np.uint8) if part_labels is not None else None
    owner = np.zeros((size, size), np.int32) if return_binding else None
    bary = np.zeros_like(world) if return_binding else None
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
        if parts is not None:
            parts[region][inside] = interpolate_part_labels(w[inside], part_labels[ids])
        if return_binding:
            owner[region][inside] = i
            bary[region][inside] = w[inside]
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-9)
    result = (world[covered], normal[covered], observed[covered], covered)
    if parts is not None:
        result += (parts[covered],)
    if return_binding:
        result += (
            {
                'triangles': mapping[indices],
                'triangleIds': owner[covered],
                'weights': bary[covered],
            },
        )
    return result


def eye_parts(p, f):
    """Identify independent template eyeballs, without marking lids as eyes."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(p), len(p))
    )
    _, components = connected_components(graph, directed=False)
    labels = np.zeros(len(p), np.uint8)
    for label in np.unique(components[468:]):
        ids = np.where(components == label)[0]
        if not 100 <= len(ids) <= 3000:
            continue
        points = p[ids]
        extent = np.ptp(points, axis=0)
        center = points.mean(0)
        if np.any(extent < 0.015) or np.any(extent > 0.050):
            continue
        for part, corners in [(1, [33, 133]), (2, [263, 362])]:
            if np.linalg.norm(center - p[corners].mean(0)) < 0.030:
                labels[ids] = part
    return labels


def measured_face_footprint(points, cage):
    """Facial coverage excludes template crown/neck beyond the measured oval."""
    polygon = cage[OVAL, :2]
    lo = polygon.min(0)
    extent = polygon.max(0) - lo
    if np.any(extent < 1e-6):
        raise ValueError('Measured face outline is degenerate.')
    mask = np.zeros((1024, 1024), np.uint8)
    cv2.fillPoly(mask, [np.rint((polygon - lo) / extent * 1023).astype(np.int32)], 255)
    xy = np.rint((points[:, :2] - lo) / extent * 1023).astype(np.int32)
    inside = (xy >= 0).all(1) & (xy <= 1023).all(1)
    xy = np.clip(xy, 0, 1023)
    return inside & (mask[xy[:, 1], xy[:, 0]] > 0)


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


def project_eyewear_mask(glasses, cam, pose, center, B, transform, head_depth):
    """Only mask eyewear in front of the head: rear photos remain usable."""
    if glasses is None:
        return None
    h, w = head_depth.shape
    mask = np.zeros((h, w), np.uint8)

    def project(points):
        world = np.asarray(points) / transform['scale'] @ B + center
        cp = world @ pose.rotation.matrix().T + pose.translation
        return cp, cam.img_from_cam(cp)

    # Project only opaque frame paths. Clear lens interiors carry actual eye
    # evidence and must not be replaced by skin or generic eyeball priors.
    paths = [
        *(list(r) + [r[0]] for r in glasses['rims']),
        glasses['bridge'],
        *glasses['temples'],
    ]
    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            samples = (
                np.asarray(a)[None]
                + np.linspace(0, 1, 40)[:, None] * (np.asarray(b) - a)[None]
            )
            cp, xy = project(samples)
            ij = np.rint(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(int)
            inside = (
                (ij[:, 0] >= 0)
                & (ij[:, 0] < w)
                & (ij[:, 1] >= 0)
                & (ij[:, 1] < h)
                & (cp[:, 2] > 0)
            )
            for j in np.where(inside)[0]:
                x, y = ij[j]
                if cp[j, 2] <= head_depth[y, x] + 0.004 / transform['scale']:
                    offset = samples[j] + [glasses['radius'] + 0.002, 0, 0]
                    _, projected = project([offset])
                    radius = max(3, int(np.linalg.norm(projected[0] - xy[j])))
                    cv2.circle(mask, (int(x), int(y)), min(radius, 25), 255, -1)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def prepare_texture_view(
    raw,
    filename,
    completion,
    frame,
    frames,
    reference=None,
    projected_mask=None,
    semantics=None,
):
    """Keep opaque exclusion separate from the continuous generated edit.

    Excluding a whole feather mask would switch cameras at its first nonzero
    pixel, even when the generated contribution there is almost zero.
    """
    pixels, opaque, audit = clean_view(
        raw,
        filename,
        completion,
        frame,
        frames,
        return_details=True,
        projected_mask=projected_mask,
        semantics=semantics,
    )
    reference_mask = None
    if reference:
        from scripts.glasses_reference import composite_reference

        pixels, reference_mask = composite_reference(pixels, reference)
        audit = {
            'method': 'registered-generated-glasses-free-reference',
            'excludedPixels': int(np.count_nonzero(opaque)),
            'estimatedReferencePixels': int(np.count_nonzero(reference_mask)),
            'hiddenSurfaceEstimated': True,
            'preservedEyePhotographs': False,
            'registrationErrorPx': reference['medianRegistrationErrorPx'],
        }
    return pixels, opaque, reference_mask, audit


def blend_estimated_colour(color, total, estimated_color, estimated_total):
    """Continuous confidence transition; the provenance gate is independent."""
    valid = estimated_total > 1e-7
    ratio = np.divide(
        total, estimated_total, out=np.full_like(total, np.inf), where=valid
    )
    weight = np.clip((0.20 - ratio) / 0.12, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    estimate = np.divide(
        estimated_color,
        estimated_total[:, None],
        out=np.zeros_like(estimated_color),
        where=valid[:, None],
    )
    result = color * (1 - weight[:, None]) + estimate * weight[:, None]
    occluded = (total < np.maximum(0.00001, estimated_total * 0.12)) & valid
    return result, occluded


def bake_photographs(
    folder,
    p,
    f,
    face_count,
    rec,
    frames,
    train,
    center,
    B,
    transform,
    completion=None,
    eyes=None,
    semantics=None,
    ear_regions=None,
    output_folder=None,
):
    output_folder = folder if output_folder is None else output_folder
    detail_audit = prepare_detail_frames(folder)
    head_capture = (
        json.loads((folder / 'capture.json').read_text()).get('captureRegion') == 'head'
    )
    import os

    size = int(os.environ.get('CONTACT_TEXTURE_SIZE', '3072'))
    if size not in (1024, 2048, 3072, 4096):
        raise ValueError('Texture size must be 1024, 2048, 3072 or 4096.')
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
    eye_labels = eye_parts(p, f)
    from scripts.ear_fit import (
        ear_vertex_labels,
        source_ear_mask,
        ear_sample_coordinates,
        predicted_ear_mask,
    )
    from scripts.glasses_reference import load_references, surface_coverage

    clean_references = load_references(folder)
    # Photographic visibility can narrow valid color observations, but must
    # not turn an unseen back of an ear into a hair-bearing scalp surface.
    ear_regions = {
        sign: {
            **region,
            'coreVertices': region.get(
                'anatomicalCoreVertices', region['coreVertices']
            ),
        }
        for sign, region in (ear_regions or {}).items()
    }
    labels = ear_vertex_labels(len(p), ear_regions)
    labels[eye_labels > 0] = eye_labels[eye_labels > 0]
    texel, tn, observed, covered, parts, surface_binding = raster_atlas(
        p, normal, mapping, indices, uv, is_face, size, labels, return_binding=True
    )
    from scripts.texture_registration import (
        ear_registration_weights,
        plane_hit_visibility,
    )

    registration = ear_registration_weights(
        p, f, face_count, ear_regions, surface_binding
    )
    corners = p[surface_binding['triangles']]
    facet_normals = np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )
    facet_normals /= np.maximum(
        np.linalg.norm(facet_normals, axis=1, keepdims=True), 1e-12
    )
    facet_world = facet_normals @ B
    bottom = (texel[:, 1] < p[468:, 1].min() + 0.0001) & (tn[:, 1] < -0.4)
    ear_surface = parts >= 3
    # Accelerate is much faster for a small matrix times a tall transpose.
    world = (B.T @ (texel / transform['scale'] + transform['center']).T).T + center
    world_n = (B.T @ tn.T).T
    verts_world = (B.T @ (p / transform['scale'] + transform['center']).T).T + center
    # Neutral shading identifies the inferred cranium instead of inventing a
    # hair/scalp texture for regions the camera never captured.
    light = np.array([-0.3, 0.7, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.6 + 0.4 * np.maximum(tn @ light, 0)
    color = np.array([0.31, 0.38, 0.36])[None] * shade[:, None]
    best = np.zeros(len(texel))
    hair_support = np.zeros(len(texel))
    total = np.zeros(len(texel))
    photo_support = np.zeros(len(texel))
    accum = np.zeros_like(color)
    hair_accum = np.zeros_like(color)
    hair_total = np.zeros(len(texel))
    visible_any = np.zeros(len(texel), bool)
    estimated_total = np.zeros(len(texel))
    estimated_best = np.zeros(len(texel))
    estimated_color = np.zeros_like(color)
    mask_audit = []
    cleaned_total = np.zeros(len(texel))
    cleaned_color = np.zeros_like(color)
    cleaned_coverage = np.zeros(len(texel))
    detail_best = np.zeros(len(texel))
    fine_detail = np.zeros_like(color)
    hair_votes = np.zeros(len(texel))
    hair_visibility = np.zeros(len(texel))
    hair_semantic_best = np.zeros(len(texel))
    ear_hidden_total = np.zeros(len(texel))
    yaw = lambda im: frames[im.name].get('cameraYaw', frames[im.name].get('yaw') or 0)
    facial = [im for im in train if frames[im.name].get('landmarks')]
    front = min(facial, key=lambda im: abs(yaw(im)))
    targets = list(range(-180, 180, 20)) if head_capture else [-45, 0, 45]
    from scripts.frame_evidence import assess_frames, choose_views

    view_quality = assess_frames(folder, output_folder=output_folder)
    selected = choose_views(train, frames, targets, view_quality)
    selected.add(front.name)
    if completion:
        # Hair evidence must reach the bake even when there is no eyewear.
        hair_views = {
            v['filename'] for v in completion.get('views', []) if v.get('hairRegions')
        }
        selected |= {im.name for im in train if im.name in hair_views}
    if completion and completion['glasses']['present']:
        annotated = {v['filename'] for v in completion['views']}
        selected |= {im.name for im in train if im.name in annotated}
        front = next(
            (im for im in train if im.name == completion['frontFilename']), front
        )
    # One frontal exposure owns the eyes, nose and mouth. Side cameras enter
    # through broad smooth weights, never a per-texel winner that cuts a face
    # into mismatched photographic fragments.
    width = np.ptp(p[:468, 0])
    side_mix = np.clip((np.abs(texel[:, 0]) - width * 0.25) / (width * 0.22), 0, 1)
    side_mix = side_mix * side_mix * (3 - 2 * side_mix)

    def smooth_region(v):
        v = np.clip(v, 0, 1)
        return v * v * (3 - 2 * v)

    frontal = (
        smooth_region((texel[:, 2] + 0.10) / 0.070)
        * smooth_region((texel[:, 1] - p[152, 1] + 0.018) / 0.020)
        * smooth_region((p[10, 1] + 0.025 - texel[:, 1]) / 0.025)
    )
    central = frontal * (1 - side_mix)
    from scripts.view_blending import frontal_preference

    toward_front = front.projection_center() - world
    toward_front /= np.maximum(
        np.linalg.norm(toward_front, axis=1, keepdims=True), 1e-9
    )
    front_facing = np.clip(np.sum(world_n * toward_front, axis=1), 0, 1)
    # Preserve the registered eyes, brows, nose and lips. Release frontal
    # ownership smoothly below the measured outer lower lip toward the chin.
    lower_lip_y = p[17, 1]
    jaw_span = max(lower_lip_y - p[152, 1], 0.005)
    jaw_release = smooth_region((lower_lip_y - texel[:, 1]) / (jaw_span * 0.65)) * (
        parts == 0
    )
    front_preference = frontal_preference(central, front_facing, True, jaw_release)
    side_preference = frontal_preference(central, front_facing, False, jaw_release)

    from scripts.texture_evidence import photographed_cheek_color

    cheek_cache = {}

    def cheek_color(im):
        if im.name not in cheek_cache:
            landmarks = frames[im.name].get('landmarks')
            rgba = np.asarray(Image.open(folder / 'images' / im.name).convert('RGBA'))
            cheek_cache[im.name] = (
                photographed_cheek_color(rgba, landmarks) if landmarks else None
            )
        return cheek_cache[im.name]

    reference_frame = front
    reference_color = cheek_color(front)
    if reference_color is None:
        for candidate in sorted(facial, key=lambda im: abs(yaw(im))):
            reference_color = cheek_color(candidate)
            if reference_color is not None:
                reference_frame = candidate
                break
        if reference_color is None:
            raise ValueError(
                'No photograph has sufficient unmasked cheek coverage to establish skin color.'
            )
    glasses = (
        build_glasses(p, completion, rec, frames, center, B, transform)
        if completion
        else None
    )
    # Each camera computes private contributions; reduce in camera order to
    # retain floating-point sums and fine-detail tie breaking exactly.
    from concurrent.futures import ThreadPoolExecutor
    import os

    observed_all = observed
    view_fields = (parts, observed, bottom, ear_surface, side_mix)
    sums = dict(
        accum=accum,
        hair_accum=hair_accum,
        hair_total=hair_total,
        total=total,
        photo_support=photo_support,
        estimated_total=estimated_total,
        estimated_color=estimated_color,
        cleaned_total=cleaned_total,
        cleaned_color=cleaned_color,
        hair_votes=hair_votes,
        hair_visibility=hair_visibility,
        ear_hidden_total=ear_hidden_total,
    )
    maximums = dict(
        best=best,
        hair_support=hair_support,
        hair_semantic_best=hair_semantic_best,
        estimated_best=estimated_best,
        cleaned_coverage=cleaned_coverage,
    )

    def project_view(im, sample_indices=None):
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        R = pose.rotation.matrix()
        translation = pose.translation
        view_world = world if sample_indices is None else world[sample_indices]
        view_normals = world_n if sample_indices is None else world_n[sample_indices]
        view_observed = (
            observed_all if sample_indices is None else observed_all[sample_indices]
        )
        cp = (R @ view_world.T).T + translation
        toward = im.projection_center() - view_world
        toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-9)
        facing = np.maximum(np.sum(view_normals * toward, axis=1), 0)
        # All contribution/ownership scores are zero behind the camera or
        # on back-facing surfaces. Keep positive grazing angles for cleanup.
        near = np.flatnonzero(
            (facing > 0) & (cp[:, 2] > 0) & (view_observed | head_capture)
        )
        cp, facing = cp[near], facing[near]
        if sample_indices is not None:
            near = sample_indices[near]
        parts, observed, bottom, ear_surface, side_mix = (
            field[near] for field in view_fields
        )
        preference = (front_preference if im.name == front.name else side_preference)[
            near
        ]
        contributions = {
            name: np.zeros((len(near), *value.shape[1:]), dtype=value.dtype)
            for name, value in sums.items()
        }
        maxima = {
            name: np.zeros((len(near), *value.shape[1:]), dtype=value.dtype)
            for name, value in maximums.items()
        }
        accum = contributions['accum']
        total = contributions['total']
        photo_support = contributions['photo_support']
        estimated_total = contributions['estimated_total']
        estimated_color = contributions['estimated_color']
        cleaned_total = contributions['cleaned_total']
        cleaned_color = contributions['cleaned_color']
        hair_votes = contributions['hair_votes']
        hair_visibility = contributions['hair_visibility']
        ear_hidden_total = contributions['ear_hidden_total']
        best = maxima['best']
        estimated_best = maxima['estimated_best']
        cleaned_coverage = maxima['cleaned_coverage']
        detail_best = np.zeros(len(near))
        fine_detail = np.zeros((len(near), 3))
        visible_any = np.zeros(len(near), bool)
        xy = cam.img_from_cam(cp)
        ij = np.floor(np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1)).astype(int)
        h, w = cam.height, cam.width
        raw = detail_image(folder, im.name)
        pixel_scale = raw.shape[1] / w
        vertex_cp = (R @ verts_world.T).T + translation
        depth = zbuffer(cam.img_from_cam(vertex_cp), vertex_cp[:, 2], f, w, h)
        projected_mask = project_eyewear_mask(
            glasses, cam, pose, center, B, transform, depth
        )
        detail_completion = completion
        if pixel_scale != 1 and completion:
            detail_completion = {
                **completion,
                'crops': {
                    name: (np.asarray(crop) * pixel_scale).tolist()
                    for name, crop in completion['crops'].items()
                },
            }
        if projected_mask is not None:
            projected_mask = cv2.resize(
                projected_mask,
                (raw.shape[1], raw.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        clean_reference = clean_references.get(im.name)
        pixels, accessory_mask, reference_mask, audit = prepare_texture_view(
            raw,
            im.name,
            detail_completion,
            frames[im.name],
            frames,
            clean_reference,
            projected_mask,
            semantics,
        )
        opaque_mask = accessory_mask
        audit = {'filename': im.name, **audit}
        inside = (
            (ij[:, 0] >= 0)
            & (ij[:, 0] < w - 1)
            & (ij[:, 1] >= 0)
            & (ij[:, 1] < h - 1)
            & (cp[:, 2] > 0)
            & (observed | head_capture)
        )
        ij[:, 0] = np.clip(ij[:, 0], 0, w - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, h - 1)
        x, y = ij.T
        sample_projection = (
            ear_sample_coordinates(
                xy,
                parts,
                p,
                ear_regions,
                semantics,
                im,
                cam,
                center,
                B,
                transform,
                pixel_scale,
                {sign: weight[near] for sign, weight in registration.items()},
            )
            if semantics and ear_regions
            else xy
        )
        sample_ij = np.floor(
            np.nan_to_num(sample_projection, nan=-1, posinf=-1, neginf=-1)
        ).astype(int)
        sample_inside = (
            (sample_ij[:, 0] >= 0)
            & (sample_ij[:, 0] < w - 1)
            & (sample_ij[:, 1] >= 0)
            & (sample_ij[:, 1] < h - 1)
        )
        sx, sy = np.clip(sample_ij[:, 0], 0, w - 1), np.clip(sample_ij[:, 1], 0, h - 1)
        visible = np.abs(cp[:, 2] - depth[y, x]) * transform['scale'] < 0.006
        # Test the same pixel-center ray as the depth rasterizer, with the
        # actual triangle plane rather than interpolated shading normals.
        # Keep the accepted central-face sampling unchanged.
        posterior = ear_surface | (
            (~observed) & (texel[near, 2] < -0.12) & (texel[near, 1] < p[10, 1] + 0.015)
        )
        ray_xy = cam.cam_from_img(ij[posterior] + 0.5)
        rays = np.column_stack([ray_xy, np.ones(len(ray_xy))])
        camera_facets = (
            facet_world[surface_binding['triangleIds'][near[posterior]]] @ R.T
        )
        first_hit = plane_hit_visibility(
            cp[posterior],
            camera_facets,
            rays,
            depth[y[posterior], x[posterior]],
            transform['scale'],
        )
        audit['occludedPosteriorSamplesRejected'] = int(
            np.count_nonzero(visible[posterior] & ~first_hit & inside[posterior])
        )
        visible[posterior] = first_hit
        audit['registeredPosteriorSamples'] = int(
            np.count_nonzero(np.linalg.norm(sample_projection - xy, axis=1) > 0.01)
        )
        visible_any |= inside & visible & (facing > 0.05)
        # Segmentation leaves a pale fringe at some neck/hair cutouts. Fade
        # projections before the cutout boundary so it cannot become a sharp
        # diagonal stripe when another view or the inferred material takes over.
        base_alpha = cv2.resize(
            pixels[:, :, 3], (w, h), interpolation=cv2.INTER_NEAREST
        )
        boundary_distance = distance_transform_edt(base_alpha > 128)
        edge_weight = np.clip(boundary_distance[sy, sx] / 10.0, 0, 1)
        edge_weight = edge_weight * edge_weight * (3 - 2 * edge_weight)
        quality = (
            facing**8
            * inside
            * sample_inside
            * visible
            * (base_alpha[sy, sx] / 255)
            * edge_weight
        )
        hair_view = next(
            (
                v
                for v in (completion or {}).get('views', [])
                if v['filename'] == im.name
            ),
            None,
        )
        quality[bottom] = 0
        ear_pixels = None
        hair_interior = np.zeros(len(near))
        erased_ear = np.zeros(len(near), bool)
        ear_mask = np.zeros((h, w), np.uint8)
        ear_occluder = np.zeros((h, w), bool)
        if semantics and ear_regions:
            local_origin = (im.projection_center() - center) @ B.T
            ear_mask, annotated = source_ear_mask(
                (h, w), im.name, semantics, local_origin, pixel_scale
            )
            if not annotated:
                ear_mask = predicted_ear_mask(
                    (h, w), ear_regions, im, cam, center, B, transform
                )
            source_part = ear_mask[sy, sx]
            # The ear outline is an approximate annotation. Its antialiased
            # rim must not become a second pink ear painted onto the scalp.
            # Keep sampling the fitted ear itself from the original outline.
            ear_occluder = (
                cv2.dilate(np.uint8(ear_mask > 0), np.ones((7, 7), np.uint8)) > 0
            )
            # A photographed ear can only land on its fitted ear surface.
            # Conversely an ear receives no cheek/hair from another view.
            quality[ear_surface & (source_part != parts)] = 0
            erased_ear = ear_occluder[sy, sx] & ~ear_surface
            if ear_mask.any():
                ear_pixels = pixels
                pixels = pixels.copy()
                native_mask = cv2.resize(
                    np.uint8(ear_occluder) * 255,
                    (pixels.shape[1], pixels.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                fill = pixels[:, :, :3].copy()
                valid = pixels[:, :, 3] > 128
                if valid.any():
                    _, nearest = distance_transform_edt(~valid, return_indices=True)
                    fill[~valid] = fill[nearest[0][~valid], nearest[1][~valid]]
                pixels[:, :, :3] = cv2.inpaint(fill, native_mask, 5, cv2.INPAINT_TELEA)
        if hair_view:
            from scripts.hair_appearance import (
                semantic_view_support,
                skin_semantic_support,
            )

            hair_mask = np.zeros((h, w), np.uint8)
            crop = np.array(completion['crops'][im.name])
            for poly in hair_view.get('hairRegions', []):
                if len(poly) >= 3:
                    cv2.fillPoly(
                        hair_mask,
                        [
                            np.rint(
                                np.asarray(poly) * (crop[2:] - crop[:2]) + crop[:2]
                            ).astype(np.int32)
                        ],
                        255,
                    )
            # An ear occludes the skull. Its skin is not evidence that the
            # underlying scalp is bald, even in a perfectly registered image.
            source_alpha = cv2.resize(
                raw[:, :, 3], (w, h), interpolation=cv2.INTER_NEAREST
            )
            skin_semantic = skin_semantic_support(
                hair_mask > 0,
                source_alpha > 200,
                max(2.0, float(np.min(crop[2:] - crop[:2])) * 0.02),
            )
            class_support = np.where(hair_mask[sy, sx] > 0, 1, skin_semantic[sy, sx])
            evidence = (
                semantic_view_support(facing, class_support)
                * inside
                * sample_inside
                * visible
                * (source_alpha[sy, sx] > 200)
                * edge_weight
                * (~ear_occluder[sy, sx])
            )
            # Opaque frames and restored alpha holes are not observations of
            # the underlying class. Source annotations outside the occluder
            # remain valid even if an edited reference covers a broader area.
            semantic_occluder = cv2.resize(
                accessory_mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            evidence *= semantic_occluder[sy, sx] == 0
            hair_votes += evidence * (hair_mask[sy, sx] > 0)
            hair_visibility += evidence
            maxima['hair_semantic_best'] = evidence
            # Only extend oblique support inside an annotated hair region.
            # Near the nape, a silhouette/template mismatch can otherwise
            # project bright neck skin into the hair as support is relaxed.
            from scripts.hair_appearance import annotated_hair_support

            hair_rgb = (
                cv2.resize(raw[:, :, :3], (w, h), interpolation=cv2.INTER_AREA) / 255.0
            )
            hair_region = annotated_hair_support(
                hair_rgb, (hair_mask > 0) & (base_alpha > 200)
            )
            hair_interior = hair_region[sy, sx]
        sample_xy = (
            np.nan_to_num(sample_projection, nan=-1, posinf=-1, neginf=-1) * pixel_scale
        )
        rgb = (
            np.stack(
                [
                    map_coordinates(
                        pixels[:, :, c].astype(float),
                        [sample_xy[:, 1] - 0.5, sample_xy[:, 0] - 0.5],
                        order=1,
                        mode='nearest',
                    )
                    for c in range(3)
                ],
                axis=1,
            )
            / 255.0
        )
        if ear_pixels is not None:
            rgb[ear_surface] = (
                np.stack(
                    [
                        map_coordinates(
                            ear_pixels[:, :, c].astype(float),
                            [
                                sample_xy[ear_surface, 1] - 0.5,
                                sample_xy[ear_surface, 0] - 0.5,
                            ],
                            order=1,
                            mode='nearest',
                        )
                        for c in range(3)
                    ],
                    axis=1,
                )
                / 255.0
            )
        # Blend exposure at low frequencies; retain fine hairs from one clear
        # camera instead of averaging misaligned eyebrow/beard/lock edges.
        smooth = cv2.GaussianBlur(
            pixels[:, :, :3].astype(np.float32) / 255, (0, 0), 1.4 * pixel_scale
        )
        low_rgb = np.stack(
            [
                map_coordinates(
                    smooth[:, :, c],
                    [sample_xy[:, 1] - 0.5, sample_xy[:, 0] - 0.5],
                    order=1,
                    mode='nearest',
                )
                for c in range(3)
            ],
            axis=1,
        )
        if ear_pixels is not None:
            ear_smooth = cv2.GaussianBlur(
                ear_pixels[:, :, :3].astype(np.float32) / 255, (0, 0), 1.4 * pixel_scale
            )
            low_rgb[ear_surface] = np.stack(
                [
                    map_coordinates(
                        ear_smooth[:, :, c],
                        [
                            sample_xy[ear_surface, 1] - 0.5,
                            sample_xy[ear_surface, 0] - 0.5,
                        ],
                        order=1,
                        mode='nearest',
                    )
                    for c in range(3)
                ],
                axis=1,
            )
        # Reject the bright cutout fringe without removing legitimate skin.
        fringe = (rgb.min(axis=1) > 0.76) & (np.ptp(rgb, axis=1) < 0.10)
        quality *= ~((~observed) & fringe)
        median = cheek_color(im)
        if median is not None:
            correction = np.clip(reference_color / np.maximum(median, 0.05), 0.8, 1.25)
            rgb *= correction
            low_rgb *= correction
        if clean_reference:
            # A verified cleanup owns its accessory region in surface space.
            # Other camera angles can otherwise reintroduce an old rim just
            # outside their approximate 2D masks. Keep the feathered boundary
            # and require actual visibility rather than deleting adjacent skin.
            feather = cv2.distanceTransform(reference_mask, cv2.DIST_L2, 5)
            feather = np.clip(feather / 25.0, 0, 1)
            feather = feather * feather * (3 - 2 * feather)
            alpha = map_coordinates(
                feather,
                [sample_xy[:, 1] - 0.5, sample_xy[:, 0] - 0.5],
                order=1,
                mode='constant',
                cval=0,
            )
            strength = quality * preference * alpha * (~ear_surface)
            cleaned_total += strength
            cleaned_color += rgb * strength[:, None]
            usable = (
                inside
                * sample_inside
                * visible
                * (base_alpha[sy, sx] / 255)
                * edge_weight
                * (~ear_surface)
                * (~bottom)
            )
            # Stronger ownership at oblique angles is needed for temple arms,
            # not entire lens regions or central features. Preserve frontal
            # ownership of the brow/forehead: a profile rim projection can
            # otherwise put a second eyebrow above the measured one.
            opaque = cv2.GaussianBlur(
                opaque_mask.astype(np.float32) / 255, (0, 0), 2 * pixel_scale
            )
            opaque_alpha = map_coordinates(
                opaque,
                [sample_xy[:, 1] - 0.5, sample_xy[:, 0] - 0.5],
                order=1,
                mode='constant',
                cval=0,
            )
            coverage = np.maximum(
                alpha * smooth_region(quality * preference / 0.02) * (~ear_surface),
                surface_coverage(alpha, facing, usable, side_mix) * opaque_alpha,
            )
            cleaned_coverage = np.maximum(cleaned_coverage, coverage)
        masked = (
            cv2.resize(accessory_mask, (w, h), interpolation=cv2.INTER_NEAREST)[sy, sx]
            > 0
        ) | erased_ear
        # The hidden skull has no photographic evidence in these pixels.
        # A 2D ear inpaint can pull dark ear shadows/hair across that missing
        # surface and wrongly suppress 3D completion as a high-confidence
        # estimate. Keep its occlusion audit but give it no color vote.
        ear_hidden_total += quality * preference * erased_ear
        quality[erased_ear] = 0
        # Clean unoccluded photographs always win. Only use estimated fills
        # when no camera can see the skin behind the accessory.
        # The small opaque frame fill replaces only those pixels. Lenses,
        # eyes and brows retain their observed photo appearance.
        # Preference chooses a coherent exposure; it cannot create evidence.
        # All masks above act on the physical score before preference is applied.
        estimate = quality * preference * masked
        estimated_best = np.maximum(estimated_best, quality * masked)
        estimated_total += estimate
        estimated_color += rgb * estimate[:, None]
        quality *= ~masked
        from scripts.hair_appearance import hair_photo_support

        # Preserve usable oblique rear photographs. The sharper facing**8
        # preference still controls their relative contribution to the bake.
        maxima['hair_support'] = hair_photo_support(quality, facing, hair_interior)
        # Relaxed oblique support is appropriate for registered hair, but a
        # large corrective image warp is not equally reliable on the scalp
        # beside the ear. Fade only that extra support with displacement; a
        # clear, front-facing observation retains its original confidence.
        displacement = np.linalg.norm(sample_projection - xy, axis=1)
        alignment = np.exp(-((displacement / max(w * 0.01, 1.0)) ** 2))
        if registration:
            # Curved, partially hidden ear/scalp junctions need a clear view
            # before oblique hair can suppress missing-material completion.
            junction = np.maximum.reduce(
                [weight[near] for weight in registration.values()]
            )
            alignment *= (1 - junction) ** 2
        maxima['hair_support'] = (
            quality + (maxima['hair_support'] - quality) * alignment
        )
        blend_weight = quality * preference
        from scripts.texture_registration import posterior_view_weight

        camera_origin = (
            (im.projection_center() - center) @ B.T - transform['center']
        ) * transform['scale']
        blend_weight = (
            posterior_view_weight(
                texel[near],
                facing,
                quality,
                camera_origin,
                (~observed) & (parts == 0) & (~bottom) & (texel[near, 1] < p[10, 1]),
            )
            * preference
        )
        hair_blend = blend_weight * hair_interior
        contributions['hair_accum'] += rgb * hair_blend[:, None]
        contributions['hair_total'] += hair_blend
        take = blend_weight > detail_best
        fine_detail[take] = (rgb - low_rgb)[take]
        detail_best = np.maximum(detail_best, blend_weight)
        best = np.maximum(best, quality)
        photo_support += quality
        total += blend_weight
        accum += low_rgb * blend_weight[:, None]
        maxima.update(
            best=best, estimated_best=estimated_best, cleaned_coverage=cleaned_coverage
        )
        return near, contributions, maxima, detail_best, fine_detail, visible_any, audit

    views = [im for im in train if im.name in selected]
    workers = max(1, min(4, int(os.environ.get('CONTACT_BAKE_WORKERS', '4'))))
    from scripts.camera_texture import CameraTexture

    ownership = (
        CameraTexture(p, f, texel, parts, labels, surface_binding, views, completion)
        if head_capture
        else None
    )
    with ThreadPoolExecutor(workers, thread_name_prefix='photo-bake') as pool:
        # Batches bound both in-flight work and retained per-camera arrays.
        for start in range(0, len(views), workers):
            batch = views[start : start + workers]
            for im, result in zip(batch, pool.map(project_view, batch)):
                near, contributions, maxima, detail, fine, visible, audit = result
                if ownership is not None:
                    ownership.observe(im, near, contributions['total'])
                for name, values in contributions.items():
                    sums[name][near] += values
                for name, values in maxima.items():
                    maximums[name][near] = np.maximum(maximums[name][near], values)
                take = detail > detail_best[near]
                fine_detail[near[take]] = fine[take]
                detail_best[near] = np.maximum(detail_best[near], detail)
                visible_any[near] |= visible
                mask_audit.append(audit)
                print('Projected capture', im.name, flush=True)
    supported = total > 1e-7
    color[supported] = (
        accum[supported] / total[supported, None] + fine_detail[supported]
    )
    ownership_audit = None
    if ownership is not None:
        revised_ids, revised_low, ownership_audit = ownership.reblend(
            views,
            project_view,
            workers,
            best,
            cleaned_coverage,
            hair_votes,
            hair_visibility,
            hair_semantic_best,
        )
        color[revised_ids] = revised_low + fine_detail[revised_ids]
    # A tiny grazing or unregistered view is not better evidence than a clean
    # estimate in the owning frontal exposure. Previously any nonzero sample
    # could reintroduce a second rim from a weaker side photograph.
    color, occluded = blend_estimated_colour(
        color, total, estimated_color, estimated_total
    )
    cleaned = cleaned_total > 1e-7
    lighting_matched = 0
    if cleaned.any():
        clean_color = np.divide(
            cleaned_color,
            cleaned_total[:, None],
            out=np.zeros_like(cleaned_color),
            where=cleaned[:, None],
        )
        from scripts.cleanup_lighting import match_cleanup_lighting

        clean_color, lighting_matched = match_cleanup_lighting(
            texel,
            tn,
            parts,
            color,
            clean_color,
            cleaned_coverage,
            best,
            cleaned,
            side_mix,
        )
        color[cleaned] = (
            color[cleaned] * (1 - cleaned_coverage[cleaned, None])
            + clean_color[cleaned] * cleaned_coverage[cleaned, None]
        )
    supported_color_reference = None
    if head_capture:
        from scripts.texture_evidence import snapshot_supported_colors

        supported_color_reference = snapshot_supported_colors(
            color, parts, best, cleaned_coverage, bottom
        )
    rear_path = folder / 'rear-prediction/rear.png'
    inferred = ~observed & (best < 0.08)
    rear = rear_reference(rear_path) if head_capture and rear_path.exists() else None
    if head_capture:
        # Use real photographed hair for missing-crown material, even when no
        # generated rear reference exists. This is texture synthesis, not an
        # additional measured viewpoint.
        frame = frames[front.name]
        photo = Image.open(folder / 'images' / front.name).convert('RGBA')
        box = photo.getchannel('A').getbbox()
        swatch = None
        if frame.get('landmarks') and box:
            forehead = frame['landmarks'][10]['y'] * photo.height
            hair_h = forehead - box[1]
            mid = (box[0] + box[2]) * 0.5
            span = (box[2] - box[0]) * 0.23
            if hair_h > 12:
                patch = np.asarray(
                    photo.crop(
                        (
                            int(mid - span),
                            int(box[1] + hair_h * 0.08),
                            int(mid + span),
                            int(box[1] + hair_h * 0.70),
                        )
                    )
                )
                rgb = patch[:, :, :3] / 255.0
                valid = (patch[:, :, 3] > 200) & (rgb.mean(2) < 0.45)
                if valid.any():
                    _, near = distance_transform_edt(~valid, return_indices=True)
                    rgb[~valid] = rgb[near[0][~valid], near[1][~valid]]
                    swatch = rgb
        from scripts.head_material import photographed_scalp_region
        from scripts.rear_hair import rear_hair_swatches

        short_swatches, short_hair_audit = rear_hair_swatches(
            folder, completion, frames, semantics
        )

        # Semantic visibility is independent of the sharper RGB-selection
        # weight. Otherwise the prior paints hair onto visible temple skin.
        scalp_override = photographed_scalp_region(
            texel, p, completion, hair_votes, hair_visibility, hair_semantic_best
        )
        # Use physical photo support for the scalp; facial and ear ownership
        # retain their existing, stricter thresholds.
        scalp_support = np.where((~observed) & (parts == 0), hair_support, best)
        scalp_override[ear_surface] = 0
        inferred_rgb, scalp = missing_head_material(
            texel,
            tn,
            p,
            completion,
            reference_color,
            rear,
            swatch,
            scalp_override,
            short_swatches,
        )
        # Discard grazing-angle projections smoothly: they carry little usable
        # texture resolution and otherwise smear the side and crown.
        weight = np.clip((0.12 - scalp_support) / 0.12, 0, 1) * (1 - frontal)
        weight = weight * weight * (3 - 2 * weight)
        original_weight = np.clip((0.12 - best) / 0.12, 0, 1) * (1 - frontal)
        original_weight = original_weight * original_weight * (3 - 2 * original_weight)
        weight *= 1 - smooth_region(estimated_best / 0.10)
        weight *= 1 - cleaned_coverage
        original_weight *= 1 - smooth_region(estimated_best / 0.10)
        original_weight *= 1 - cleaned_coverage
        # Ear observations already fade by physical support above. A tiny
        # nonzero projection must not disable completion and retain black
        # or stretched pixels on unseen folds.
        # New support must use color from the SAME hair observations. The
        # unrestricted blend may contain neck skin from another camera.
        from scripts.hair_appearance import blend_hair_completion

        completed = blend_hair_completion(
            color, inferred_rgb, hair_accum, hair_total, original_weight, weight
        )
        # Retain this fallback's contribution separately. Lower-skin harmonic
        # completion can then replace it once, rather than blending through a
        # bright cheek swatch a second time at intermediate confidence.
        skin_completion_delta = completed - color
        color = color + skin_completion_delta
        from scripts.ear_appearance import hidden_scalp_completion

        hidden_hair = hidden_scalp_completion(
            ear_hidden_total,
            total,
            estimated_total,
            scalp,
            parts,
            np.maximum(best, hair_support),
        )
        color = color * (1 - hidden_hair[:, None]) + inferred_rgb * hidden_hair[:, None]
        skin_completion_delta *= 1 - hidden_hair[:, None]
    # A complete template contains hidden mouth surfaces and the backs of the
    # eyeballs. Those are not failed facial photographs. Measure coverage on
    # the exposed surface and give hidden internal anatomy a neutral material.
    exposed = observed & visible_any & (parts == 0) & measured_face_footprint(texel, p)
    interior = observed & ~visible_any & (parts == 0)
    color[interior] = reference_color * 0.65
    if head_capture:
        skin_completion_delta[interior] = 0
    mouth = (
        interior
        & (texel[:, 1] < p[13, 1] + 0.006)
        & (texel[:, 1] > p[152, 1] + 0.01)
        & (np.abs(texel[:, 0]) < 0.045)
    )
    color[mouth] = np.array([0.14, 0.055, 0.045])
    missing = exposed & (photo_support < 0.00001) & ~occluded
    low = float(np.mean(missing[exposed]))
    if low > 0.12:
        np.savez_compressed(
            output_folder / 'texture-diagnostic.npz',
            texel=texel,
            observed=observed,
            missing=missing,
            total=total,
        )
    if low > 0.12:
        raise ValueError(
            f'Projection support is insufficient for {low:.0%} of the measured facial outline.'
        )
    if np.any(missing):
        supported = exposed & ~missing
        _, closest = cKDTree(texel[supported]).query(texel[missing])
        color[missing] = color[supported][closest]
        if head_capture:
            skin_completion_delta[missing] = 0
    lower_skin_texels = 0
    ear_skin_texels = 0
    lower_surface = None
    if head_capture:
        from scripts.skin_continuation import continue_lower_skin, continue_ear_skin

        color, ear_skin_texels = continue_ear_skin(
            texel,
            color,
            best,
            parts,
            tn,
            vertices=p,
            faces=f,
            binding=surface_binding,
            regions=ear_regions,
        )
        confidence = np.maximum(best, np.minimum(estimated_best, 1.0))
        color, lower_skin_texels, lower_surface = continue_lower_skin(
            texel,
            color,
            confidence,
            p,
            parts,
            scalp,
            f,
            surface_binding,
            source_confidence=best,
            preserve=mouth | bottom,
            pre_completion_color=color - skin_completion_delta,
        )
    eye_texels = 0
    socket_shading = None
    if eyes:
        color, eye_texels = apply_eye_material(
            folder, texel, np.where(parts < 3, parts, 0), p, eye_labels, color, eyes
        )
        from scripts.eye_shading import apply_estimated_eye_shading

        estimated_parts = [
            spec['part'] for spec in eyes['eyes'].values() if not spec['photoUsable']
        ]
        color, socket_shading = apply_estimated_eye_shading(
            p, f, eye_labels, texel, parts, color, estimated_parts
        )
    estimated_eye = ((parts > 0) & (parts < 3)) if eyes else np.zeros(len(texel), bool)
    # A closed neck section is a presentation surface, never a photographed
    # anatomical underside. Prevent beard, ears and cutout edges on this cap.
    color[bottom] = reference_color * 0.78
    supported_color_audit = None
    if supported_color_reference is not None:
        from scripts.texture_evidence import (
            SupportedColorRegression,
            validate_supported_colors,
        )

        try:
            supported_color_audit = validate_supported_colors(
                supported_color_reference, color, exclude=mouth | interior
            )
        except SupportedColorRegression as error:
            diagnostic = error.diagnostic
            for sample in diagnostic['samples']:
                index = sample['texelId']
                sample['position'] = texel[index].tolist()
                sample['physicalSupport'] = float(best[index])
            atomic(output_folder / 'supported-color-regression.json', diagnostic)
            raise
    texture = np.zeros((size, size, 3), np.uint8)
    if os.environ.get('CONTACT_TEXTURE_DIAGNOSTICS') == '1':
        np.savez_compressed(
            folder / 'texture-evidence.npz',
            positions=texel,
            normals=tn,
            parts=parts,
            observed=observed,
            color=color,
            confidence=best,
            hairSupport=hair_support,
            scalp=scalp if head_capture else np.zeros(len(texel)),
            estimated=estimated_best,
            earOccluded=ear_hidden_total,
            total=total,
            triangles=surface_binding['triangles'],
            triangleIds=surface_binding['triangleIds'],
            weights=surface_binding['weights'],
        )
    texture[covered] = np.uint8(np.clip(color, 0, 1) * 255)
    _, nearest = distance_transform_edt(~covered, return_indices=True)
    texture[~covered] = texture[nearest[0][~covered], nearest[1][~covered]]
    Image.fromarray(texture[::-1]).save(output_folder / 'appearance.png')
    if eyes:
        rough = np.full((size, size), 199, np.uint8)
        rough[covered] = np.where(estimated_eye, 51, 199)
        rough[~covered] = rough[nearest[0][~covered], nearest[1][~covered]]
        Image.fromarray(rough[::-1]).convert('RGB').save(
            output_folder / 'appearance-roughness.png'
        )
    # An explicit audit prevents detection alone being reported as successful
    # cleanup. Retain which views were masked, propagated or excluded.
    atomic(
        output_folder / 'eyewear-mask-audit.json',
        {
            'views': mask_audit,
            'estimatedFaceTexels': int(np.count_nonzero(occluded & exposed)),
            'estimatedFaceTexelsScope': 'Opaque-occlusion fallback only; does not count all generated-reference appearance.',
            'cleanupOverrideFaceTexels': int(
                np.count_nonzero((cleaned_coverage > 0.5) & exposed)
            ),
            'estimatedEyeTexels': int(estimated_eye.sum()),
            'lensInteriorsExcluded': bool(clean_references),
            'opaqueFramePixelsReplaced': True,
            'eyeDetail': (
                eyes['summary'] if eyes else 'Unverified photographic projection'
            ),
            'limitation': (
                (
                    'Glasses-affected skin uses a registered generated estimate; '
                    'a real glasses-free reference is needed to verify hidden '
                    'skin.'
                )
                if clean_references
                else (
                    'Skin through clear lenses is retained, so lens '
                    'tint/reflections may remain. Eyeballs use the separately '
                    'audited eye material.'
                )
            ),
        },
    )
    metadata = {
        'mapping': mapping.tolist(),
        'indices': indices.ravel().tolist(),
        'uv': uv.ravel().tolist(),
        'texture': f'/api/face-asset?id={folder.name}&asset=appearance.png',
        'stats': {
            'textureSize': size,
            'sourceViews': len(selected),
            'lowConfidenceFraction': low,
            'hairTextureFromPhotos': head_capture,
            'shortHairCompletion': short_hair_audit if head_capture else None,
            'rearAppearance': (
                'Captured rear photographs with inferred gaps'
                if any(abs(yaw(im)) > 115 for im in train)
                else (
                    'AI-predicted rear reference'
                    if rear_path.exists()
                    else (
                        'Photographic material continuation'
                        if head_capture
                        else 'Unobserved gray'
                    )
                )
            ),
            'photographedCapFraction': float(np.mean(best[~observed] > 0.001)),
            'usablePhotographedCapFraction': float(
                np.mean(hair_support[~observed] >= 0.12)
            ),
            'hairCompletionSupport': 'Visibility and projected resolution, independent of camera blend preference',
            'material': 'lit',
            'astraCompletion': bool(completion),
            'eyewearRemovedFromSkin': bool(
                completion and completion['glasses']['present']
            ),
            'crownMapping': 'Cartesian triplanar photo swatch; no spherical pole',
            'method': (
                'Photographic face and hair with frontal feature ownership. '
                'Narrow opaque rim cleanup preserves observed eyes and brows; '
                'separate glasses retain their own material. Missing rear '
                'appearance is labeled estimated.'
            ),
        },
    }
    metadata['stats'].update(
        material='lit',
        eyewearRemovedFromSkin=False,
        opaqueFrameCleanupApplied=any(a['excludedPixels'] > 0 for a in mask_audit),
        eyewearMaskedViews=sum(a['excludedPixels'] > 0 for a in mask_audit),
        eyewearSkippedViews=sum(
            a['method'] == 'unverified-view-excluded' for a in mask_audit
        ),
        estimatedOccludedFaceFraction=float(np.mean(occluded[exposed])),
        lensInteriorsExcluded=bool(clean_references),
        skinDetail='3072px photographic color; observed eyes, eyebrows and hair preserved',
    )
    metadata['stats']['eyes'] = (
        eyes['summary']
        if eyes
        else 'Legacy photographic projection; eye detail has not been quality checked.'
    )
    metadata['stats']['eyeMaterialTexels'] = eye_texels
    if socket_shading:
        metadata['stats']['eyeSocketShading'] = socket_shading
    metadata['stats']['nativeVideoDetailViews'] = len(detail_audit['frames'])
    metadata['stats']['fineDetail'] = (
        'Native video pixels; a single visible camera owns fine eyebrow, beard '
        'and hair texture; only broad color is blended.'
    )
    metadata['stats']['material'] = 'photo'
    if supported_color_audit is not None:
        metadata['stats']['supportedColorPreservation'] = supported_color_audit
    metadata['stats']['semanticEarOwnership'] = bool(semantics and ear_regions)
    metadata['stats']['textureRegistration'] = {
        'version': 1,
        'earWarp': 'Local affine photo registration with geodesic falloff onto the connected scalp; measured face pinned',
        'semanticCoordinates': 'Hair, skin, alpha and occlusion masks share the registered RGB sampling coordinates',
        'posteriorVisibility': 'Triangle-plane first-hit depth at the same pixel-center ray, 0.5 mm tolerance',
        'occludedPosteriorSamplesRejected': sum(
            a.get('occludedPosteriorSamplesRejected', 0) for a in mask_audit
        ),
        'registeredPosteriorSamples': sum(
            a.get('registeredPosteriorSamples', 0) for a in mask_audit
        ),
        'geometryChanged': False,
        'scalpViewSelection': 'Broad head-relative camera direction with exact occlusion and source masks; physical support remains independent',
    }
    metadata['stats']['frameSelection'] = (
        'Registered angular coverage with measured sharpness/exposure ranking; '
        'semantic reference views retained.'
    )
    metadata['stats']['generatedGlassesCleanup'] = bool(clean_references)
    metadata['stats']['cleanupLightingMatchedTexels'] = lighting_matched
    metadata['stats']['exposureReferenceFrame'] = reference_frame.name
    metadata['stats']['exposureSkippedViews'] = sorted(
        name for name, value in cheek_cache.items() if value is None
    )
    metadata['stats'][
        'exposureDonors'
    ] = 'Opaque photographed cheek pixels; alpha-masked backgrounds cannot set exposure.'
    metadata['stats']['viewBlending'] = {
        'method': 'Jaw/neck frontal preference fades with the surface angle; registered upper facial features keep their exposure.',
        'coverageUsesUnpreferredEvidence': True,
        'missingProjectionSupportFraction': float(
            np.mean(photo_support[exposed] < 0.00001)
        ),
        'unresolvedFaceFraction': low,
        'physicalConfidenceExcludesPreference': True,
        'physicalConfidenceMaximum': float(best.max()),
        'frontPreferenceReleasedTexels': int(
            np.count_nonzero(
                (central > 0.5) & (front_facing <= 0.55) & (jaw_release > 0.99)
            )
        ),
        'limitation': 'Support is a viewing heuristic on prepared source images, including generated cleanup and restored alpha holes; it is not measured skin coverage, a calibrated probability or recovered albedo.',
    }
    if ownership_audit:
        metadata['stats']['cameraOwnershipSmoothing'] = ownership_audit
    if clean_references:
        metadata['stats']['glassesCleanup'] = (
            'Registered edited reference restricted to the glasses-affected '
            'region; hidden skin and lens-affected appearance are estimated.'
        )
    metadata['stats'][
        'neckClosure'
    ] = 'Closed planar neck section with an estimated skin material; no photo projection.'
    metadata['stats']['estimatedLowerSkinTexels'] = lower_skin_texels
    if lower_surface:
        metadata['stats']['lowerSurfaceContinuation'] = lower_surface
    metadata['stats']['estimatedEarSkinTexels'] = ear_skin_texels
    metadata['stats']['faceCoverageRegion'] = (
        'Visible skin within the measured facial outline; excludes unobserved '
        'crown, neck, eyeballs and hidden cavities.'
    )
    metadata['stats']['colour'] = (
        'Captured video colour and lighting retained; no '
        'additional studio relighting at rest.'
    )
    metadata['textureSha256'] = hashlib.sha256(
        (output_folder / 'appearance.png').read_bytes()
    ).hexdigest()
    metadata['positionsSha256'] = hashlib.sha256(
        np.asarray(p, dtype='<f4').tobytes()
    ).hexdigest()
    if eyes:
        metadata['roughnessTexture'] = (
            f'/api/face-asset?id={folder.name}&asset=appearance-roughness.png'
        )
    metadata['stats']['skinDetail'] = (
        (
            '3072px photographic skin, eyebrows and hair; eyeballs receive '
            'separate quality-checked iris material.'
        )
        if eyes
        else metadata['stats']['skinDetail']
    )
    metadata['stats']['method'] = (
        (
            'Photographic skin and hair with separate eye materials and 3D '
            'glasses. Missing eye detail is explicitly labeled estimated.'
        )
        if eyes
        else metadata['stats']['method']
    )
    atomic(output_folder / 'texture-atlas.json', metadata)
    return metadata['stats']
