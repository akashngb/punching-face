"""Fit the visible hair envelope to the photographed head silhouettes.

This estimates a continuous outer hair surface, not individual hair strands.
No generated hair texture or Gaussian representation is used.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import csr_matrix


def ear_hair_allowance(vertices, faces, ear_regions):
    """Keep hair-envelope fitting off ears, with a smooth attachment margin."""
    from scipy.sparse.csgraph import dijkstra

    protected = np.zeros(len(vertices), bool)
    for region in (ear_regions or {}).values():
        protected[np.asarray(region['vertices'], int)] = True
    if not protected.any():
        return np.ones(len(vertices))
    edges = np.unique(
        np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
        ),
        axis=0,
    )
    a, b = edges.T
    length = np.linalg.norm(vertices[a] - vertices[b], axis=1)
    graph = csr_matrix(
        (np.r_[length, length], (np.r_[a, b], np.r_[b, a])),
        shape=(len(vertices), len(vertices)),
    )
    distance = dijkstra(
        graph,
        directed=False,
        indices=np.flatnonzero(protected),
        min_only=True,
        limit=0.012,
    )
    weight = np.clip(distance / 0.012, 0, 1)
    return weight * weight * (3 - 2 * weight)


def fit_hair_envelope(
    folder, vertices, faces, face_count, rec, frames, train, center, B, transform
):
    p = vertices.copy()
    front = np.zeros(len(p), bool)
    front[np.unique(faces[:face_count])] = True
    # The shared face-oval seam stays fixed; the surrounding cap may move.
    ids = np.where(~front)[0]
    origin = np.array([0.0, 0.015, -0.065])
    ray = p[ids] - origin
    sample = np.linspace(0.45, 1.9, 46)
    candidates = origin + ray[:, None, :] * sample[None, :, None]
    world = (B.T @ (candidates.reshape(-1, 3) / transform['scale']).T).T + center
    selected = {
        min(train, key=lambda im: abs(frames[im.name]['yaw'] - angle)).name
        for angle in np.linspace(-55, 55, 13)
    }
    constraints = []
    for im in train:
        if im.name not in selected:
            continue
        mask = np.asarray(Image.open(folder / 'images' / im.name).getchannel('A')) > 200
        # A small tolerance accounts for the 256-pixel segmentation network.
        signed = gaussian_filter(
            distance_transform_edt(mask) - distance_transform_edt(~mask), 0.8
        )
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        cp = (pose.rotation.matrix() @ world.T).T + pose.translation
        xy = cam.img_from_cam(cp)
        ij = np.rint(np.nan_to_num(xy, nan=-1)).astype(int)
        h, w = mask.shape
        inside = (
            (ij[:, 0] >= 0)
            & (ij[:, 0] < w)
            & (ij[:, 1] >= 0)
            & (ij[:, 1] < h)
            & (cp[:, 2] > 0)
        )
        ij[:, 0] = np.clip(ij[:, 0], 0, w - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, h - 1)
        distance = np.where(inside, signed[ij[:, 1], ij[:, 0]], -1000)
        constraints.append(distance.reshape(len(ids), len(sample)))
    # Require support in at least 85% of selected views. A stray segmentation
    # error in one photo must not erase the hairstyle from every view.
    supported = np.quantile(np.stack(constraints), 0.15, axis=0) > -2.5
    valid = np.any(supported, axis=1)
    last = np.max(np.where(supported, np.arange(len(sample))[None, :], -1), axis=1)
    scale = np.where(valid, sample[np.maximum(last, 0)], 1.0)
    # Silhouettes bound transverse dimensions; posterior depth is unobserved.
    # Blend back to the explicit prior where front-only views are ambiguous.
    upper = np.clip((p[ids, 1] - 0.025) / 0.07, 0, 1)
    forward = np.clip((p[ids, 2] + 0.15) / 0.10, 0, 1)
    weight = np.maximum(upper * 0.9, forward * 0.6)
    scale = 1 + (np.clip(scale, 0.60, 1.70) - 1) * weight
    displacement = ray * (scale[:, None] - 1)
    # Smooth neighboring cap displacements without changing face landmarks.
    adjacency = [set() for _ in p]
    for tri in faces[face_count:]:
        for j in range(3):
            adjacency[tri[j]].update([tri[(j + 1) % 3], tri[(j + 2) % 3]])
    delta = np.zeros_like(p)
    delta[ids] = displacement
    for _ in range(8):
        new = delta.copy()
        for i in ids:
            if adjacency[i]:
                new[i] = (
                    delta[i] * 0.5 + np.mean(delta[list(adjacency[i])], axis=0) * 0.5
                )
        delta = new
    # Front and side silhouettes cannot constrain posterior depth. Keep that
    # coordinate close to Astra's head prior instead of allowing a tall cone.
    delta[ids, 2] *= np.clip((vertices[ids, 2] + 0.14) / 0.12, 0, 1)
    p += delta
    crown = []
    for im in train:
        frame = frames[im.name]
        if abs(frame['yaw']) > 15:
            continue
        mask = Image.open(folder / 'images' / im.name).getchannel('A')
        box = mask.getbbox()
        h = mask.height
        forehead = frame['landmarks'][10]['y'] * h
        chin = frame['landmarks'][152]['y'] * h
        if chin - forehead > 80:
            crown.append((forehead - box[1]) / (chin - forehead))
    ratio = float(np.clip(np.median(crown) if crown else 0.25, 0.08, 0.55))
    hairline = vertices[10, 1]
    target_top = hairline + 0.20 * ratio
    above = (~front) & (p[:, 1] > hairline)
    if np.any(above):
        p[above, 1] = hairline + (p[above, 1] - hairline) * (target_top - hairline) / (
            p[above, 1].max() - hairline
        )
    row = []
    col = []
    values = []
    for i, neighbors in enumerate(adjacency):
        if neighbors:
            row.extend([i] * len(neighbors))
            col.extend(neighbors)
            values.extend([1 / len(neighbors)] * len(neighbors))
    average = csr_matrix((values, (row, col)), shape=(len(p), len(p)))
    for _ in range(14):
        for weight in [0.4, -0.42]:
            update = (average @ p) - p
            p[~front] += weight * update[~front]
    delta = p - vertices
    return p, {
        'method': (
            'Head-mask silhouette envelope fitted across recovered photo '
            'cameras; captured photographic hair texture.'
        ),
        'crownAboveHairlineMm': round(0.20 * ratio * 1000, 2),
        'silhouetteViews': len(selected),
        'maximumCapAdjustmentMm': float(np.max(np.linalg.norm(delta, axis=1)) * 1000),
        'limitation': (
            'A continuous outer hair surface is estimated. Individual strands '
            'and unobserved rear hair are not measured.'
        ),
    }


def fit_template_hair(
    folder,
    vertices,
    faces,
    face_count,
    rec,
    frames,
    views,
    center,
    B,
    transform,
    ear_regions=None,
):
    """Fit the photographed asymmetric envelope without smoothing away locks."""
    from scipy.sparse import csr_matrix

    p = vertices.copy()
    coarse = vertices[:468]
    hairline = coarse[10, 1]
    crown = []
    allowance = ear_hair_allowance(vertices, faces, ear_regions)
    for im in views:
        frame = frames[im.name]
        if not frame.get('landmarks') or abs(frame['yaw']) > 15:
            continue
        mask = Image.open(folder / 'images' / im.name).getchannel('A')
        box = mask.getbbox()
        h = mask.height
        forehead = frame['landmarks'][10]['y'] * h
        chin = frame['landmarks'][152]['y'] * h
        if chin - forehead > 80:
            crown.append((forehead - box[1]) / (chin - forehead))
    target_top = hairline + 0.20 * float(
        np.clip(np.median(crown) if crown else 0.30, 0.08, 0.55)
    )
    upper = p[:, 1] > hairline
    p[upper, 1] = hairline + (p[upper, 1] - hairline) * (target_top - hairline) / (
        p[:, 1].max() - hairline
    )
    origin = np.array([0, 0.025, -0.09])
    angle = np.abs(np.arctan2(p[:, 0], p[:, 2] - origin[2]))
    threshold = np.interp(angle, [0, 1, 1.7, np.pi], [hairline, 0.065, 0.06, -0.035])
    weight = np.clip((p[:, 1] - threshold) / 0.018, 0, 1)
    weight[:468] = 0
    weight[np.unique(faces[:face_count])] = 0
    weight[allowance == 0] = 0
    ids = np.where(weight > 0.001)[0]
    from scripts.frame_evidence import assess_frames, choose_views

    selected = choose_views(
        views, frames, np.arange(-180, 180, 30), assess_frames(folder)
    )
    ray = p[ids] - origin
    samples = np.linspace(0.45, 1.48, 181)
    candidates = origin + ray[:, None] * samples[None, :, None]
    world = (B.T @ (candidates.reshape(-1, 3) / transform['scale']).T).T + center
    constraints = []
    for im in views:
        if im.name not in selected:
            continue
        mask = np.asarray(Image.open(folder / 'images' / im.name).getchannel('A')) > 200
        signed = gaussian_filter(
            distance_transform_edt(mask) - distance_transform_edt(~mask), 1
        )
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        cp = (pose.rotation.matrix() @ world.T).T + pose.translation
        xy = cam.img_from_cam(cp)
        ij = np.rint(np.nan_to_num(xy, nan=-1)).astype(int)
        h, w = mask.shape
        inside = (
            (ij[:, 0] >= 0)
            & (ij[:, 0] < w)
            & (ij[:, 1] >= 0)
            & (ij[:, 1] < h)
            & (cp[:, 2] > 0)
        )
        ij[:, 0] = np.clip(ij[:, 0], 0, w - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, h - 1)
        constraints.append(
            np.where(inside, signed[ij[:, 1], ij[:, 0]], -1000).reshape(
                len(ids), len(samples)
            )
        )
    distances = np.quantile(np.stack(constraints), 0.05, axis=0)
    supported = distances > -0.5
    last = np.max(np.where(supported, np.arange(len(samples))[None], -1), axis=1)
    # A ray with no feasible sample must minimize its violation; keeping the
    # original inflated cap was adding hair outside the photographed outline.
    best = np.argmax(distances, axis=1)
    scale = samples[np.where(last >= 0, last, best)]
    delta = np.zeros_like(p)
    delta[ids] = ray * (scale[:, None] - 1) * weight[ids, None]
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    degree = np.bincount(rows, minlength=len(p))
    average = csr_matrix((1 / degree[rows], (rows, cols)), shape=(len(p), len(p)))
    # Keep a data term at every iteration. Repeated unanchored averaging plus
    # multiplying by the hairline weight previously erased the quiff/taper.
    target = delta.copy()
    for _ in range(6):
        delta = target * 0.72 + (average @ delta) * 0.28
    delta[weight == 0] = 0
    p += delta
    # Radial silhouette expansion also raises Y. With a ground-level orbit,
    # the unseen middle of the crown can grow into a cone inside all silhouettes.
    # Use the conservative crown only as an INITIAL prior. A final global cap
    # erased photographed quiff peaks; the local contour solve below restores
    # those without expanding the unseen crown into a visual-hull cone.
    ceiling = (
        target_top
        - 0.010 * (p[:, 0] / 0.12) ** 2
        - 0.018 * (np.clip(-p[:, 2] - 0.07, 0, None) / 0.16) ** 2
    )
    active = weight > 0
    y = p[active, 1]
    bound = ceiling[active]
    p[active, 1] = (y + bound - np.sqrt((y - bound) ** 2 + 0.004**2)) * 0.5
    p = vertices + (p - vertices) * allowance[:, None]
    p[:468] = coarse
    p[np.unique(faces[:face_count])] = vertices[np.unique(faces[:face_count])]
    from scripts.hair_silhouette import refine_hair_contours

    p, contour_fit = refine_hair_contours(
        folder,
        p,
        faces,
        weight,
        allowance,
        [im for im in views if im.name in selected],
        rec,
        center,
        B,
        transform,
    )
    return p, {
        'method': (
            'Multiview hair volume with local contour refinement; '
            'measured face and template ears pinned.'
        ),
        'contourRefinement': contour_fit,
        'earVerticesProtected': int(np.sum(allowance == 0)),
        'crownAboveHairlineMm': round((target_top - hairline) * 1000, 2),
        'actualCrownAboveHairlineMm': round((p[:, 1].max() - hairline) * 1000, 2),
        'silhouetteViews': len(selected),
        'maximumHairAdjustmentMm': float(
            np.linalg.norm(p - vertices, axis=1).max() * 1000
        ),
        'limitation': (
            'Outer hair volume is silhouette-constrained. Interior roots and '
            'unseen concavities remain estimates.'
        ),
    }
