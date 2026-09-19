"""Locally fit the hair outline without inflating the unseen crown.

The visual hull is a useful volume prior, but an orbit below the crown leaves
its top underconstrained. Refine that prior with visible contour residuals and
a screened membrane on displacement. Hidden vertices inherit only a smooth,
bounded correction; they never expand until they hit an unrelated silhouette.
"""

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import coo_matrix, eye, kron
from scipy.sparse.linalg import spsolve


def silhouette_field(mask):
    signed = gaussian_filter(
        distance_transform_edt(mask) - distance_transform_edt(~mask), 0.65
    )
    dy, dx = np.gradient(signed)
    return signed, dx, dy


def sample(image, xy):
    return cv2.remap(
        image.astype(np.float32),
        xy[:, 0].astype(np.float32).reshape(-1, 1),
        xy[:, 1].astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).ravel()


def membrane(faces, active, count):
    """A displacement penalty with fixed zero corrections at protected skin."""
    edges = np.unique(
        np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1),
        axis=0,
    )
    index = np.full(count, -1, int)
    index[active] = np.arange(len(active))
    edges = edges[np.any(index[edges] >= 0, axis=1)]
    row = np.repeat(np.arange(len(edges)), 2)
    col = index[edges].ravel()
    value = np.tile([1.0, -1.0], len(edges))
    keep = col >= 0
    gradient = coo_matrix(
        (value[keep], (row[keep], col[keep])), shape=(len(edges), len(active))
    ).tocsr()
    return gradient.T @ gradient


def solve_correction(prior, normals, residuals, strengths, smoothing=0.3):
    """Solve coupled XYZ contour constraints, retaining the unobserved prior.

    Each residual is in metres and its normal is the image contour gradient
    lifted by the camera Jacobian. This avoids converting brightness to depth.
    """
    count = prior.shape[0]
    block = np.einsum('vni,vnj,vn->nij', normals, normals, strengths)
    rhs = np.einsum('vni,vn,vn->ni', normals, residuals, strengths)
    rows = np.repeat(np.arange(count) * 3, 9) + np.tile(
        np.repeat(np.arange(3), 3), count
    )
    cols = np.repeat(np.arange(count) * 3, 9) + np.tile(np.tile(np.arange(3), 3), count)
    system = coo_matrix(
        (block.ravel(), (rows, cols)), shape=(count * 3, count * 3)
    ).tocsr()
    system += kron(prior * smoothing + eye(count) * 0.035, eye(3), format='csr')
    return spsolve(system, rhs.ravel()).reshape(-1, 3)


def refine_hair_contours(
    folder,
    vertices,
    faces,
    weight,
    allowance,
    views,
    rec,
    center,
    B,
    transform,
    iterations=7,
):
    from scripts.photo_geometry import zbuffer

    seed = np.asarray(vertices, float).copy()
    p = seed.copy()
    # A fractional ear margin is part of the protected attachment, not hair.
    active = np.where((weight > 0.15) & (allowance > 0.99))[0]
    if not len(active) or len(views) < 3:
        return p, {'available': False, 'reason': 'Insufficient hair silhouette support'}
    regularizer = membrane(faces, active, len(p))
    fields = []
    scale = transform['scale']
    for im in views:
        mask = np.asarray(Image.open(folder / 'images' / im.name).getchannel('A')) > 200
        # Source borders created by cropping cannot constrain hair volume.
        fields.append((im, mask, silhouette_field(mask)))
    support = np.zeros(len(active), int)
    initial_error = []
    final_error = []
    quality = {}
    for iteration in range(iterations + 1):
        mesh = trimesh.Trimesh(p, faces, process=False)
        vn = np.asarray(mesh.vertex_normals)[active]
        world = p / scale @ B + center
        all_normals, all_residuals, all_strengths = [], [], []
        errors = []
        for im, mask, field in fields:
            cam = rec.cameras[im.camera_id]
            pose = im.cam_from_world()
            cp = world @ pose.rotation.matrix().T + pose.translation
            xy_all = cam.img_from_cam(cp)
            xy = xy_all[active]
            h, w = mask.shape
            depth = zbuffer(xy_all, cp[:, 2], faces, w, h)
            rendered = np.isfinite(depth)
            # Distance to the rendered outer edge rejects interior contours,
            # back-facing vertices and clumps occluded by other hair.
            edge_distance = distance_transform_edt(rendered)
            ix = np.clip(np.rint(xy[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.rint(xy[:, 1]).astype(int), 0, h - 1)
            toward = im.projection_center() - world[active]
            toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-9)
            facing = np.sum((vn @ B) * toward, axis=1)
            visible_depth = cv2.dilate(
                np.where(rendered, -depth, -1e9), np.ones((3, 3))
            )
            visible = np.abs(cp[active, 2] + visible_depth[iy, ix]) * scale < 0.008
            inside = (
                (xy[:, 0] > 3)
                & (xy[:, 0] < w - 4)
                & (xy[:, 1] > 3)
                & (xy[:, 1] < h - 4)
                & (cp[active, 2] > 0)
            )
            distance, gx, gy = [sample(v, xy) for v in field]
            jacobian = []
            for axis in np.eye(3) * 0.00025:
                camera_step = (axis / scale @ B) @ pose.rotation.matrix().T
                differential = (
                    cam.img_from_cam(cp[active] + camera_step) - xy
                ) / 0.00025
                jacobian.append(gx * differential[:, 0] + gy * differential[:, 1])
            jacobian = np.array(jacobian).T
            norm = np.linalg.norm(jacobian, axis=1)
            usable = (
                inside
                & visible
                & (sample(edge_distance, xy) < 2.6)
                & (abs(facing) < 0.65)
                & (norm > 100)
                & (abs(distance) < 24)
            )
            normal = jacobian / np.maximum(norm[:, None], 1e-8)
            # Solve total displacement relative to seed, not accumulated free
            # expansion. Robust residuals limit a bad mask's local influence.
            residual = np.clip(-distance / np.maximum(norm, 1), -0.012, 0.012)
            residual += np.sum(normal * (p[active] - seed[active]), axis=1)
            strength = usable.astype(float) * np.clip(
                8 / np.maximum(abs(distance), 1), 0, 1
            )
            all_normals.append(normal)
            all_residuals.append(residual)
            all_strengths.append(strength)
            errors.extend(abs(distance[usable]).tolist())
            support += usable
        if iteration == 0:
            initial_error = errors
        final_error = errors
        if iteration == iterations:
            break
        correction = solve_correction(
            regularizer,
            np.array(all_normals),
            np.array(all_residuals),
            np.array(all_strengths),
        )
        length = np.linalg.norm(correction, axis=1)
        correction *= np.minimum(1, 0.022 / np.maximum(length, 1e-8))[:, None]
        target = seed[active] + correction
        p[active] = p[active] * 0.25 + target * 0.75
        if iteration == iterations - 1:
            from scripts.ear_deformation import bounded_surface_step

            p, quality = bounded_surface_step(seed, p, faces)
    return p, {
        'available': True,
        'method': 'Visible multiview contour solve with a screened displacement membrane',
        'views': len(fields),
        'directlyConstrainedVertices': int(np.sum(support > 0)),
        'maximumCorrectionMm': round(
            float(np.linalg.norm(p - seed, axis=1).max()) * 1000, 3
        ),
        'surfaceQuality': quality,
        'initialContourResidualMedianPx': (
            round(float(np.median(initial_error)), 3) if initial_error else None
        ),
        'finalContourResidualMedianPx': (
            round(float(np.median(final_error)), 3) if final_error else None
        ),
        'unseenCrown': 'Smooth continuation of nearby silhouette corrections; not measured strands',
    }
