"""Surface-continuous photo registration and first-hit visibility.

Design references (algorithmic guidance, no borrowed code/weights):
Zhou & Koltun, Color Map Optimization (2014), local image correction;
Im2SurfTex (2025), geodesic surface neighborhoods for coherent backprojection;
GOATex (NeurIPS 2025), explicit occlusion layers instead of angle-only blending.
See docs/TEXTURE_REGISTRATION.md for the research and implementation boundary.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def ear_registration_weights(vertices, faces, face_count, regions, binding):
    """Smooth ear photo warps onto the attached scalp across actual mesh edges.

    A hard material-label switch in sample coordinates produces a seam even
    when both neighboring pixels are from the same photograph. Distances here
    never jump across an ear/scalp gap or across disconnected surface sheets.
    The measured facial surface and rig cage are exact zero-displacement pins.
    """
    p, f = np.asarray(vertices), np.asarray(faces)
    edges = np.unique(
        np.sort(
            np.vstack(
                [
                    f[:, [0, 1]],
                    f[:, [1, 2]],
                    f[:, [2, 0]],
                ]
            ),
            axis=1,
        ),
        axis=0,
    )
    a, b = edges.T
    length = np.linalg.norm(p[a] - p[b], axis=1)
    graph = coo_matrix(
        (np.r_[length, length], (np.r_[a, b], np.r_[b, a])),
        shape=(len(p), len(p)),
    ).tocsr()
    fixed = np.zeros(len(p), bool)
    fixed[:468] = True
    fixed[np.unique(f[:face_count])] = True
    result = {}
    for sign, region in (regions or {}).items():
        core = np.asarray(region['coreVertices'], int)
        if not len(core):
            continue
        anchors = region.get('anchors', {})
        height = (
            np.linalg.norm(p[anchors['top']] - p[anchors['bottom']])
            if 'top' in anchors and 'bottom' in anchors
            else 0.06
        )
        radius = float(np.clip(height * 0.8, 0.025, 0.055))
        distance = dijkstra(
            graph, directed=False, indices=core, min_only=True, limit=radius
        )
        weight = np.clip(1 - distance / radius, 0, 1)
        weight = weight * weight * (3 - 2 * weight)
        weight[fixed] = 0
        # Original surface correspondence is shared across UV chart seams.
        texel = np.empty(len(binding['triangleIds']), np.float32)
        for start in range(0, len(texel), 65536):
            part = slice(start, start + 65536)
            corners = binding['triangles'][binding['triangleIds'][part]]
            texel[part] = np.sum(weight[corners] * binding['weights'][part], axis=1)
        result[sign] = np.clip(texel, 0, 1)
    return result


def plane_hit_visibility(
    camera_points, camera_normals, pixel_rays, first_depth, scale, tolerance_mm=0.5
):
    """Test the surface plane against the front hit at the same pixel center.

    Comparing raw texel depth with a neighboring pixel-center depth requires
    a large tolerance on sloping triangles. Correcting for the plane slope
    lets us reject a second surface only 1-6 mm behind an ear. No geometry,
    camera poses, source images or depth buffers are changed.
    """
    cp, normal, rays = [
        np.asarray(v) for v in (camera_points, camera_normals, pixel_rays)
    ]
    numerator = np.einsum('ij,ij->i', normal, cp)
    denominator = np.einsum('ij,ij->i', normal, rays)
    depth = np.divide(
        numerator,
        denominator,
        out=np.full(len(cp), np.inf),
        where=abs(denominator) > 1e-9,
    )
    valid = np.isfinite(first_depth) & np.isfinite(depth) & (depth > 0)
    error = np.full(len(cp), np.inf)
    error[valid] = abs(depth[valid] - np.asarray(first_depth)[valid]) * scale
    return valid & (error < tolerance_mm * 0.001)


def posterior_view_weight(points, facing, quality, camera_origin, eligible):
    """Coherent scalp view preference, separate from photographic confidence.

    Fine tessellation or reconstructed hair ridges must not switch ownership
    between distant photographs every few pixels. Use a broad head-relative
    direction to select cameras while retaining actual visibility and source
    masks in quality. Grazing projections remain disfavored by projected area.
    """
    radial = np.asarray(points)[:, [0, 2]].copy()
    radial[:, 1] += 0.11
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    direction = np.asarray(camera_origin)[[0, 2]] - points[:, [0, 2]]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    alignment = np.clip(np.sum(radial * direction, axis=1), 0, 1)
    area = np.divide(
        quality, facing**6, out=np.zeros_like(quality), where=facing > 0.001
    )
    coherent = area * (0.01 + 0.99 * alignment**8)
    fade = np.clip((-points[:, 2] - 0.075) / 0.05, 0, 1)
    fade = fade * fade * (3 - 2 * fade) * np.asarray(eligible)
    return quality * (1 - fade) + coherent * fade
