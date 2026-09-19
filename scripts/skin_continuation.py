"""Continue observed lower-head color into unobserved neck/jaw regions.

This is a local appearance estimate, not recovered anatomy or skin albedo.
It avoids constant-color triangle patches at the segmented video boundary.
"""

import numpy as np
from scipy.spatial import cKDTree


def continue_ear_skin(
    points,
    color,
    confidence,
    parts,
    normals,
    *,
    vertices=None,
    faces=None,
    binding=None,
    regions=None,
):
    """Fill unsupported ear patches from nearby evidence on that same ear.

    A cheek-colored constant produced bright triangles at unseen helix edges.
    This is bounded appearance continuation, not additional measured coverage.
    Opposite-facing folds, the other ear and neighboring scalp cannot donate.
    """
    result = color.copy()
    count = 0
    for part in (3, 4):
        source = np.flatnonzero((parts == part) & (confidence > 0.15))
        target = np.flatnonzero((parts == part) & (confidence < 0.08))
        if len(source) < 8 or not len(target):
            continue
        # Equal spatial support regardless of atlas tessellation/resolution.
        _, unique = np.unique(
            np.floor(points[source] / 0.0015).astype(int), axis=0, return_index=True
        )
        source = source[unique]
        if len(source) < 8:
            continue
        distance, near = cKDTree(points[source]).query(
            points[target], k=min(16, len(source))
        )
        donors = source[near]
        alignment = np.einsum('nij,nj->ni', normals[donors], normals[target])
        valid = (distance < 0.018) & (alignment > 0.25)
        weight = valid * np.maximum(alignment, 0) ** 2 / (distance + 0.002) ** 2
        enough = (valid.sum(axis=1) >= 3) & (weight.sum(axis=1) > 0)
        if not enough.any():
            continue
        target = target[enough]
        weight = weight[enough]
        donors = donors[enough]
        weight /= weight.sum(axis=1, keepdims=True)
        estimate = np.sum(color[donors] * weight[:, :, None], axis=1)
        alpha = np.clip((0.08 - confidence[target]) / 0.08, 0, 1)
        alpha = alpha * alpha * (3 - 2 * alpha)
        result[target] = (
            color[target] * (1 - alpha[:, None]) + estimate * alpha[:, None]
        )
        count += int(np.count_nonzero(alpha > 0.5))
    if binding is not None and regions:
        # A back-facing ear has no aligned front-facing donors, so the local
        # normal-filtered fill above cannot cover it. Continue color around
        # the actual connected auricle instead of leaving the scalp material
        # or copying across the air gap. Reliable photographs remain exact.
        from scripts.surface_completion import complete_surface_colors

        for sign, region in regions.items():
            part = 3 if int(sign) < 0 else 4
            domain = np.zeros(len(vertices), bool)
            domain[np.asarray(region['coreVertices'], int)] = True
            source = (parts == part) & (confidence > 0.15)
            targets = np.flatnonzero((parts == part) & (confidence < 0.08))
            if not len(targets) or np.count_nonzero(source) < 8:
                continue
            field, resolved, _ = complete_surface_colors(
                vertices, faces, binding, color, source, domain
            )
            for start in range(0, len(targets), 65536):
                ids = targets[start : start + 65536]
                corners = binding['triangles'][binding['triangleIds'][ids]]
                valid = resolved[corners].all(axis=1)
                ids, corners = ids[valid], corners[valid]
                bary = np.maximum(binding['weights'][ids], 0)
                bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-9)
                estimate = np.sum(field[corners] * bary[:, :, None], axis=1)
                alpha = np.clip((0.08 - confidence[ids]) / 0.08, 0, 1)
                alpha = alpha * alpha * (3 - 2 * alpha)
                result[ids] = (
                    color[ids] * (1 - alpha[:, None]) + estimate * alpha[:, None]
                )
        count = int(np.count_nonzero(np.max(abs(result - color), axis=1) > 1e-8))
    return result, count


def continue_lower_skin(
    points,
    color,
    confidence,
    face,
    parts,
    scalp,
    faces,
    binding,
    source_confidence=None,
    preserve=None,
    pre_completion_color=None,
):
    """Continue photographed anchors over the connected lower-head surface.

    Reliable pixels stay exact; unknown texels share a barycentric mesh field,
    including across UV seams. The artificial neck cap supplies no graph edges.
    Mouth interiors, eyes, ears and scalp retain their separate materials.
    With pre_completion_color, resolved lower skin blends once from its
    prepared-image base (which may include cleanup estimates); the fallback remains outside that ownership
    region and wherever the surface field cannot be resolved.
    """
    from scripts.surface_completion import complete_surface_colors

    def smooth(v):
        v = np.clip(v, 0, 1)
        return v * v * (3 - 2 * v)

    chin = face[152, 1]
    preserve = (
        np.zeros(len(points), bool) if preserve is None else np.asarray(preserve, bool)
    )
    evidence = confidence if source_confidence is None else source_confidence
    region = (
        smooth((chin + 0.085 - points[:, 1]) / 0.025)
        * smooth((-0.005 - points[:, 2]) / 0.010)
        * (parts == 0)
        * (1 - scalp)
    )
    if pre_completion_color is not None:
        pre_completion_color = np.asarray(pre_completion_color)
        if pre_completion_color.shape != color.shape:
            raise ValueError('Pre-completion colors must match the texel colors.')
        region[preserve] = 0
    weight = region * smooth((0.12 - confidence) / 0.12)
    weight[preserve] = 0
    source = (
        (points[:, 1] < chin + 0.115)
        & (points[:, 2] < -0.025)
        & (parts == 0)
        & (scalp < 0.1)
        & (evidence > 0.15)
        & ~preserve
    )
    # A planar presentation cap must not connect opposite sides of the neck.
    used = np.unique(faces)
    cut = face[used, 1].min()
    cap = np.all(face[faces, 1] < cut + 0.0001, axis=1)
    surface_faces = faces[~cap]
    vertex_color, resolved, audit = complete_surface_colors(
        face, surface_faces, binding, color, source, face[:, 1] < chin + 0.115
    )
    audit['excludedCapTriangles'] = int(cap.sum())
    result = color.copy()
    count = 0
    unresolved = 0
    targets = np.flatnonzero(
        region > 0 if pre_completion_color is not None else weight > 0.001
    )
    for start in range(0, len(targets), 65536):
        ids = targets[start : start + 65536]
        corners = binding['triangles'][binding['triangleIds'][ids]]
        valid = resolved[corners].all(axis=1)
        unresolved += int(np.count_nonzero(~valid))
        ids = ids[valid]
        corners = corners[valid]
        if not len(ids):
            continue
        bary = np.maximum(binding['weights'][ids].astype(float), 0)
        bary /= bary.sum(axis=1, keepdims=True)
        estimate = np.sum(vertex_color[corners] * bary[:, :, None], axis=1)
        alpha = weight[ids, None]
        result[ids] = color[ids] * (1 - alpha) + estimate * alpha
        if pre_completion_color is not None:
            # Undo only the general fallback owned by this resolved skin
            # region. Otherwise two fades introduce a bright intermediate-
            # confidence belt even when photo and harmonic colors agree.
            result[ids] += (pre_completion_color[ids] - color[ids]) * (
                region[ids, None] - alpha
            )
        count += int(np.count_nonzero(alpha > 0.5))
    audit.update(
        estimatedTexels=count,
        unresolvedTargetTexels=unresolved,
        photographedTexelsUnchanged=True,
    )
    return result, count, audit
