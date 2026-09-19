"""Estimated head appearance behind source-ear occlusion, along mesh edges only."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from scripts.surface_completion import complete_surface_colors


def _graph(vertices, faces):
    edges = np.unique(
        np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
            axis=1,
        ),
        axis=0,
    )
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    good = (edges[:, 0] != edges[:, 1]) & (length > 0)
    edges, length = edges[good], length[good]
    return coo_matrix(
        (
            np.r_[length, length],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()


def continue_ear_occlusion(
    vertices,
    faces,
    binding,
    photo_color,
    parts,
    vertex_parts,
    confidence,
    ear_occluded,
    *,
    photographed,
    preserve=None,
    observed_hair=None,
    max_distance=0.020,
):
    """Return ``(estimate, alpha, ownership, audit)``; the estimate is NOT already blended.

    Inputs use fitted model metres. ``confidence`` is the strongest unmasked
    ``max(best, hair_support)``. ``photographed`` explicitly identifies actual,
    unmasked head photographs, excluding generated cleanup and inpainted RGB.
    ``photo_color`` must be the photographic field before generic completion.
    ``preserve`` combines cleanup, mouth and bottom exclusions. ``observed_hair``
    protects positively observed semantic hair from becoming a fill target.
    ``ear_occluded`` is a Boolean mask or bounded soft ownership in [0, 1].

    Only part-0 targets below .12 support can receive nonzero alpha. Faces
    touching ear/eye vertices and planar bottom caps are excluded. Each donor
    vertex anchor reaches a target through actual mesh edges followed by a
    straight segment inside the target triangle; long triangles need not be
    discarded. The farthest contributing source-texel-to-anchor distance is
    added conservatively, so every contributing photographic color has a valid
    surface path strictly shorter than 20 mm. This bounded path estimate may
    overestimate a continuous surface geodesic, but cannot cross disconnected
    surfaces or jump Euclidean-near folds. Ownership fades over the last 4 mm.
    The radius is experimental, not a recovered anatomical measurement.

    To replace earlier fallback once, use ``current + ownership *
    (photo_color - current) + alpha * (estimate - photo_color)``. Zero
    ownership retains existing fallback; reserve ownership from later stages. This helper changes no physical/observed coverage fields.
    """
    p, f = np.asarray(vertices), np.asarray(faces)
    triangles = np.asarray(binding['triangles'])
    tids = np.asarray(binding['triangleIds'])
    bary = np.asarray(binding['weights'])
    color = np.asarray(photo_color)
    n, count = len(p), len(color)
    parts, labels = np.asarray(parts), np.asarray(vertex_parts)
    strength = np.asarray(confidence)
    occlusion = np.asarray(ear_occluded, dtype=float)
    source = np.asarray(photographed, dtype=bool)
    protected = (
        np.zeros(count, bool) if preserve is None else np.asarray(preserve, bool)
    )
    hair = (
        np.zeros(count, bool)
        if observed_hair is None
        else np.asarray(observed_hair, bool)
    )
    if (
        p.shape != (n, 3)
        or f.ndim != 2
        or f.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or color.shape != (count, 3)
        or bary.shape != color.shape
        or tids.shape != (count,)
        or labels.shape != (n,)
        or any(
            a.shape != (count,)
            for a in (parts, strength, occlusion, source, protected, hair)
        )
        or not np.isfinite(max_distance)
        or not 0 < max_distance <= 0.020
    ):
        raise ValueError('Invalid ear-occlusion continuation arrays or radius.')
    for array, bound in ((f, n), (triangles, n), (tids, len(triangles))):
        if not np.issubdtype(array.dtype, np.integer) or (
            array.size and (array.min() < 0 or array.max() >= bound)
        ):
            raise ValueError('Invalid ear-occlusion mesh binding.')
    if (
        not all(np.isfinite(a).all() for a in (p, color, strength, occlusion))
        or np.any(strength < 0)
        or np.any(occlusion < 0)
        or np.any(occlusion > 1)
    ):
        raise ValueError('Ear-occlusion inputs must be finite with valid support.')
    estimate, alpha = color.copy(), np.zeros(count, dtype=float)
    ownership = np.zeros(count, dtype=float)
    audit = dict(
        estimated=True,
        method='Compact positive edge-and-within-face interpolation of photographed anchors.',
        maxGraphDistanceMm=max_distance * 1000,
        targetTexels=0,
        resolvedTexels=0,
        donorTexels=0,
        unresolvedTargetTexels=0,
        excludedCapTriangles=0,
        physicalCoverageChanged=False,
        limitation='Interpolated appearance behind source ears; not observed skin or new coverage.',
    )
    if not len(f) or not count:
        return estimate, alpha, ownership, audit
    used = np.unique(f)
    cut = p[used, 1].min()
    cap = np.all(p[f, 1] <= cut + 0.0001, axis=1)
    safe = (labels[f] == 0).all(axis=1) & ~cap
    safe_faces = f[safe]
    audit['excludedCapTriangles'] = int(cap.sum())
    # Binding order may differ from the geometry face order; match original
    # vertex triples, not atlas neighbours or triangle indices by coincidence.
    face_keys = {tuple(row) for row in np.sort(safe_faces, axis=1)}
    allowed = np.array(
        [tuple(row) in face_keys for row in np.sort(triangles, axis=1)], dtype=bool
    )
    target = (
        (parts == 0)
        & (strength < 0.12)
        & (occlusion > 0)
        & ~protected
        & ~hair
        & allowed[tids]
    )
    audit['targetTexels'] = int(target.sum())
    audit['unresolvedTargetTexels'] = int(target.sum())
    if not target.any():
        return estimate, alpha, ownership, audit
    seeds = np.unique(triangles[tids[target]])
    graph = _graph(p, safe_faces)
    distance = dijkstra(graph, indices=seeds, min_only=True, limit=max_distance)
    domain = np.isfinite(distance)
    # Retain a donor face touching the bounded search neighbourhood even when
    # another corner is far away. Footprint distances below prevent that far
    # corner's photograph from becoming a zero-distance nearby anchor.
    touches_domain = domain[triangles].any(axis=1)
    donors = (
        source
        & (parts == 0)
        & (strength > 0.15)
        & ~protected
        & allowed[tids]
        & touches_domain[tids]
    )
    audit['donorTexels'] = int(donors.sum())
    if not donors.any():
        return estimate, alpha, ownership, audit
    vertex_color, anchored, anchor_audit = complete_surface_colors(
        p, safe_faces, binding, color, donors, np.zeros(n, bool)
    )
    footprint = np.zeros(n)
    donor_ids = np.flatnonzero(donors)
    for start in range(0, len(donor_ids), 65536):
        chunk = donor_ids[start : start + 65536]
        corners = triangles[tids[chunk]]
        weights = np.maximum(bary[chunk].astype(float), 0)
        weights /= weights.sum(axis=1, keepdims=True)
        points = np.sum(p[corners] * weights[:, :, None], axis=1)
        for corner in range(3):
            contributing = weights[:, corner] > 0
            radius = np.linalg.norm(points - p[corners[:, corner]], axis=1)
            np.maximum.at(
                footprint, corners[contributing, corner], radius[contributing]
            )
    anchor_ids = np.flatnonzero(anchored & domain & (footprint < max_distance))
    ids = np.flatnonzero(target)
    target_points = np.empty((len(ids), 3))
    corners = triangles[tids[ids]]
    for start in range(0, len(ids), 65536):
        chunk = ids[start : start + 65536]
        weights = bary[chunk].astype(float)
        if (
            not np.isfinite(weights).all()
            or np.any(weights < -1.1e-5)
            or np.any(np.maximum(weights, 0).sum(axis=1) <= 0)
        ):
            raise ValueError('Invalid target barycentric weights.')
        weights = np.maximum(weights, 0)
        weights /= weights.sum(axis=1, keepdims=True)
        target_points[start : start + len(chunk)] = np.sum(
            p[corners[start : start + len(chunk)]] * weights[:, :, None], axis=1
        )
    accumulated = np.zeros((len(ids), 3))
    mass = np.zeros(len(ids))
    nearest_path = np.full(len(ids), np.inf)
    greatest_distance = 0.0
    # 32 x vertex_count and 32 x target_chunk scratch arrays. Evaluate actual
    # texels, not a field interpolated from three all-or-nothing corner solves.
    for start in range(0, len(anchor_ids), 32):
        anchors = anchor_ids[start : start + 32]
        distances = np.atleast_2d(dijkstra(graph, indices=anchors, limit=max_distance))
        for offset in range(0, len(ids), 16384):
            end = min(offset + 16384, len(ids))
            local = corners[offset:end]
            points = target_points[offset:end]
            path = np.full((len(anchors), end - offset), np.inf)
            for corner in range(3):
                within_face = np.linalg.norm(points - p[local[:, corner]], axis=1)
                np.minimum(path, distances[:, local[:, corner]] + within_face, out=path)
            path += footprint[anchors, None]
            valid = path < max_distance
            weight = np.zeros_like(path)
            d = path[valid]
            weight[valid] = (1 - d / max_distance) ** 2 / (d + 0.002) ** 2
            if len(d):
                greatest_distance = max(greatest_distance, float(d.max()))
            accumulated[offset:end] += weight.T @ vertex_color[anchors]
            mass[offset:end] += weight.sum(axis=0)
            nearest_path[offset:end] = np.minimum(
                nearest_path[offset:end], path.min(axis=0)
            )
    resolved = mass > 0
    filled_ids = ids[resolved]
    estimate[filled_ids] = accumulated[resolved] / mass[resolved, None]
    reach = np.clip(
        (max_distance - nearest_path[resolved]) / min(0.004, max_distance), 0, 1
    )
    reach = reach * reach * (3 - 2 * reach)
    confidence_fade = np.clip((0.12 - strength[filled_ids]) / 0.12, 0, 1)
    confidence_fade = confidence_fade * confidence_fade * (3 - 2 * confidence_fade)
    ownership[filled_ids] = occlusion[filled_ids] * reach
    alpha[filled_ids] = ownership[filled_ids] * confidence_fade
    ownership[alpha == 0] = 0
    audit.update(
        resolvedTexels=int(np.count_nonzero(alpha)),
        unresolvedTargetTexels=int(np.count_nonzero(target & (alpha == 0))),
        anchoredVertices=int(anchored.sum()),
        eligibleAnchors=int(len(anchor_ids)),
        maximumUsedSurfacePathMm=greatest_distance * 1000,
        maximumDonorFootprintMm=(
            float(footprint[anchor_ids].max()) * 1000 if len(anchor_ids) else 0.0
        ),
        ownershipReachFadeMm=min(0.004, max_distance) * 1000,
        anchorAccumulation=anchor_audit,
        graphWeighting='(1-d/radius)^2 / (d+2mm)^2; d includes source footprint, mesh edges and target-face segment.',
        distanceLimitation='Conservative edge-plus-face paths depend on mesh tessellation; not exact continuous geodesics.',
    )
    return estimate, alpha, ownership, audit
