"""Match a cleanup estimate to nearby captured illumination on the surface.

This adjusts a smooth color offset, not identity detail or recovered albedo.
Surface distances avoid UV-island seams and unrelated packed atlas neighbors.
"""

import numpy as np
from scipy.spatial import cKDTree


def match_cleanup_lighting(
    points, normals, parts, photo, clean, coverage, confidence, available, side_weight
):
    result = clean.copy()
    # Learn the exposure difference from the feather's outer skin, where the
    # original photographs still dominate. Do not use the hidden frame center.
    source = np.flatnonzero(
        (parts == 0)
        & available
        & (coverage > 0.025)
        & (coverage < 0.35)
        & (confidence > 0.15)
    )
    target = np.flatnonzero(
        (parts == 0) & available & (coverage > 0) & (side_weight > 0.05)
    )
    if len(source) < 12 or not len(target):
        return result, 0
    _, unique = np.unique(
        np.floor(points[source] / 0.0015).astype(int), axis=0, return_index=True
    )
    source = source[unique]
    if len(source) < 12:
        return result, 0
    tree = cKDTree(points[source])
    difference = photo[source] - clean[source]
    count = 0
    for first in range(0, len(target), 20000):
        ids = target[first : first + 20000]
        distance, near = tree.query(points[ids], k=min(32, len(source)))
        alignment = np.einsum('nij,nj->ni', normals[source[near]], normals[ids])
        valid = (distance < 0.025) & (alignment > 0.7)
        weights = (
            valid
            * np.exp(-0.5 * (distance / 0.010) ** 2)
            * np.maximum(alignment, 0) ** 4
        )
        enough = (valid.sum(axis=1) >= 5) & (weights.sum(axis=1) > 1e-7)
        if not enough.any():
            continue
        weights = weights[enough]
        delta = difference[near[enough]]
        # Bound outliers from residual rims or shadows, then use a robust
        # local mean. A single dark frame pixel cannot set the whole correction.
        median = np.nanmedian(np.where(valid[enough, :, None], delta, np.nan), axis=1)
        robust = np.minimum(
            1.0,
            0.06 / np.maximum(np.linalg.norm(delta - median[:, None], axis=2), 1e-8),
        )
        weights *= robust
        weights /= weights.sum(axis=1, keepdims=True)
        offset = np.clip(np.sum(delta * weights[:, :, None], axis=1), -0.25, 0.25)
        ids = ids[enough]
        amount = np.clip((side_weight[ids] - 0.05) / 0.45, 0, 1)
        amount = amount * amount * (3 - 2 * amount)
        result[ids] = np.clip(clean[ids] + offset * amount[:, None], 0, 1)
        count += len(ids)
    return result, count
