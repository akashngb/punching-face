"""Local shape-preserving ear deformation with explicit element-quality gates."""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized
from scipy.sparse.csgraph import dijkstra


def surface_quality(rest, candidate, faces):
    a = rest[faces]
    b = candidate[faces]
    na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    sa = np.linalg.norm(na, axis=1)
    sb = np.linalg.norm(nb, axis=1)
    valid = sa > 1e-12
    cosine = np.sum(na[valid] * nb[valid], axis=1) / np.maximum(
        sa[valid] * sb[valid], 1e-30
    )
    ratio = sb[valid] / sa[valid]
    return {
        'reversedTriangles': int(np.sum(cosine < 0)),
        'minimumNormalAgreement': float(cosine.min(initial=1)),
        'minimumAreaRatio': float(ratio.min(initial=1)),
        'maximumAreaRatio': float(ratio.max(initial=1)),
    }


def bounded_surface_step(rest, desired, faces):
    """Find a large verified step without discarding 25% for a narrow miss."""
    from scripts.surface_intersections import new_crossings

    def evaluate(fraction, intersections=False):
        candidate = rest + (desired - rest) * fraction
        quality = surface_quality(rest, candidate, faces)
        valid = (
            quality['minimumNormalAgreement'] > 0.1
            and quality['minimumAreaRatio'] > 0.25
            and quality['maximumAreaRatio'] < 3
        )
        if valid and intersections:
            quality.update(new_crossings(rest, candidate, faces))
            valid = not quality['newCrossings']
        return candidate, quality, valid

    fraction = 1.0
    for _ in range(20):
        candidate, quality, valid = evaluate(fraction, True)
        if valid:
            break
        fraction *= 0.75
    else:
        candidate = rest.copy()
        fraction = 0.0
        quality = surface_quality(rest, rest, faces)
        quality.update(existingCrossings=0, newCrossings=0, remainingCrossings=0)
    if 0 < fraction < 1:
        accepted = fraction
        low = fraction
        high = min(1.0, fraction / 0.75)
        # First refine cheap element checks. Most near-threshold misses need
        # only one additional intersection test instead of six full searches.
        for _ in range(6):
            middle = (low + high) * 0.5
            if evaluate(middle)[2]:
                low = middle
            else:
                high = middle
        proposal, report, valid = evaluate(low, True)
        if valid:
            candidate, quality, fraction = proposal, report, low
        else:
            high = low
            low = accepted
            for _ in range(6):
                middle = (low + high) * 0.5
                proposal, report, valid = evaluate(middle, True)
                if valid:
                    candidate, quality, fraction = proposal, report, middle
                    low = middle
                else:
                    high = middle
    quality.update(appliedFraction=fraction, limitedBySurfaceQuality=fraction < 1)
    return candidate, quality


def regularize(rest, desired, faces, regions, protected):
    n = len(rest)
    allowed = np.zeros(n, bool)
    soft = np.zeros(n)
    for region in regions.values():
        allowed[region['vertices']] = True
        soft[region['vertices']] = np.maximum(
            soft[region['vertices']], np.asarray(region['weights']) * 0.05
        )
        for index in region['anchors'].values():
            soft[index] = 10
    edges = np.unique(
        np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
        ),
        axis=0,
    )
    i, j = edges.T
    rr = np.r_[i, j]
    cc = np.r_[j, i]
    degree = np.bincount(rr, minlength=n)
    lengths = np.linalg.norm(rest[i] - rest[j], axis=1)
    graph = coo_matrix((np.r_[lengths, lengths], (rr, cc)), shape=(n, n)).tocsr()
    distance = dijkstra(
        graph,
        directed=False,
        indices=np.flatnonzero(allowed),
        min_only=True,
        limit=0.035,
    )
    allowed |= distance < 0.035
    fixed = ~allowed | protected
    soft[fixed] = 1e6
    target = desired.copy()
    target[fixed] = rest[fixed]
    L = diags(degree) - coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)).tocsr()
    solve = factorized((L + diags(soft + 0.001)).tocsc())
    edge = rest[i] - rest[j]
    positions = rest.copy()
    for _ in range(100):
        current = positions[i] - positions[j]
        outer = np.einsum('ni,nj->nij', current, edge)
        covariance = np.zeros((n, 3, 3))
        np.add.at(covariance, i, outer)
        np.add.at(covariance, j, outer)
        u, _, v = np.linalg.svd(covariance)
        u[:, :, -1] *= np.linalg.det(u @ v)[:, None]
        rotation = u @ v
        rhs = np.zeros((n, 3))
        value = np.einsum('nij,nj->ni', (rotation[i] + rotation[j]) * 0.5, edge)
        np.add.at(rhs, i, value)
        np.add.at(rhs, j, -value)
        rhs += target * soft[:, None] + rest * 0.001
        positions = np.stack([solve(rhs[:, axis]) for axis in range(3)], axis=1)
        positions[fixed] = rest[fixed]
    # A watertight mesh can still contain folded/sliver triangles. Back off
    # rather than publishing an overfit measurement at the scalp attachment.
    candidate, quality = bounded_surface_step(rest, positions, faces)
    if not np.array_equal(candidate[protected], rest[protected]):
        raise ValueError('Ear regularization changed protected geometry.')
    quality.update(
        method=(
            'Whole-ear as-rigid-as-possible fit with geodesic attachment '
            'support, area/orientation bounds, refined step search and '
            'non-adjacent crossing checks.'
        )
    )
    return candidate, quality
