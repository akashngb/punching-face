"""Harmonic continuation of photographed color along actual surface edges.

UV seams and spatially close but disconnected surfaces do not create edges.
Unknown components without photograph anchors deliberately remain unresolved.
"""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def complete_surface_colors(
    vertices, faces, binding, texel_color, source_mask, vertex_domain
):
    """Return ``(vertex_colors, resolved_vertex_mask, audit)``.

    ``binding`` contains original-vertex ``triangles``, per-texel ``triangleIds``
    and barycentric ``weights``. Trusted texels are accumulated onto their own
    triangle vertices, whose weighted means become fixed Dirichlet anchors.
    Unknown vertices are solved only inside ``vertex_domain``; supported
    vertices outside it may supply boundary values. Unsupported outside-domain
    vertices cannot provide a shortcut. Unresolved colors are zero placeholders.

    All unique mesh edges have positive unit conductance. This combinatorial
    Laplacian avoids negative cotangent weights and obeys the maximum principle.
    The caller should omit unwanted neck-cap faces from ``faces``.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    triangles = np.asarray(binding['triangles'])
    triangle_ids = np.asarray(binding['triangleIds'])
    weights = np.asarray(binding['weights'])
    color = np.asarray(texel_color)
    source = np.asarray(source_mask, dtype=bool)
    domain = np.asarray(vertex_domain, dtype=bool)
    n = len(vertices)
    if (
        vertices.shape != (n, 3)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
        or not np.issubdtype(triangles.dtype, np.integer)
        or not np.issubdtype(triangle_ids.dtype, np.integer)
        or triangle_ids.ndim != 1
        or color.shape != (len(triangle_ids), 3)
        or weights.shape != color.shape
        or source.shape != triangle_ids.shape
        or domain.shape != (n,)
    ):
        raise ValueError('Invalid surface color completion arrays.')
    for array, bound in ((faces, n), (triangles, n), (triangle_ids, len(triangles))):
        if array.size and (array.min() < 0 or array.max() >= bound):
            raise ValueError(
                'Surface color binding references an invalid vertex or triangle.'
            )
    total = np.zeros(n)
    accumulated = np.zeros((n, 3))
    # Atlas-sized arrays can contain millions of texels. Only a bounded chunk
    # gets expanded to triangle vertices; never allocate texels x 3 x RGB.
    for start in range(0, len(color), 65536):
        selected = np.flatnonzero(source[start : start + 65536]) + start
        if not len(selected):
            continue
        bary = weights[selected].astype(float)
        rgb = color[selected].astype(float)
        if (
            not np.isfinite(bary).all()
            or not np.isfinite(rgb).all()
            or np.any(bary < -1.1e-5)
        ):
            raise ValueError(
                'Trusted surface color samples have invalid values or barycentric weights.'
            )
        # Rasterization admits tiny negative border weights, including float32
        # rounding. Remove them before accumulation to preserve convexity.
        bary = np.maximum(bary, 0)
        sums = bary.sum(axis=1)
        if np.any(sums <= 0):
            raise ValueError(
                'Trusted surface color samples have zero barycentric support.'
            )
        bary /= sums[:, None]
        corners = triangles[triangle_ids[selected]]
        for corner in range(3):
            w = bary[:, corner]
            np.add.at(total, corners[:, corner], w)
            np.add.at(accumulated, corners[:, corner], rgb * w[:, None])
    used = np.zeros(n, bool)
    used[faces.ravel()] = True
    supported = (total > 0) & used
    result = np.zeros((n, 3))
    result[supported] = accumulated[supported] / total[supported, None]
    resolved = supported.copy()
    audit = {
        'estimated': True,
        'method': 'Surface-edge harmonic color continuation with photographed vertex anchors.',
        'edgeWeighting': 'Positive unit weight per unique mesh edge.',
        'sourceTexels': int(source.sum()),
        'anchoredVertices': int(supported.sum()),
        'solvedVertices': 0,
        'unresolvedDomainVertices': int(np.count_nonzero(domain & used & ~resolved)),
        'ignoredIsolatedVertices': int(np.count_nonzero(~used)),
        'unanchoredComponents': 0,
        'maximumEquationResidual': 0.0,
        'limitation': (
            'Missing surface appearance is interpolated, not photographed; '
            'components without anchors remain unresolved.'
        ),
    }
    if not len(faces):
        return result, resolved, audit
    edges = np.unique(
        np.sort(
            np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        ),
        axis=0,
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    active = used & (domain | supported)
    edges = edges[active[edges].all(axis=1)]
    ids = np.flatnonzero(active)
    if not len(ids):
        return result, resolved, audit
    local = np.full(n, -1, dtype=int)
    local[ids] = np.arange(len(ids))
    rows = np.r_[local[edges[:, 0]], local[edges[:, 1]]]
    cols = np.r_[local[edges[:, 1]], local[edges[:, 0]]]
    adjacency = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(ids), len(ids))
    ).tocsr()
    count, components = connected_components(adjacency, directed=False)
    anchors = supported[ids]
    anchored_component = np.bincount(components[anchors], minlength=count) > 0
    eligible = anchored_component[components]
    unknown = np.flatnonzero(~anchors & eligible)
    audit['unanchoredComponents'] = int(np.count_nonzero(~anchored_component))
    if len(unknown):
        pinned = np.flatnonzero(anchors)
        laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
        matrix = laplacian.tocsr()[unknown][:, unknown]
        rhs = adjacency[unknown][:, pinned] @ result[ids[pinned]]
        solved = np.asarray(spsolve(matrix.tocsc(), rhs)).reshape(len(unknown), 3)
        if not np.isfinite(solved).all():
            raise ValueError('Harmonic surface color solve produced nonfinite values.')
        low = np.full((count, 3), np.inf)
        high = np.full((count, 3), -np.inf)
        np.minimum.at(low, components[pinned], result[ids[pinned]])
        np.maximum.at(high, components[pinned], result[ids[pinned]])
        lower, upper = low[components[unknown]], high[components[unknown]]
        tolerance = 1e-8 * np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
        if np.any(solved < lower - tolerance) or np.any(solved > upper + tolerance):
            raise ValueError('Harmonic surface color solve violated its anchor bounds.')
        solved = np.clip(solved, lower, upper)  # Remove roundoff-only overshoot.
        audit['maximumEquationResidual'] = float(np.max(np.abs(matrix @ solved - rhs)))
        result[ids[unknown]] = solved
        resolved[ids[unknown]] = True
        audit['solvedVertices'] = int(len(unknown))
    audit['unresolvedDomainVertices'] = int(np.count_nonzero(domain & used & ~resolved))
    return result, resolved, audit
