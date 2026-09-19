"""Experimental local smoothing of camera ownership, never RGB or confidence.

Probabilities have shape (samples, cameras). Bindings contain original vertex
triples in ``triangles``, one ``triangleIds`` entry per sample, and barycentric
``weights``. The caller supplies actual mesh faces with cap shortcuts removed.

Defaults are conservative experiments: three half-steps, Gaussian edge scale
3 mm in estimated model units, and no edge longer than 8 mm. This gives small
local influence, not a strict 3 mm radius: the mathematical support is at most
three graph hops (24 mm); long-edge influence is exponentially suppressed.
Only sampled vertices conduct diffusion. Observed vertices outside the allowed
region are fixed boundary values; unsupported vertices have exactly zero delta.
"""

import numpy as np
from scipy.sparse import coo_matrix

_CHUNK = 16384


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer_array(value, name):
    raw = np.asarray(value)
    _require(
        np.issubdtype(raw.dtype, np.integer), name + ' must contain integer indices.'
    )
    return raw.astype(np.int64, copy=False)


def _binding(binding, count, vertex_count):
    triangles = _integer_array(binding['triangles'], 'triangles')
    ids = _integer_array(binding['triangleIds'], 'triangleIds')
    weights = np.asarray(binding['weights'])
    _require(
        triangles.ndim == 2
        and triangles.shape[1] == 3
        and ids.shape == (count,)
        and weights.shape == (count, 3),
        'Invalid surface binding shape.',
    )
    _require(
        np.all((triangles >= 0) & (triangles < vertex_count))
        and np.all((ids >= 0) & (ids < len(triangles))),
        'Surface binding index is out of bounds.',
    )
    return triangles, ids, weights


def _barycentric(weights):
    value = np.asarray(weights, dtype=np.float64)
    _require(
        np.isfinite(value).all() and (value >= -1.1e-5).all(),
        'Invalid barycentric weights.',
    )
    value = np.maximum(value, 0)
    mass = value.sum(axis=1)
    _require(
        (mass > 0).all() and np.allclose(mass, 1, rtol=0, atol=1e-4),
        'Barycentric weights must sum to one.',
    )
    return value / mass[:, None]


def _probabilities(value):
    p = np.asarray(value)
    _require(
        p.ndim == 2 and p.shape[1] > 0 and np.issubdtype(p.dtype, np.floating),
        'Probabilities must be a floating (samples, cameras) array.',
    )
    return p


def _check_probabilities(p):
    _require(
        np.isfinite(p).all() and (p >= 0).all() and (p <= 1 + 1e-6).all(),
        'Camera probabilities must be finite and bounded.',
    )
    total = p.sum(axis=1, dtype=np.float64)
    _require(
        np.all((total == 0) | (np.abs(total - 1) <= 1e-5)),
        'Nonempty camera probabilities must sum to one.',
    )
    return total


def fit_camera_ownership_delta(
    vertices,
    faces,
    binding,
    probabilities,
    sample_eligible=None,
    vertex_domain=None,
    *,
    steps=3,
    step_size=0.5,
    diffusion_scale=0.003,
    max_edge_length=0.008,
):
    """Return (vertex_delta[V, cameras], audit) from normalized photo ownership.

    Zero probability rows are unsupported. Eligibility excludes samples such
    as estimates, eyes and scalp. Domain controls which supported vertices may
    change; supported neighbors outside it provide pinned boundary conditions.
    No spatial nearest-neighbor edges or unsupported-vertex bridges are added.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    _require(
        vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all(),
        'Vertices must be finite 3D positions.',
    )
    faces = _integer_array(faces, 'faces')
    _require(
        faces.ndim == 2
        and faces.shape[1] == 3
        and np.all((faces >= 0) & (faces < len(vertices))),
        'Invalid mesh faces.',
    )
    p = _probabilities(probabilities)
    triangles, ids, bary = _binding(binding, len(p), len(vertices))
    eligible = (
        np.ones(len(p), bool)
        if sample_eligible is None
        else np.asarray(sample_eligible, bool)
    )
    domain = (
        np.ones(len(vertices), bool)
        if vertex_domain is None
        else np.asarray(vertex_domain, bool)
    )
    _require(
        eligible.shape == (len(p),) and domain.shape == (len(vertices),),
        'Invalid eligibility/domain shape.',
    )
    _require(
        isinstance(steps, (int, np.integer)) and 0 <= steps <= 16,
        'Diffusion steps must be an integer from zero to 16.',
    )
    _require(
        np.isfinite(step_size)
        and 0 <= step_size <= 1
        and np.isfinite(diffusion_scale)
        and diffusion_scale > 0
        and np.isfinite(max_edge_length)
        and max_edge_length > 0,
        'Invalid diffusion parameters.',
    )
    numerator = np.zeros((len(vertices), p.shape[1]), dtype=np.float64)
    mass = np.zeros(len(vertices), dtype=np.float64)
    source_count = 0
    for start in range(0, len(p), _CHUNK):
        stop = min(start + _CHUNK, len(p))
        row_mass = _check_probabilities(p[start:stop])
        weights = _barycentric(bary[start:stop])
        take = eligible[start:stop] & (row_mass > 0)
        corners = triangles[ids[start:stop][take]]
        weights = weights[take]
        values = p[start:stop][take].astype(np.float64) / row_mass[take, None]
        source_count += int(take.sum())
        for corner in range(3):
            np.add.at(mass, corners[:, corner], weights[:, corner])
            np.add.at(numerator, corners[:, corner], weights[:, corner, None] * values)
    supported = mass > 1e-12
    original = np.divide(
        numerator, mass[:, None], out=np.zeros_like(numerator), where=supported[:, None]
    )
    edges = np.unique(
        np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
            axis=1,
        ),
        axis=0,
    )
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    keep = (
        (length > 1e-12)
        & (length <= max_edge_length)
        & supported[edges[:, 0]]
        & supported[edges[:, 1]]
    )
    edges, length = edges[keep], length[keep]
    # Symmetric flux with a degree bound gives a positive convex update. It
    # preserves total vertex probability mass when no boundary is pinned.
    degree = np.bincount(edges.ravel(), minlength=len(vertices))
    conductance = np.exp(-((length / diffusion_scale) ** 2)) / np.maximum(
        degree[edges[:, 0]], degree[edges[:, 1]]
    )
    graph = coo_matrix(
        (
            np.r_[conductance, conductance],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    outgoing = np.asarray(graph.sum(axis=1)).ravel()
    current = original.copy()
    active = supported & domain
    for _ in range(steps):
        proposal = current * (1 - step_size * outgoing[:, None]) + step_size * (
            graph @ current
        )
        current[active] = proposal[active]
    delta = current - original
    delta[~active] = 0
    audit = {
        'estimated': True,
        'method': 'Positive local mesh-edge diffusion of normalized photographic camera ownership.',
        'sourceSamples': source_count,
        'supportedVertices': int(supported.sum()),
        'activeVertices': int(active.sum()),
        'pinnedVertices': int((supported & ~domain).sum()),
        'unsupportedVertices': int((~supported).sum()),
        'diffusionEdges': len(edges),
        'steps': int(steps),
        'stepSize': float(step_size),
        'edgeScaleMm': float(diffusion_scale * 1000),
        'maximumEdgeMm': float(max_edge_length * 1000),
        'maximumGraphReachMm': (
            float(steps * length.max() * 1000) if len(length) else 0.0
        ),
        'maximumProbabilityDelta': float(np.max(np.abs(delta))) if delta.size else 0.0,
        'sourceRGBChanged': False,
        'physicalConfidenceChanged': False,
        'limitation': 'Experimental ownership smoothing; it cannot recover absent camera support.',
    }
    return delta, audit


def apply_camera_ownership_delta(
    probabilities,
    binding,
    vertex_delta,
    support_mask=None,
    blend=1.0,
    *,
    retained_mass_low=0.8,
    retained_mass_high=0.95,
):
    """Return (probabilities, audit), applying only the interpolated correction.

    The original camera support is reapplied before normalization. Blend may be
    scalar or per texel, so callers can pin central features and protected parts
    exactly. If masking loses >20% of candidate mass, retain the original mix;
    between 80% and 95% retention, smoothly release this safeguard. Existing
    zero-support rows remain zero. Original fine-detail winners are not inputs.
    """
    p = _probabilities(probabilities)
    delta = np.asarray(vertex_delta, dtype=np.float64)
    _require(
        delta.ndim == 2 and delta.shape[1] == p.shape[1] and np.isfinite(delta).all(),
        'Invalid vertex probability delta.',
    )
    _require(
        np.all(np.abs(delta.sum(axis=1)) <= 1e-5),
        'Vertex deltas must preserve probability mass.',
    )
    triangles, ids, bary = _binding(binding, len(p), len(delta))
    support = None if support_mask is None else np.asarray(support_mask, dtype=bool)
    _require(
        support is None or support.shape == p.shape, 'Invalid camera support shape.'
    )
    amount = np.asarray(blend, dtype=float)
    _require(
        amount.ndim == 0 or amount.shape == (len(p),),
        'Blend must be scalar or one value per texel.',
    )
    _require(
        np.isfinite(amount).all() and (amount >= 0).all() and (amount <= 1).all(),
        'Blend must be in [0, 1].',
    )
    _require(
        0 <= retained_mass_low < retained_mass_high <= 1,
        'Invalid retained-mass thresholds.',
    )
    result = p.copy()
    changed = retreated = no_support = 0
    minimum_retained = 1.0
    for start in range(0, len(p), _CHUNK):
        stop = min(start + _CHUNK, len(p))
        old = p[start:stop].astype(np.float64)
        old_mass = _check_probabilities(old)
        allowed = old > 0 if support is None else support[start:stop]
        _require(
            not np.any((old > 0) & ~allowed),
            'Support mask rejects an existing camera probability.',
        )
        mix = (
            np.full(stop - start, float(amount))
            if amount.ndim == 0
            else amount[start:stop]
        )
        if not np.any(mix > 0):
            continue  # Exact zero-blend identity, with no rasterization roundtrip.
        weights = _barycentric(bary[start:stop])
        corners = triangles[ids[start:stop]]
        correction = np.sum(delta[corners] * weights[:, :, None], axis=1)
        active = (mix > 0) & (old_mass > 0) & np.any(correction != 0, axis=1)
        no_support += int(np.count_nonzero((mix > 0) & (old_mass == 0)))
        if not active.any():
            continue
        tentative = np.maximum(old[active] + correction[active], 0)
        total_mass = tentative.sum(axis=1)
        tentative *= allowed[active]
        retained = tentative.sum(axis=1)
        ratio = np.divide(
            retained, total_mass, out=np.zeros_like(retained), where=total_mass > 0
        )
        trust = np.clip(
            (ratio - retained_mass_low) / (retained_mass_high - retained_mass_low), 0, 1
        )
        trust *= trust * (3 - 2 * trust)
        alpha = mix[active] * trust
        candidate = np.divide(
            tentative,
            retained[:, None],
            out=np.zeros_like(tentative),
            where=retained[:, None] > 0,
        )
        updated = old[active] * (1 - alpha[:, None]) + candidate * alpha[:, None]
        rows = start + np.flatnonzero(active)
        # Zero trust must retain original floating-point bits, not re-normalize.
        take = alpha > 0
        result[rows[take]] = updated[take]
        changed += int(
            np.count_nonzero(np.any(result[rows[take]] != p[rows[take]], axis=1))
        )
        retreated += int(np.count_nonzero(trust < 1))
        minimum_retained = min(minimum_retained, float(ratio.min()))
    return result, {
        'changedTexels': changed,
        'retreatedTexels': retreated,
        'unsupportedTexelsUnchanged': no_support,
        'minimumRetainedMass': minimum_retained,
        'sourceRGBChanged': False,
        'physicalConfidenceChanged': False,
    }
