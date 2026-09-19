"""Continue fitted hair displacement into the posterior neck on mesh edges.

The normalized head convention reserves vertices 0..467 for the measured cage
and uses metres. This is an estimated transition, not recovered neck anatomy.
The full hair envelope and existing ear fit remain fixed. No files are read or
written; callers must carry the returned displacement onto their pre-ear
baseline before saving a geometry refinement.
"""

import warnings

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from scripts.ear_deformation import bounded_surface_step, surface_quality
from scripts.surface_intersections import new_crossings


def _edges(faces):
    return np.unique(
        np.sort(
            np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            axis=1,
        ),
        axis=0,
    )


def continue_hair_transition(
    template_base, pre_ear_hair, current, faces, face_count, ear_regions
):
    """Return ``(candidate, audit)`` without changing any input array.

    All three surfaces must have identical vertex correspondence. ``faces``
    uses original vertex IDs; its first ``face_count`` triangles are measured
    face geometry. Ear regions contain ``vertices`` as in ``ear_fit``.

    Conservative experimental domain: posterior Z < -150 mm, original hair
    envelope weight < .98, and above the pinned bottom 8 mm of the neck cut.
    The 35 mm ear attachment support is measured on the pre-ear surface, just
    as in the ear regularizer, with two extra edge rings for its rotation stencil.
    A weighted graph biharmonic solve continues the
    hair displacement, retaining the current ear displacement. Components
    without a nonzero, full-envelope boundary donor are left unchanged.

    Existing area/orientation/intersection gates back off unsafe proposals.
    An unanchored or unchanged domain is an exact no-op. Invalid inputs raise
    ValueError; an unsolvable numerical system safely leaves current intact.
    """
    original = np.asarray(current)
    base, pre, now = [
        np.asarray(value, dtype=np.float64)
        for value in (template_base, pre_ear_hair, current)
    ]
    if (
        base.ndim != 2
        or base.shape[1:] != (3,)
        or pre.shape != base.shape
        or now.shape != base.shape
        or not all(np.isfinite(value).all() for value in (base, pre, now))
    ):
        raise ValueError('Hair transition requires corresponding finite Nx3 surfaces.')
    triangles = np.asarray(faces)
    if (
        triangles.ndim != 2
        or triangles.shape[1:] != (3,)
        or triangles.dtype.kind not in 'iu'
        or np.any(triangles < 0)
        or np.any(triangles >= len(base))
    ):
        raise ValueError('Hair transition requires valid integer triangle indices.')
    triangles = triangles.astype(np.int64, copy=False)
    if not isinstance(face_count, (int, np.integer)) or not 0 <= face_count <= len(
        triangles
    ):
        raise ValueError('Invalid measured face triangle count.')
    # Preserve caller precision and every fixed bit, including float32 inputs.
    dtype = original.dtype if original.dtype.kind == 'f' else np.dtype(np.float64)
    untouched = np.array(original, dtype=dtype, copy=True)
    audit = {
        'method': 'Pinned-envelope biharmonic continuation of posterior hair displacement',
        'estimatedGeometry': True,
        'applied': False,
        'changedVertices': 0,
        'maximumChangeMm': 0.0,
        'attachmentRadiusMm': 35,
        'earRotationBufferRings': 2,
        'neckCutMarginMm': 8,
        'fullEnvelopeWeightThreshold': 0.98,
        'posteriorDomainZ': -0.15,
        'fixedPositionsExact': True,
        'limitation': 'Smooth estimated nape transition; does not recover unseen neck anatomy.',
    }

    def no_op(reason):
        return untouched.copy(), {**audit, 'reason': reason}

    hair = pre - base
    if not np.any(np.linalg.norm(hair, axis=1) > 1e-8):
        return no_op('No fitted hair displacement.')
    if len(triangles) == 0 or len(base) <= 468:
        return no_op('No unprotected connected surface.')

    used = np.unique(triangles)
    cut = float(base[used, 1].min())
    cap = (np.ptp(base[triangles, 1], axis=1) < 1e-8) & (
        np.abs(base[triangles, 1].mean(axis=1) - cut) < 1e-7
    )
    edges = _edges(triangles[~cap])
    a, b = edges.T
    lengths = np.linalg.norm(base[a] - base[b], axis=1)
    weights = 1 / np.maximum(lengths, 0.001)
    adjacency = coo_matrix(
        (np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])),
        shape=(len(base), len(base)),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = diags(degree) - adjacency

    protected = np.zeros(len(base), bool)
    protected[:468] = True
    protected[np.unique(triangles[:face_count])] = True
    ears = np.zeros(len(base), bool)
    for region in (ear_regions or {}).values():
        vertices = np.asarray(region['vertices'])
        if vertices.size == 0:
            continue
        if (
            vertices.dtype.kind not in 'iu'
            or np.any(vertices < 0)
            or np.any(vertices >= len(base))
        ):
            raise ValueError('Invalid ear-region vertex indices.')
        ears[vertices] = True
    attachment = ears.copy()
    if ears.any():
        # Match the ear regularizer's full-topology geodesic support exactly.
        ea, eb = _edges(triangles).T
        distance = np.linalg.norm(pre[ea] - pre[eb], axis=1)
        graph = coo_matrix(
            (np.r_[distance, distance], (np.r_[ea, eb], np.r_[eb, ea])),
            shape=(len(base), len(base)),
        ).tocsr()
        attachment |= (
            dijkstra(
                graph,
                directed=False,
                indices=np.flatnonzero(ears),
                min_only=True,
                limit=0.035,
            )
            < 0.035
        )
        # ARAP uses rotations at both ends of each edge. A free ear vertex
        # therefore depends on the rest geometry two edges beyond its support.
        # Pin that stencil so a later ear refit cannot respond to this nape edit.
        for _ in range(2):
            touch = attachment[ea] | attachment[eb]
            attachment[np.unique(np.r_[ea[touch], eb[touch]])] = True
    # Also retain any meaningful legacy ear deformation beyond nominal support.
    protected |= attachment | (np.linalg.norm(now - pre, axis=1) > 1e-5)
    neck_cut = base[:, 1] < cut + 0.008
    protected |= neck_cut
    angle = np.abs(np.arctan2(base[:, 0], base[:, 2] + 0.09))
    threshold = np.interp(angle, [0, 1, 1.7, np.pi], [base[10, 1], 0.065, 0.06, -0.035])
    full_envelope = np.clip((base[:, 1] - threshold) / 0.018, 0, 1) >= 0.98
    free = (base[:, 2] < -0.15) & ~full_envelope & ~protected & (degree > 0)
    audit.update(
        capFacesExcluded=int(cap.sum()),
        earAttachmentVerticesPinned=int(attachment.sum()),
        fullEnvelopeVerticesPinned=int(full_envelope.sum()),
        neckCutVerticesPinned=int(neck_cut.sum()),
        candidateDomainVertices=int(free.sum()),
    )
    if not free.any():
        return no_op('No unprotected connected surface.')

    # Do not erase a displaced but unsupported island by solving against only
    # zero-displacement neck/face constraints. Require real envelope donors.
    ids = np.flatnonzero(free)
    _, component = connected_components(adjacency[ids][:, ids], directed=False)
    donors = full_envelope & (np.linalg.norm(hair, axis=1) > 1e-8)
    touches_donor = np.asarray(adjacency[ids][:, donors].sum(axis=1)).ravel() > 0
    supported = np.isin(component, np.unique(component[touches_donor]))
    free[ids[~supported]] = False
    audit['unsupportedDomainVertices'] = int((~supported).sum())
    if not free.any():
        return no_op('No connected full-envelope displacement donors.')

    ids, anchors = np.flatnonzero(free), np.flatnonzero(~free)
    audit.update(solvedVertices=len(ids), pinnedVertices=len(anchors))
    system = laplacian.T @ diags(1 / np.maximum(degree, 1)) @ laplacian
    rhs = -(system[ids][:, anchors] @ hair[anchors])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            solved = spsolve(system[ids][:, ids].tocsc(), rhs)
    except (MatrixRankWarning, RuntimeError, ValueError):
        return no_op('The anchored transition system could not be solved safely.')
    if not np.isfinite(solved).all():
        return no_op('The transition solve returned nonfinite displacement.')
    proposal = now.copy()
    proposal[ids] += solved - hair[ids]
    candidate, quality = bounded_surface_step(now, proposal, triangles)
    candidate = candidate.astype(dtype)
    candidate[anchors] = untouched[anchors]
    # Recheck the returned precision: a float32 caller must get the geometry
    # that passed the gates, including after a refined step near a threshold.
    quality.update(surface_quality(now, candidate.astype(float), triangles))
    quality.update(new_crossings(now, candidate.astype(float), triangles))
    if (
        not np.isfinite(candidate).all()
        or quality['minimumNormalAgreement'] <= 0.1
        or quality['minimumAreaRatio'] <= 0.25
        or quality['maximumAreaRatio'] >= 3
        or quality['newCrossings']
    ):
        return no_op('The transition failed geometry checks at output precision.')
    if not np.array_equal(candidate[anchors], untouched[anchors]):
        raise ValueError('Hair transition changed fixed geometry.')
    change = np.linalg.norm(candidate.astype(float) - now, axis=1)
    audit.update(
        applied=bool(np.any(change > 1e-8)),
        changedVertices=int(np.sum(change > 1e-8)),
        maximumChangeMm=float(change.max(initial=0) * 1000),
        surfaceQuality=quality,
        fullEnvelopeExact=bool(
            np.array_equal(candidate[full_envelope], untouched[full_envelope])
        ),
        protectedExact=bool(np.array_equal(candidate[protected], untouched[protected])),
        baselineUpdate='Add candidate minus current to the saved pre-ear surface.',
    )
    return candidate, audit
