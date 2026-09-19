"""Detect non-adjacent surface crossings near a proposed mesh deformation."""

import numpy as np
import trimesh


def crossing_pairs(points, faces, affected=None):
    triangles = points[faces]
    tree = trimesh.triangles.bounds_tree(triangles)
    bounds = np.column_stack((triangles.min(axis=1), triangles.max(axis=1)))
    candidates = set()
    for i in (range(len(faces)) if affected is None else np.flatnonzero(affected)):
        for j in tree.intersection(bounds[i]):
            if i == j:
                continue
            candidates.add((min(int(i), j), max(int(i), j)))
    if not candidates:
        return set()
    pairs = np.array(sorted(candidates))
    # Reject shared-vertex neighbors in one array operation. Calling np.isin
    # for every R-tree candidate dominated the entire ear quality gate.
    adjacent = np.any(
        faces[pairs[:, 0], :, None] == faces[pairs[:, 1], None, :], axis=(1, 2)
    )
    pairs = pairs[~adjacent]
    crossings = set()
    for begin in range(0, len(pairs), 65536):
        chunk = pairs[begin : begin + 65536]
        a, b = triangles[chunk[:, 0]], triangles[chunk[:, 1]]
        hit = np.zeros(len(chunk), bool)
        for source, target in ((a, b), (b, a)):
            e1 = target[:, 1] - target[:, 0]
            e2 = target[:, 2] - target[:, 0]
            for edge in range(3):
                origin = source[:, edge]
                direction = source[:, (edge + 1) % 3] - origin
                h = np.cross(direction, e2)
                det = np.sum(e1 * h, axis=1)
                valid = np.abs(det) > 1e-14
                inverse = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
                delta = origin - target[:, 0]
                u = np.sum(delta * h, axis=1) * inverse
                q = np.cross(delta, e1)
                v = np.sum(direction * q, axis=1) * inverse
                t = np.sum(e2 * q, axis=1) * inverse
                hit |= (
                    valid
                    & (u > 1e-6)
                    & (v > 1e-6)
                    & (u + v < 1 - 1e-6)
                    & (t > 1e-6)
                    & (t < 1 - 1e-6)
                )
        crossings.update(map(tuple, chunk[hit]))
    return crossings


def new_crossings(rest, candidate, faces):
    moved = np.linalg.norm(candidate - rest, axis=1) > 1e-7
    affected = moved[faces].any(1)
    before = crossing_pairs(rest, faces, affected)
    after = crossing_pairs(candidate, faces, affected)
    return {
        'existingCrossings': len(before),
        'newCrossings': len(after - before),
        'remainingCrossings': len(after),
    }
