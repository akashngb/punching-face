"""Geometry continuation preserves evidence anchors and rejects unsupported solves."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from scripts.hair_transition import continue_hair_transition


def fixture():
    # A connected posterior patch plus the pipeline's unused measured cage.
    xs, ys = np.linspace(-0.07, 0.07, 15), np.linspace(-0.12, 0.04, 21)
    xx, yy = np.meshgrid(xs, ys)
    patch_vertices = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, -0.22)]
    base = np.vstack([np.zeros((468, 3)), patch_vertices])
    base[10, 1] = 0.075
    faces = []
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            a = 468 + row * len(xs) + column
            b, c, d = a + 1, a + len(xs), a + len(xs) + 1
            faces.extend([[a, b, c], [b, d, c]])
    faces = np.asarray(faces)
    angle = np.abs(np.arctan2(base[:, 0], base[:, 2] + 0.09))
    threshold = np.interp(angle, [0, 1, 1.7, np.pi], [base[10, 1], 0.065, 0.06, -0.035])
    full = (base[:, 1] - threshold) / 0.018 >= 0.98
    full[:468] = False
    pre = base.copy()
    pre[full, 2] -= 0.018
    return base, pre, faces, full


def maximum_dihedral(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    owners = {}
    angles = []
    for index, triangle in enumerate(faces):
        for a, b in zip(triangle, np.roll(triangle, -1)):
            edge = tuple(sorted((a, b)))
            if edge in owners:
                cosine = np.clip(normals[index] @ normals[owners[edge]], -1, 1)
                angles.append(np.degrees(np.arccos(cosine)))
            else:
                owners[edge] = index
    return max(angles)


class HairTransitionTests(unittest.TestCase):
    def test_sharp_displacement_ramp_smooths_with_envelope_and_cut_exact(self):
        base, pre, faces, full = fixture()
        original = pre.copy()
        candidate, audit = continue_hair_transition(base, pre, pre, faces, 0, {})
        self.assertTrue(audit['applied'])
        self.assertEqual(audit['surfaceQuality']['appliedFraction'], 1)
        self.assertLess(
            maximum_dihedral(candidate, faces), maximum_dihedral(pre, faces) * 0.8
        )
        np.testing.assert_array_equal(candidate[full], pre[full])
        np.testing.assert_array_equal(candidate[:468], pre[:468])
        cut = base[:, 1] < -0.112
        np.testing.assert_array_equal(candidate[cut], pre[cut])
        self.assertEqual(audit['surfaceQuality']['newCrossings'], 0)
        self.assertEqual(audit['surfaceQuality']['reversedTriangles'], 0)
        self.assertTrue(np.isfinite(candidate).all())
        np.testing.assert_array_equal(pre, original)
        repeated, second = continue_hair_transition(base, pre, pre, faces, 0, {})
        np.testing.assert_array_equal(candidate, repeated)
        self.assertEqual(audit, second)

    def test_face_cage_ear_and_actual_geodesic_attachment_support_are_pinned(self):
        base, pre, faces, _ = fixture()
        # Make a lower posterior interior triangle a measured face constraint.
        selected = 2 * (7 * 14 + 12)
        faces = np.concatenate(
            [faces[selected : selected + 1], np.delete(faces, selected, axis=0)]
        )
        ear = 468 + 10 * 15
        current = pre.copy()
        current[ear] += [0.001, 0.002, 0]
        # A legacy deformation outside the nominal support must also stay put.
        legacy = 468 + 9 * 15 + 10
        current[legacy, 0] += 0.0005
        candidate, audit = continue_hair_transition(
            base, pre, current, faces, 1, {'-1': {'vertices': [ear]}}
        )
        edges = np.unique(
            np.sort(
                np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                axis=1,
            ),
            axis=0,
        )
        a, b = edges.T
        length = np.linalg.norm(pre[a] - pre[b], axis=1)
        graph = coo_matrix(
            (np.r_[length, length], (np.r_[a, b], np.r_[b, a])),
            shape=(len(pre), len(pre)),
        ).tocsr()
        distance = dijkstra(
            graph, directed=False, indices=[ear], min_only=True, limit=0.035
        )
        support = distance < 0.035
        self.assertGreater(support.sum(), 10)
        rotation_stencil = support.copy()
        for _ in range(2):
            touch = rotation_stencil[a] | rotation_stencil[b]
            rotation_stencil[np.unique(edges[touch])] = True
        self.assertGreater(rotation_stencil.sum(), support.sum())
        support = rotation_stencil
        self.assertEqual(audit['earAttachmentVerticesPinned'], support.sum())
        np.testing.assert_array_equal(candidate[support], current[support])
        np.testing.assert_array_equal(candidate[faces[0]], current[faces[0]])
        np.testing.assert_array_equal(candidate[:468], current[:468])
        np.testing.assert_array_equal(candidate[legacy], current[legacy])
        self.assertTrue(audit['applied'])

    def test_later_ear_regularization_keeps_the_corrected_surface_exact(self):
        from scripts.ear_deformation import regularize

        base, pre, faces, _ = fixture()
        ear = 468 + 10 * 15
        regions = {'-1': {'vertices': [ear], 'weights': [1], 'anchors': {'top': ear}}}
        protected = np.arange(len(pre)) < 468
        desired = pre.copy()
        desired[ear] += [0.001, 0, 0.005]
        current, _ = regularize(pre, desired, faces, regions, protected)
        candidate, audit = continue_hair_transition(
            base, pre, current, faces, 0, regions
        )
        self.assertTrue(audit['applied'])
        shift = candidate - current
        revised_base = pre + shift
        refitted, _ = regularize(
            revised_base, desired + shift, faces, regions, protected
        )
        np.testing.assert_array_equal(
            refitted.astype(np.float32), candidate.astype(np.float32)
        )

    def test_no_hair_displacement_and_missing_boundary_donors_are_exact_noops(self):
        base, pre, faces, full = fixture()
        unchanged, audit = continue_hair_transition(base, base, base, faces, 0, {})
        np.testing.assert_array_equal(unchanged, base)
        self.assertFalse(audit['applied'])
        self.assertIn('No fitted hair', audit['reason'])
        # Nonzero interior displacement alone is not an observed envelope anchor.
        unsupported = base.copy()
        interior = (
            (base[:, 1] > -0.09) & (base[:, 1] < -0.06) & (np.arange(len(base)) >= 468)
        )
        unsupported[interior, 2] -= 0.004
        unchanged, audit = continue_hair_transition(
            base, unsupported, unsupported, faces, 0, {}
        )
        np.testing.assert_array_equal(unchanged, unsupported)
        self.assertFalse(audit['applied'])
        self.assertIn('No connected full-envelope', audit['reason'])

    def test_disconnected_unanchored_patch_is_not_erased_by_nearby_hair(self):
        base, pre, faces, _ = fixture()
        island = np.array(
            [[0, -0.075, -0.2201], [0.008, -0.075, -0.2201], [0, -0.065, -0.2201]]
        )
        n = len(base)
        base = np.vstack([base, island])
        pre = np.vstack([pre, island + [0, 0, -0.002]])
        faces = np.vstack([faces, np.array([[n, n + 1, n + 2]])])
        candidate, audit = continue_hair_transition(base, pre, pre, faces, 0, {})
        self.assertTrue(audit['applied'])
        np.testing.assert_array_equal(candidate[n:], pre[n:])
        self.assertEqual(audit['unsupportedDomainVertices'], 3)

    def test_cap_diagonals_do_not_shortcut_the_solve(self):
        base, pre, faces, _ = fixture()
        # Coplanar artificial cap faces may connect distant cut vertices, but
        # must contribute no displacement graph edge or biharmonic penalty.
        center = len(base)
        base = np.vstack([base, [[0, -0.12, -0.18]]])
        pre = np.vstack([pre, base[-1:]])
        with_cap = np.vstack([faces, [[468, 482, center], [482, 477, center]]])
        expected, _ = continue_hair_transition(base, pre, pre, faces, 0, {})
        candidate, audit = continue_hair_transition(base, pre, pre, with_cap, 0, {})
        np.testing.assert_array_equal(candidate, expected)
        self.assertEqual(audit['capFacesExcluded'], 2)

    def test_float32_fixed_bits_and_nonfinite_solve_fail_closed(self):
        base, pre, faces, full = fixture()
        current = pre.astype(np.float32)
        candidate, audit = continue_hair_transition(base, pre, current, faces, 0, {})
        self.assertEqual(candidate.dtype, np.float32)
        np.testing.assert_array_equal(candidate[full], current[full])
        self.assertTrue(audit['fixedPositionsExact'])
        self.assertEqual(audit['surfaceQuality']['newCrossings'], 0)
        with patch(
            'scripts.hair_transition.spsolve',
            side_effect=lambda matrix, rhs: np.full(rhs.shape, np.nan),
        ):
            failed, audit = continue_hair_transition(base, pre, current, faces, 0, {})
        np.testing.assert_array_equal(failed, current)
        self.assertFalse(audit['applied'])
        self.assertIn('nonfinite', audit['reason'])

    def test_finite_but_unsafe_displacement_is_limited_by_geometry_gates(self):
        base, pre, faces, full = fixture()

        def unsafe_solution(matrix, rhs):
            result = np.zeros_like(rhs)
            result[::2, 2] = 10
            result[1::2, 2] = -10
            return result

        with patch('scripts.hair_transition.spsolve', side_effect=unsafe_solution):
            candidate, audit = continue_hair_transition(base, pre, pre, faces, 0, {})
        quality = audit['surfaceQuality']
        self.assertLess(quality['appliedFraction'], 1)
        self.assertGreater(quality['minimumNormalAgreement'], 0.1)
        self.assertGreater(quality['minimumAreaRatio'], 0.25)
        self.assertLess(quality['maximumAreaRatio'], 3)
        self.assertEqual(quality['newCrossings'], 0)
        np.testing.assert_array_equal(candidate[full], pre[full])

    def test_invalid_correspondence_is_rejected(self):
        base, pre, faces, _ = fixture()
        with self.assertRaises(ValueError):
            continue_hair_transition(base[:-1], pre, pre, faces, 0, {})
        bad = pre.copy()
        bad[500, 0] = np.nan
        with self.assertRaises(ValueError):
            continue_hair_transition(base, bad, pre, faces, 0, {})


if __name__ == '__main__':
    unittest.main()
