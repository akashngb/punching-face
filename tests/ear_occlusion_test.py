"""Conservative, graph-local continuation behind photographed ear occluders."""

import unittest
import numpy as np

from scripts.ear_occlusion import continue_ear_occlusion


def strip(columns=9):
    vertices = np.array(
        [[x * 0.004, y * 0.003, 0.0] for x in range(columns) for y in range(2)]
    )
    faces = np.array(
        [[2 * x, 2 * x + 2, 2 * x + 1] for x in range(columns - 1)]
        + [[2 * x + 1, 2 * x + 2, 2 * x + 3] for x in range(columns - 1)]
    )
    tids, weights = [], []
    for vertex in range(len(vertices)):
        triangle = np.flatnonzero(np.any(faces == vertex, axis=1))[0]
        tids.append(triangle)
        weights.append((faces[triangle] == vertex).astype(float))
    return (
        vertices,
        faces,
        dict(
            triangles=faces.copy(),
            triangleIds=np.array(tids),
            weights=np.array(weights),
        ),
    )


def inputs(columns=9):
    p, f, b = strip(columns)
    n = len(p)
    color = np.tile([0.02, 0.03, 0.04], (n, 1))
    color[:2], color[-2:] = [0.2, 0.3, 0.4], [0.8, 0.6, 0.5]
    confidence = np.zeros(n)
    confidence[:2] = confidence[-2:] = 0.8
    photographed = confidence > 0.15
    return dict(
        vertices=p,
        faces=f,
        binding=b,
        photo_color=color,
        parts=np.zeros(n, int),
        vertex_parts=np.zeros(n, int),
        confidence=confidence,
        ear_occluded=np.ones(n),
        photographed=photographed,
    )


def long_triangle(refined=False, source_x=0.0):
    p = np.array([[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.02, 0.0]])
    f = np.array([[0, 1, 2]])
    if refined:
        p = np.r_[p, [[0.02, 0.0, 0.0]]]
        f = np.array([[0, 3, 2], [3, 1, 2]])
    # A photograph and a dense atlas ray along the same 40 mm mesh edge.
    xs = np.r_[source_x, np.linspace(0.001, 0.039, 191)]
    tids = np.where((xs > 0.02) & refined, 1, 0)
    weights = []
    for x, tid in zip(xs, tids):
        q = p[f[tid]]
        weights.append(
            np.linalg.solve(np.r_[q[:, :2].T, np.ones((1, 3))], [x, 0.0, 1.0])
        )
    n = len(xs)
    color = np.tile([0.02, 0.03, 0.04], (n, 1))
    color[0] = [0.6, 0.4, 0.2]
    confidence = np.zeros(n)
    confidence[0] = 0.8
    return xs, dict(
        vertices=p,
        faces=f,
        binding=dict(triangles=f.copy(), triangleIds=tids, weights=np.array(weights)),
        photo_color=color,
        parts=np.zeros(n, int),
        vertex_parts=np.zeros(len(p), int),
        confidence=confidence,
        ear_occluded=np.ones(n),
        photographed=confidence > 0.15,
    )


class EarOcclusionTests(unittest.TestCase):
    def test_smooth_bounded_deterministic_fill_and_exact_photographs(self):
        args = inputs()
        estimate, alpha, ownership, audit = continue_ear_occlusion(**args)
        again = continue_ear_occlusion(**args)
        for a, b in zip((estimate, alpha, ownership), again[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(estimate).all())
        self.assertTrue(np.all(alpha[2:-2] == 1))
        self.assertTrue(np.all(ownership[2:-2] == 1))
        self.assertTrue(np.all(np.diff(estimate[::2, 0]) >= -1e-12))
        self.assertGreater(estimate[10, 0] - estimate[6, 0], 0.1)
        self.assertTrue(np.all(estimate[2:-2] >= np.array([0.2, 0.3, 0.4]) - 1e-12))
        self.assertTrue(np.all(estimate[2:-2] <= np.array([0.8, 0.6, 0.5]) + 1e-12))
        np.testing.assert_array_equal(
            estimate[[0, 1, -2, -1]], args['photo_color'][[0, 1, -2, -1]]
        )
        np.testing.assert_array_equal(alpha[[0, 1, -2, -1]], 0)
        np.testing.assert_array_equal(ownership[[0, 1, -2, -1]], 0)
        self.assertTrue(audit['estimated'])
        self.assertFalse(audit['physicalCoverageChanged'])

    def test_protected_hair_cleanup_eye_and_ear_targets_stay_exact(self):
        args = inputs()
        n = len(args['parts'])
        args['observed_hair'] = np.arange(n) == 4
        args['preserve'] = np.isin(np.arange(n), [6, 7])  # cleanup/mouth/bottom
        args['parts'][8], args['parts'][10] = 1, 3
        args['confidence'][12] = 0.12
        estimate, alpha, ownership, _ = continue_ear_occlusion(**args)
        protected = [4, 6, 7, 8, 10, 12]
        np.testing.assert_array_equal(
            estimate[protected], args['photo_color'][protected]
        )
        np.testing.assert_array_equal(alpha[protected], 0)
        np.testing.assert_array_equal(ownership[protected], 0)

    def test_ear_donors_and_ear_graph_bridge_cannot_bleed(self):
        args = inputs()
        args['parts'][:2] = 3
        args['photo_color'][:2] = [1, 0, 0]
        estimate, alpha, _, _ = continue_ear_occlusion(**args)
        # The remaining real head donor is constant; red ear RGB never anchors.
        np.testing.assert_allclose(
            estimate[alpha > 0], np.tile([0.8, 0.6, 0.5], (np.count_nonzero(alpha), 1))
        )
        args = inputs()
        args['confidence'][-2:] = 0
        args['photographed'][-2:] = False
        args['vertex_parts'][8:10] = 3  # a labelled auricle separates the graph
        _, alpha, ownership, _ = continue_ear_occlusion(**args)
        np.testing.assert_array_equal(alpha[10:], 0)
        np.testing.assert_array_equal(ownership[10:], 0)

    def test_disconnected_nearby_surface_has_no_donors(self):
        args = inputs(5)
        p, f, b = args['vertices'], args['faces'], args['binding']
        n, nf = len(p), len(f)
        args['vertices'] = np.r_[p, p + [0, 0, 0.00001]]
        args['faces'] = np.r_[f, f + n]
        args['binding'] = dict(
            triangles=args['faces'],
            triangleIds=np.r_[b['triangleIds'], b['triangleIds'] + nf],
            weights=np.r_[b['weights'], b['weights']],
        )
        for name in [
            'photo_color',
            'parts',
            'vertex_parts',
            'confidence',
            'ear_occluded',
            'photographed',
        ]:
            args[name] = np.concatenate([args[name], args[name]])
        args['confidence'][n:] = 0
        args['photographed'][n:] = False
        estimate, alpha, ownership, _ = continue_ear_occlusion(**args)
        np.testing.assert_array_equal(estimate[n:], args['photo_color'][n:])
        np.testing.assert_array_equal(alpha[n:], 0)
        np.testing.assert_array_equal(ownership[n:], 0)

    def test_absent_or_generated_only_anchors_retain_fallback(self):
        args = inputs()
        args['photographed'][:] = False
        estimate, alpha, ownership, audit = continue_ear_occlusion(**args)
        np.testing.assert_array_equal(estimate, args['photo_color'])
        np.testing.assert_array_equal(alpha, 0)
        np.testing.assert_array_equal(ownership, 0)
        self.assertEqual(audit['donorTexels'], 0)

    def test_folded_close_surface_cannot_bypass_graph_radius(self):
        args = inputs(20)
        angle = np.repeat(np.linspace(0, 2 * np.pi - 0.1, 20), 2)
        args['vertices'][:, 0] = 0.012 * np.cos(angle)
        args['vertices'][:, 2] = 0.012 * np.sin(angle)
        args['confidence'][-2:] = 0
        args['photographed'][-2:] = False
        args['ear_occluded'][:] = 0
        args['ear_occluded'][-2:] = 1
        _, alpha, ownership, _ = continue_ear_occlusion(**args)
        self.assertLess(
            np.linalg.norm(args['vertices'][0] - args['vertices'][-2]), 0.002
        )
        np.testing.assert_array_equal(alpha, 0)
        np.testing.assert_array_equal(ownership, 0)

    def test_fractional_ownership_replaces_old_fallback_only_once(self):
        args = inputs()
        args['photo_color'][:] = [0.4, 0.3, 0.2]
        args['confidence'][2:-2] = 0.06
        args['ear_occluded'][:] = 0.4
        estimate, alpha, ownership, _ = continue_ear_occlusion(**args)
        old_fallback = np.full_like(estimate, 0.9)
        result = old_fallback + ownership[:, None] * (
            args['photo_color'] - old_fallback
        )
        result += alpha[:, None] * (estimate - args['photo_color'])
        np.testing.assert_allclose(ownership[2:-2], 0.4)
        np.testing.assert_allclose(alpha[2:-2], 0.2)
        expected = 0.6 * 0.9 + 0.4 * np.array([0.4, 0.3, 0.2])
        np.testing.assert_allclose(
            result[2:-2], np.tile(expected, (len(result) - 4, 1))
        )

    def test_long_target_chain_cannot_transmit_remote_donor_color(self):
        args = inputs(25)
        # Leave the entire connected strip eligible. A union-neighbourhood
        # harmonic solve would let the far red endpoint affect the near end.
        args['confidence'][8:10] = 0.8
        args['photographed'][8:10] = True
        args['photo_color'][8:10] = [0.2, 0.3, 0.4]
        first = continue_ear_occlusion(**args)
        args['photo_color'][-2:] = [1.0, 0.0, 0.0]
        second = continue_ear_occlusion(**args)
        np.testing.assert_array_equal(first[0][2:8], second[0][2:8])
        self.assertLess(second[3]['maximumUsedSurfacePathMm'], 20.0)
        # The middle of this long strip has no donor within the radius.
        np.testing.assert_array_equal(second[2][22:28], 0)

    def test_long_triangle_resolves_near_anchor_and_fades_before_far_corner(self):
        xs, args = long_triangle()
        estimate, alpha, ownership, audit = continue_ear_occlusion(**args)
        near = (xs < 0.016) & (xs > 0)
        far = xs >= 0.020
        np.testing.assert_allclose(ownership[near], 1)
        np.testing.assert_array_equal(ownership[far], 0)
        np.testing.assert_allclose(
            estimate[alpha > 0], np.tile([0.6, 0.4, 0.2], (np.count_nonzero(alpha), 1))
        )
        self.assertLess(audit['maximumUsedSurfacePathMm'], 20)
        self.assertEqual(audit['ownershipReachFadeMm'], 4)
        transition = (xs >= 0.016) & (xs <= 0.020)
        d = (20 - xs[transition] * 1000) / 4
        np.testing.assert_allclose(
            ownership[transition], d * d * (3 - 2 * d), atol=1e-12
        )
        self.assertLess(np.max(np.abs(np.diff(ownership[1:]))), 0.076)
        np.testing.assert_array_equal(estimate[0], args['photo_color'][0])
        self.assertEqual(ownership[0], 0)

    def test_collinear_edge_subdivision_and_atlas_sampling_preserve_fade(self):
        _, coarse = long_triangle()
        _, fine = long_triangle(refined=True)
        a, b = continue_ear_occlusion(**coarse), continue_ear_occlusion(**fine)
        for left, right in zip(a[:3], b[:3]):
            np.testing.assert_allclose(left, right, atol=1e-12)
        # Changing atlas sample density cannot change an existing texel.
        small = dict(coarse)
        keep = np.arange(0, len(coarse['confidence']), 2)
        for key in [
            'photo_color',
            'parts',
            'confidence',
            'ear_occluded',
            'photographed',
        ]:
            small[key] = coarse[key][keep]
        small['binding'] = dict(coarse['binding'])
        for key in ['triangleIds', 'weights']:
            small['binding'][key] = coarse['binding'][key][keep]
        sparse = continue_ear_occlusion(**small)
        for full, reduced in zip(a[:3], sparse[:3]):
            np.testing.assert_allclose(full[keep], reduced, atol=1e-12)

    def test_long_donor_face_cannot_relocate_distant_photo_to_near_vertex(self):
        xs, args = long_triangle(source_x=0.030)
        _, alpha, ownership, audit = continue_ear_occlusion(**args)
        # The source has nonzero barycentric weight at vertex 0, but is 30 mm
        # away. That vertex cannot become a nearby 0-distance color anchor.
        np.testing.assert_array_equal(ownership[(xs < 0.010) & (xs > 0)], 0)
        np.testing.assert_array_equal(alpha[(xs < 0.010) & (xs > 0)], 0)
        self.assertGreater(audit['maximumDonorFootprintMm'], 0)

    def test_planar_cap_and_invalid_binding_cannot_create_shortcuts(self):
        args = inputs()
        args['vertices'][:, 1] = 0
        _, alpha, ownership, audit = continue_ear_occlusion(**args)
        np.testing.assert_array_equal(alpha, 0)
        np.testing.assert_array_equal(ownership, 0)
        self.assertEqual(audit['excludedCapTriangles'], len(args['faces']))
        args = inputs()
        args['binding']['triangleIds'][0] = -1
        with self.assertRaises(ValueError):
            continue_ear_occlusion(**args)


if __name__ == '__main__':
    unittest.main()
