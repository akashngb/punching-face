"""Topology-constrained continuation preserves measured colors and bounds."""

import unittest
import numpy as np
from scripts.surface_completion import complete_surface_colors


def strip(width=9, height=3):
    vertices = np.array([[x, y, 0.0] for y in range(height) for x in range(width)])
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            a = y * width + x
            faces.extend([[a, a + 1, a + width], [a + 1, a + width + 1, a + width]])
    return vertices, np.asarray(faces)


def vertex_samples(faces, anchors, colors):
    ids, weights = [], []
    for anchor in anchors:
        triangle, corner = np.argwhere(faces == anchor)[0]
        ids.append(triangle)
        weight = np.zeros(3)
        weight[corner] = 1
        weights.append(weight)
    return {
        'triangles': faces,
        'triangleIds': np.asarray(ids),
        'weights': np.asarray(weights, np.float32),
    }, np.asarray(colors, float)


class SurfaceCompletionTests(unittest.TestCase):
    def test_large_missing_strip_is_smooth_and_anchor_colors_are_exact(self):
        vertices, faces = strip()
        anchors = [0, 9, 18, 8, 17, 26]
        colors = [[0.8, 0.2, 0.1]] * 3 + [[0.1, 0.4, 0.7]] * 3
        binding, colors = vertex_samples(faces, anchors, colors)
        # Observed anchors can sit just outside the continuation domain.
        domain = np.ones(len(vertices), bool)
        domain[anchors] = False
        args = (vertices, faces, binding, colors, np.ones(6, bool), domain)
        result, resolved, audit = complete_surface_colors(*args)
        repeated, repeated_mask, repeated_audit = complete_surface_colors(*args)
        self.assertTrue(resolved.all())
        np.testing.assert_array_equal(result[anchors], colors)
        np.testing.assert_array_equal(result, repeated)
        np.testing.assert_array_equal(resolved, repeated_mask)
        self.assertEqual(audit, repeated_audit)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result >= colors.min(axis=0)))
        self.assertTrue(np.all(result <= colors.max(axis=0)))
        middle = result[9:18, 0]
        self.assertTrue(np.all(np.diff(middle) < 0))
        self.assertLess(np.max(np.abs(np.diff(middle))), 0.15)
        self.assertGreater(middle[4], 0.3)
        self.assertLess(middle[4], 0.6)
        self.assertEqual(audit['solvedVertices'], 21)
        self.assertLess(audit['maximumEquationResidual'], 1e-10)

    def test_nearby_disconnected_surface_stays_unresolved_and_isolated_is_ignored(self):
        vertices, faces = strip(3, 2)
        count = len(vertices)
        vertices = np.vstack((vertices, vertices + [0, 0, 0.00001], [100, 100, 100]))
        faces = np.vstack((faces, faces + count))
        binding, colors = vertex_samples(faces, [0], [[0.2, 0.3, 0.4]])
        result, resolved, audit = complete_surface_colors(
            vertices, faces, binding, colors, [True], np.ones(len(vertices), bool)
        )
        self.assertTrue(resolved[:count].all())
        self.assertFalse(resolved[count:].any())
        np.testing.assert_allclose(result[:count], np.tile(colors[0], (count, 1)))
        self.assertEqual(audit['unanchoredComponents'], 1)
        self.assertEqual(audit['ignoredIsolatedVertices'], 1)

    def test_outside_domain_unknown_vertices_cannot_bridge_a_gap(self):
        vertices, faces = strip()
        binding, colors = vertex_samples(faces, [0], [[0.2, 0.3, 0.4]])
        domain = vertices[:, 0] != 4
        _, resolved, _ = complete_surface_colors(
            vertices, faces, binding, colors, [True], domain
        )
        self.assertTrue(resolved[vertices[:, 0] < 4].all())
        self.assertFalse(resolved[vertices[:, 0] >= 4].any())

    def test_float32_border_weights_are_clamped_and_renormalized(self):
        vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        binding = {
            'triangles': faces,
            'triangleIds': np.array([0, 0]),
            'weights': np.array([[-1e-5, 0.25, 0.75001], [0, 1, 0]], np.float32),
        }
        colors = np.array([[0.8, 0.2, 0.1], [0.1, 0.4, 0.7]])
        result, resolved, audit = complete_surface_colors(
            vertices, faces, binding, colors, [True, True], np.ones(3, bool)
        )
        first_weight = float(binding['weights'][0, 1]) / sum(
            map(float, binding['weights'][0, 1:])
        )
        expected = (colors[0] * first_weight + colors[1]) / (first_weight + 1)
        np.testing.assert_allclose(result[1], expected)
        np.testing.assert_array_equal(result[2], colors[0])
        np.testing.assert_allclose(result[0], (result[1] + result[2]) / 2)
        self.assertTrue(resolved.all())
        self.assertEqual(audit['anchoredVertices'], 2)

    def test_chunk_boundary_sources_are_used_and_untrusted_colors_are_ignored(self):
        vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        count = 65537
        colors = np.full((count, 3), np.nan)
        weights = np.full((count, 3), np.nan, np.float32)
        colors[0], colors[-1] = [0.2, 0.4, 0.6], [0.6, 0.4, 0.2]
        weights[0], weights[-1] = [1, 0, 0], [0, 0, 1]
        source = np.zeros(count, bool)
        source[[0, -1]] = True
        binding = {
            'triangles': faces,
            'triangleIds': np.zeros(count, np.int32),
            'weights': weights,
        }
        result, resolved, audit = complete_surface_colors(
            vertices, faces, binding, colors, source, np.ones(3, bool)
        )
        np.testing.assert_array_equal(result[[0, 2]], colors[[0, -1]])
        np.testing.assert_allclose(result[1], [0.4, 0.4, 0.4])
        self.assertTrue(resolved.all())
        self.assertEqual(audit['sourceTexels'], 2)


if __name__ == '__main__':
    unittest.main()
