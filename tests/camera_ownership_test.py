"""Ownership diffusion preserves evidence and cannot cross missing topology."""

import unittest
import numpy as np
from scripts.camera_ownership import (
    fit_camera_ownership_delta,
    apply_camera_ownership_delta,
)


class CameraOwnershipTests(unittest.TestCase):
    def fixture(self):
        vertices = np.array(
            [[0, 0, 0], [0.003, 0, 0], [0, 0.003, 0], [0.003, 0.003, 0]]
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        binding = {
            'triangles': faces,
            'triangleIds': np.array([0, 0, 0, 1]),
            'weights': np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 1.0, 0]]),
        }
        probabilities = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        return vertices, faces, binding, probabilities

    def test_local_seam_reduces_without_recoloring_or_mean_tone_drift(self):
        vertices, faces, binding, probabilities = self.fixture()
        original = probabilities.copy()
        delta, audit = fit_camera_ownership_delta(
            vertices, faces, binding, probabilities
        )
        result, _ = apply_camera_ownership_delta(probabilities, binding, delta)
        # These camera RGB values are evidence, never parameters of the helper.
        rgb = np.array([[0.7, 0.5, 0.4], [0.4, 0.3, 0.2]])
        before, after = probabilities @ rgb, result @ rgb
        self.assertLess(
            np.linalg.norm(after[0] - after[1]), np.linalg.norm(before[0] - before[1])
        )
        np.testing.assert_allclose(after.mean(axis=0), before.mean(axis=0), atol=1e-15)
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-15)
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        np.testing.assert_array_equal(probabilities, original)
        self.assertFalse(audit['sourceRGBChanged'])
        self.assertFalse(audit['physicalConfidenceChanged'])

    def test_zero_diffusion_and_zero_blend_are_exact_despite_coarse_binding(self):
        vertices, faces, binding, probabilities = self.fixture()
        delta, _ = fit_camera_ownership_delta(
            vertices, faces, binding, probabilities, steps=0
        )
        np.testing.assert_array_equal(delta, 0)
        coarse = {**binding, 'weights': np.full((4, 3), 1 / 3)}
        result, _ = apply_camera_ownership_delta(probabilities, coarse, delta)
        np.testing.assert_array_equal(result, probabilities)
        nonzero, _ = fit_camera_ownership_delta(vertices, faces, binding, probabilities)
        result, _ = apply_camera_ownership_delta(
            probabilities.astype(np.float32), coarse, nonzero, blend=0
        )
        np.testing.assert_array_equal(result, probabilities.astype(np.float32))

    def test_pinned_domain_and_ineligible_samples_have_zero_delta(self):
        vertices, faces, binding, probabilities = self.fixture()
        domain = np.array([False, True, True, True])
        eligible = np.array([True, True, True, False])
        delta, audit = fit_camera_ownership_delta(
            vertices,
            faces,
            binding,
            probabilities,
            sample_eligible=eligible,
            vertex_domain=domain,
        )
        np.testing.assert_array_equal(delta[[0, 3]], 0)
        self.assertEqual(audit['pinnedVertices'], 1)
        self.assertEqual(audit['unsupportedVertices'], 1)
        result, _ = apply_camera_ownership_delta(
            probabilities, binding, delta, blend=[0, 1, 1, 0]
        )
        np.testing.assert_array_equal(result[[0, 3]], probabilities[[0, 3]])
        self.assertTrue(np.any(result[1:3] != probabilities[1:3]))

    def test_no_long_edge_or_unsupported_vertex_bridge(self):
        vertices = np.array([[0, 0, 0], [0.005, 0, 0], [0.01, 0, 0]])
        faces = np.array([[0, 1, 2]])
        binding = {
            'triangles': faces,
            'triangleIds': np.zeros(3, int),
            'weights': np.eye(3),
        }
        p = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        delta, audit = fit_camera_ownership_delta(
            vertices, faces, binding, p, sample_eligible=[True, False, True]
        )
        np.testing.assert_array_equal(delta, 0)
        self.assertEqual(audit['diffusionEdges'], 0)

    def test_nearby_disconnected_surface_cannot_donate(self):
        vertices, faces, binding, probabilities = self.fixture()
        vertices = np.r_[vertices, vertices[:3] + [0, 0, 0.0001]]
        faces = np.r_[faces, [[4, 5, 6]]]
        binding = {
            'triangles': faces,
            'triangleIds': np.r_[binding['triangleIds'], [2, 2, 2]],
            'weights': np.r_[binding['weights'], np.eye(3)],
        }
        probabilities = np.r_[probabilities, np.tile([0.3, 0.7], (3, 1))]
        delta, _ = fit_camera_ownership_delta(vertices, faces, binding, probabilities)
        np.testing.assert_array_equal(delta[4:], 0)
        self.assertTrue(np.any(delta[:4] != 0))

    def test_original_support_mask_and_low_retained_mass_retreat(self):
        binding = {
            'triangles': np.array([[0, 1, 2]]),
            'triangleIds': np.zeros(3, int),
            'weights': np.eye(3),
        }
        p = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]])
        delta = np.array([[-0.4, -0.4, 0.8], [-0.1, 0, 0.1], [-0.02, 0.01, 0.01]])
        result, audit = apply_camera_ownership_delta(
            p, binding, delta, support_mask=p > 0
        )
        np.testing.assert_array_equal(result[0], p[0])  # Most mass was rejected.
        np.testing.assert_array_equal(result[:, 2], 0)
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-15)
        # Retained 90%: partial, smooth retreat between no change and full result.
        full = np.array([0.4, 0.5]) / 0.9
        self.assertGreater(result[1, 0], full[0])
        self.assertLess(result[1, 0], p[1, 0])
        self.assertEqual(audit['retreatedTexels'], 2)
        self.assertGreater(audit['changedTexels'], 0)
        with self.assertRaisesRegex(ValueError, 'rejects an existing'):
            apply_camera_ownership_delta(
                p, binding, delta, support_mask=np.zeros_like(p, bool)
            )

    def test_empty_evidence_and_float32_border_weights_are_safe(self):
        vertices, faces, binding, probabilities = self.fixture()
        empty, _ = fit_camera_ownership_delta(
            vertices, faces, binding, np.zeros_like(probabilities)
        )
        np.testing.assert_array_equal(empty, 0)
        border = {**binding, 'weights': binding['weights'].astype(np.float32)}
        border['weights'][0] = [1.00001, -1e-5, 0]
        a, _ = fit_camera_ownership_delta(vertices, faces, border, probabilities)
        b, _ = fit_camera_ownership_delta(vertices, faces, border, probabilities)
        np.testing.assert_array_equal(a, b)
        result, _ = apply_camera_ownership_delta(probabilities, border, a)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-15)
        unsupported, _ = apply_camera_ownership_delta(
            np.zeros_like(probabilities),
            border,
            a,
            support_mask=np.ones_like(probabilities, bool),
        )
        np.testing.assert_array_equal(unsupported, 0)

    def test_chunk_boundaries_preserve_camera_probabilities(self):
        vertices, faces, binding, probabilities = self.fixture()
        repeat = 4100  # Cross the 16,384-sample accumulation/application chunk.
        many = np.tile(probabilities, (repeat, 1))
        repeated = {
            'triangles': faces,
            'triangleIds': np.tile(binding['triangleIds'], repeat),
            'weights': np.tile(binding['weights'], (repeat, 1)),
        }
        expected, _ = fit_camera_ownership_delta(
            vertices, faces, binding, probabilities
        )
        delta, audit = fit_camera_ownership_delta(vertices, faces, repeated, many)
        np.testing.assert_allclose(delta, expected, atol=1e-13)
        self.assertEqual(audit['sourceSamples'], len(many))
        result, _ = apply_camera_ownership_delta(many, repeated, delta)
        np.testing.assert_allclose(result[:4], result[-4:], atol=0)
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
