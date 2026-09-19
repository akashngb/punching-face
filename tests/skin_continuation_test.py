"""Resolved lower skin gets one fallback blend, with bounded local ownership."""

import unittest
from unittest.mock import patch
import numpy as np
from scripts.skin_continuation import continue_lower_skin, continue_ear_skin


def smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


class SkinContinuationTests(unittest.TestCase):
    def test_unseen_back_of_ear_uses_connected_skin_and_preserves_front(self):
        # The target faces away from every donor; spatial/normal fill alone
        # cannot reach it. An adjacent but disconnected dark triangle cannot
        # contribute to continuation around the ear surface.
        vertices = np.array(
            [
                [0, 0, 0],
                [0.01, 0, 0],
                [0, 0.01, 0],
                [0.01, 0.01, -0.002],
                [0, 0, -0.001],
                [0.01, 0, -0.001],
                [0, 0.01, -0.001],
            ]
        )
        faces = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]])
        weights = np.vstack(
            [np.tile([0.4, 0.3, 0.3], (9, 1)), [0, 1, 0], [0.3, 0.3, 0.4]]
        )
        tri = np.r_[np.zeros(9, int), 1, 2]
        binding = {'triangles': faces, 'triangleIds': tri, 'weights': weights}
        points = np.sum(vertices[faces[tri]] * weights[:, :, None], axis=1)
        skin = np.array([0.5, 0.3, 0.2])
        colors = np.vstack(
            [np.tile(skin, (9, 1)), [0.01, 0.01, 0.01], [0.05, 0.05, 0.05]]
        )
        normals = np.tile([0.0, 0, 1], (11, 1))
        normals[9] *= -1
        parts = np.r_[np.full(10, 3), 0]
        confidence = np.r_[np.ones(9), 0, 1]
        result, _ = continue_ear_skin(
            points,
            colors,
            confidence,
            parts,
            normals,
            vertices=vertices,
            faces=faces,
            binding=binding,
            regions={'-1': {'coreVertices': [0, 1, 2, 3]}},
        )
        np.testing.assert_allclose(result[9], skin)
        np.testing.assert_array_equal(result[:9], colors[:9])
        np.testing.assert_array_equal(result[10], colors[10])

    def fixture(self, count):
        vertices = np.zeros((153, 3))
        vertices[:3] = [[-0.02, -0.03, -0.05], [0.02, -0.02, -0.05], [0, -0.01, -0.05]]
        faces = np.array([[0, 1, 2]])
        weights = np.tile([0.3, 0.3, 0.4], (count, 1))
        weights[:3] = np.eye(3)
        points = weights @ vertices[:3]
        binding = {
            'triangles': faces,
            'triangleIds': np.zeros(count, dtype=int),
            'weights': weights,
        }
        return dict(
            points=points,
            face=vertices,
            faces=faces,
            binding=binding,
            parts=np.zeros(count, dtype=int),
            scalp=np.zeros(count),
        )

    def test_constant_field_stays_constant_through_confidence_ramp(self):
        q = np.r_[np.full(3, 0.3), np.linspace(0, 0.12, 25)]
        fixture = self.fixture(len(q))
        base = np.tile([0.43, 0.327, 0.281], (len(q), 1))
        reference = np.array([0.725, 0.502, 0.416])
        w = smooth((0.12 - q) / 0.12)
        current = base * (1 - w[:, None]) + reference * w[:, None]
        old, _, _ = continue_lower_skin(color=current, confidence=q, **fixture)
        self.assertGreater(np.max(old[:, 0] - base[:, 0]), 0.07)
        result, _, audit = continue_lower_skin(
            color=current, confidence=q, pre_completion_color=base, **fixture
        )
        np.testing.assert_allclose(result, base, atol=1e-14)
        np.testing.assert_array_equal(result[:3], current[:3])
        self.assertEqual(audit['unresolvedTargetTexels'], 0)

    def test_fractional_ownership_preserves_fallback_and_protected_texels(self):
        fixture = self.fixture(8)
        # Coordinate ownership is 0.5 for these samples (z fade midpoint).
        fixture['points'][3:6, 2] = -0.01
        fixture['scalp'][6] = 1  # Entirely outside skin ownership.
        preserve = np.zeros(8, bool)
        preserve[7] = True
        confidence = np.array([0.3, 0.3, 0.3, 0, 0.06, 0.12, 0, 0])
        base = np.tile([0.3, 0.2, 0.1], (8, 1))
        current = np.tile([0.8, 0.7, 0.6], (8, 1))
        harmonic = np.tile([0.4, 0.35, 0.25], (153, 1))
        resolved = np.zeros(153, bool)
        resolved[:3] = True
        with patch(
            'scripts.surface_completion.complete_surface_colors',
            return_value=(harmonic, resolved, {}),
        ):
            result, _, _ = continue_lower_skin(
                color=current,
                confidence=confidence,
                pre_completion_color=base,
                preserve=preserve,
                **fixture,
            )
        w = smooth((0.12 - confidence[3:6]) / 0.12)[:, None]
        expected = current[3:6] * 0.5 + (base[3:6] * (1 - w) + harmonic[:3] * w) * 0.5
        np.testing.assert_allclose(result[3:6], expected, atol=1e-15)
        np.testing.assert_array_equal(result[6:], current[6:])

    def test_unresolved_component_retains_general_fallback_exactly(self):
        fixture = self.fixture(6)
        # Separate triangle has no photographed anchors and must remain as-is.
        fixture['face'][3:6] = fixture['face'][:3] + [0.1, 0, 0]
        fixture['faces'] = np.array([[0, 1, 2], [3, 4, 5]])
        fixture['binding']['triangles'] = fixture['faces']
        fixture['binding']['triangleIds'][3:] = 1
        fixture['points'][3:] += [0.1, 0, 0]
        q = np.array([0.3, 0.3, 0.3, 0, 0.06, 0.12])
        current = np.tile([0.8, 0.7, 0.6], (6, 1))
        base = np.tile([0.3, 0.2, 0.1], (6, 1))
        result, _, audit = continue_lower_skin(
            color=current, confidence=q, pre_completion_color=base, **fixture
        )
        np.testing.assert_array_equal(result[3:], current[3:])
        self.assertEqual(audit['unresolvedTargetTexels'], 3)


if __name__ == '__main__':
    unittest.main()
