"""Generated socket shading follows geometry without altering eye reflectance."""

import unittest
import numpy as np
import trimesh
from scripts.eye_shading import estimate_eye_shading, apply_estimated_eye_shading


class EyeShadingTests(unittest.TestCase):
    def fixture(self, lid=True, lid_distance=0.0):
        globe = trimesh.creation.icosphere(subdivisions=2, radius=0.014)
        points = np.asarray(globe.vertices)
        faces = np.asarray(globe.faces)
        labels = np.ones(len(points), np.uint8)
        if lid:
            patch = np.array(
                [
                    [-0.020, 0.008, 0.003],
                    [0.020, 0.008, 0.003],
                    [0.020, 0.008, 0.032],
                    [-0.020, 0.008, 0.032],
                ]
            )
            patch[:, 1] += lid_distance
            faces = np.vstack((faces, np.array([[0, 1, 2], [0, 2, 3]]) + len(points)))
            points = np.vstack((points, patch))
            labels = np.r_[labels, np.zeros(4, np.uint8)]
        y = np.array([0.006, -0.006, 0.003, -0.003])
        texels = np.column_stack((np.zeros(4), y, np.sqrt(0.014**2 - y**2)))
        # Identical spatial samples with another owner must remain unchanged.
        texels = np.vstack((texels, texels[0], [0, 0, -0.014]))
        return points, faces, labels, texels, np.array([1, 1, 1, 1, 0, 1])

    def test_upper_lid_darkens_upper_globe_and_removal_restores_baseline(self):
        shaded, audit = estimate_eye_shading(*self.fixture())
        baseline, _ = estimate_eye_shading(*self.fixture(lid=False))
        np.testing.assert_array_equal(baseline, np.ones(len(baseline)))
        self.assertLess(shaded[0], shaded[1])
        self.assertLess(shaded[0], 0.99)
        self.assertEqual(shaded[4], 1.0)  # Skin is never modified.
        self.assertEqual(shaded[5], 1.0)  # Hidden rear globe is not a front lid.
        self.assertTrue(audit['estimated'])
        self.assertLessEqual(audit['raysCast'], 96 * 24)

    def test_deterministic_finite_bounded_scalar_shading_preserves_chroma(self):
        first, audit = estimate_eye_shading(*self.fixture())
        second, second_audit = estimate_eye_shading(*self.fixture())
        np.testing.assert_array_equal(first, second)
        self.assertEqual(audit, second_audit)
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.all((first >= 0.55) & (first <= 1)))
        color = np.tile([0.88, 0.85, 0.80], (len(first), 1))
        result = color * first[:, None]
        np.testing.assert_allclose(
            result / result.sum(axis=1)[:, None], color / color.sum(axis=1)[:, None]
        )
        np.testing.assert_array_equal(result[4], color[4])

    def test_remote_surface_does_not_cast_local_socket_shadow(self):
        shading, _ = estimate_eye_shading(*self.fixture(lid_distance=0.1))
        np.testing.assert_array_equal(shading, np.ones(len(shading)))

    def test_material_shading_preserves_skin_and_recorded_eye_pixels_exactly(self):
        p, f, labels, texels, parts = self.fixture()
        parts[2] = 2
        colors = np.tile([0.88, 0.85, 0.8], (len(parts), 1))
        output, audit = apply_estimated_eye_shading(
            p, f, labels, texels, parts, colors, [1]
        )
        np.testing.assert_array_equal(output[parts != 1], colors[parts != 1])
        self.assertTrue(np.all(output[0] < colors[0]))
        self.assertTrue(audit['photographicIrisUnchanged'])
        unchanged, audit = apply_estimated_eye_shading(
            p, f, labels, texels, parts, colors, []
        )
        np.testing.assert_array_equal(unchanged, colors)
        self.assertFalse(audit['applied'])


if __name__ == '__main__':
    unittest.main()
