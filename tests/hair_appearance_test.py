"""Visible rear photographs must not be replaced by frontal texture fallback."""

import unittest
import numpy as np
from scripts.hair_appearance import (
    hair_photo_support,
    blend_hair_completion,
    annotated_hair_support,
    semantic_angular_support,
    skin_semantic_support,
    semantic_view_support,
)


class HairAppearanceTests(unittest.TestCase):
    def test_missed_hair_boundary_is_unknown_not_observed_skin(self):
        from scripts.head_material import photographed_scalp_region

        hair = np.zeros((100, 100), bool)
        hair[:, :40] = True
        support = skin_semantic_support(hair, np.ones_like(hair), 6)
        self.assertEqual(support[50, 42], 0)  # Missed curl outside polygon.
        self.assertEqual(support[50, 60], 1)  # Clearly separated skin.
        self.assertEqual(support[2, 60], 0)  # Original source cutout fringe.
        self.assertTrue((np.diff(support[50, 39:60]) >= 0).all())
        face = np.zeros((468, 3))
        face[10, 1], face[152, 1] = 0.1, -0.1
        points = np.tile([-0.07, 0.0, -0.17], (2, 1))
        evidence = semantic_view_support(0.5, support[50, [42, 60]])
        scalp = photographed_scalp_region(
            points, face, {}, evidence * 0, evidence, evidence
        )
        np.testing.assert_array_equal(scalp, [1, 0])

    def test_boundary_guard_retains_established_skin_observations(self):
        from scripts.head_material import photographed_scalp_region

        face = np.zeros((468, 3))
        face[10, 1], face[152, 1] = 0.1, -0.1
        points = np.array([[-0.07, 0.0, -0.17]])
        # An already useful side view near a coarse boundary must not be
        # deleted: doing so would bring the false dark hair prior back.
        facing = np.array([0.7])
        evidence = semantic_view_support(facing, 0)
        np.testing.assert_array_equal(evidence, facing**3)
        scalp = photographed_scalp_region(
            points, face, {}, evidence * 0, evidence, evidence
        )
        np.testing.assert_array_equal(scalp, 0)
        np.testing.assert_array_equal(
            semantic_view_support(facing, 1), semantic_angular_support(facing)
        )

    def test_source_alpha_holes_cannot_supply_skin_negatives(self):
        hair = np.zeros((100, 100), bool)
        foreground = np.ones_like(hair)
        foreground[45:55, 45:55] = False
        support = skin_semantic_support(hair, foreground, 4)
        self.assertEqual(support[50, 50], 0)
        self.assertEqual(support[50, 57], 0)
        self.assertEqual(support[50, 70], 1)
        np.testing.assert_array_equal(
            skin_semantic_support(np.ones_like(hair), foreground, 4), 0
        )

    def test_semantic_boundary_scales_with_image_resolution(self):
        hair = np.zeros((100, 100), bool)
        hair[:, :40] = True
        fg = np.ones_like(hair)
        small = skin_semantic_support(hair, fg, 6)
        large = skin_semantic_support(
            np.repeat(np.repeat(hair, 2, 0), 2, 1), np.ones((200, 200), bool), 12
        )
        np.testing.assert_allclose(small[50, 40:60], large[101, 81:121:2])

    def test_oblique_classes_survive_tiny_rgb_preference(self):
        from scripts.head_material import photographed_scalp_region

        face = np.zeros((468, 3))
        face[10, 1], face[152, 1] = 0.1, -0.1
        points = np.tile([-0.07, 0.0, -0.17], (5, 1))
        facing = np.cos(np.radians([0, 30, 45, 60, 65]))
        evidence = semantic_angular_support(facing)
        # Same visible skin patch across camera angles, despite a hair prior.
        scalp = photographed_scalp_region(
            points, face, {}, evidence * 0, evidence, evidence
        )
        np.testing.assert_array_equal(scalp, 0)
        self.assertLess(facing[-1] ** 8, 0.002)
        # Actual hair observations remain hair at those same angles.
        scalp = photographed_scalp_region(
            points, face, {}, evidence, evidence, evidence
        )
        np.testing.assert_array_equal(scalp, 1)

    def test_grazing_or_occluded_observations_cannot_erase_the_prior(self):
        from scripts.head_material import photographed_scalp_region

        face = np.zeros((468, 3))
        face[10, 1], face[152, 1] = 0.1, -0.1
        points = np.tile([-0.07, 0.0, -0.17], (7, 1))
        facing = np.cos(np.radians([75, 80, 85, 90, 45, 45, 45]))
        # Last three samples: opaque frame, ear occlusion, missing source alpha.
        visibility = np.array([1, 1, 1, 1, 0, 0, 0])
        evidence = semantic_angular_support(facing) * visibility
        scalp = photographed_scalp_region(
            points, face, {}, evidence * 0, evidence * 100, evidence
        )
        np.testing.assert_array_equal(scalp, 1)

    def test_semantic_angle_score_is_bounded_continuous_and_monotonic(self):
        facing = np.linspace(-1, 1, 20001)
        support = semantic_angular_support(facing)
        self.assertTrue((support >= 0).all())
        self.assertTrue((support <= 1).all())
        self.assertTrue((np.diff(support) >= 0).all())
        self.assertLess(np.max(np.diff(support)), 0.001)
        np.testing.assert_array_equal(support[facing <= 0.25], 0)

    def test_coarse_polygon_does_not_promote_neck_skin_as_hair(self):
        rgb = np.full((80, 80, 3), 0.12)
        rgb[60:] = [0.7, 0.5, 0.4]
        mask = np.zeros((80, 80), bool)
        mask[5:70, 5:75] = True  # Includes ten erroneous skin rows.
        support = annotated_hair_support(rgb, mask)
        self.assertEqual(support[30, 30], 1)
        self.assertEqual(support[62, 30], 0)
        self.assertEqual(support[75, 30], 0)

    def test_light_hair_uses_its_own_color_distribution(self):
        rgb = np.full((80, 80, 3), [0.88, 0.83, 0.69])
        mask = np.zeros((80, 80), bool)
        mask[5:75, 5:75] = True
        self.assertEqual(annotated_hair_support(rgb, mask)[30, 30], 1)

    def test_visible_oblique_rear_photo_survives_completion_threshold(self):
        facing = np.array([1.0, 0.7, 0.5, 0.25])
        support = hair_photo_support(facing**8, facing, np.ones(4))
        np.testing.assert_allclose(support, facing**2)
        self.assertLess(0.5**8, 0.12)
        self.assertGreater(support[2], 0.12)
        self.assertLess(support[3], 0.12)

    def test_occlusion_masks_and_cutout_fade_remain_authoritative(self):
        facing = np.array([0.8, 0.6, 0.0, 0.5, 1e-4])
        coverage = np.array([0.0, 0.0, 1.0, 0.2, 1.0])
        quality = facing**8 * coverage
        support = hair_photo_support(quality, facing, np.ones(5))
        np.testing.assert_allclose(support, [0, 0, 0, 0.05, quality[-1]])
        self.assertTrue(np.isfinite(support).all())

    def test_nape_skin_and_unclassified_views_do_not_gain_support(self):
        facing = np.full(4, 0.5)
        quality = facing**8
        support = hair_photo_support(quality, facing, [0, 0.25, 0.5, 1])
        np.testing.assert_allclose(support, [quality[0], 0.0625, 0.125, 0.25])
        # Hair boundary fades monotonically without a hard semantic switch.
        self.assertTrue((np.diff(support) > 0).all())

    def test_recovered_hair_uses_its_own_photo_not_competing_neck_skin(self):
        photo = np.full((3, 3), 0.8)  # Competing camera sees pale neck skin.
        inferred = np.full((3, 3), 0.1)
        hair_total = np.array([0.004, 0.004, 0.0])
        hair_accum = np.full((3, 3), 0.2) * hair_total[:, None]
        old = np.array([1.0, 0.0, 1.0])
        new = np.zeros(3)
        result = blend_hair_completion(
            photo, inferred, hair_accum, hair_total, old, new
        )
        np.testing.assert_allclose(result[0], 0.2)  # Actual photographed hair.
        np.testing.assert_array_equal(result[1], photo[1])  # Trusted photo stays exact.
        np.testing.assert_array_equal(
            result[2], inferred[2]
        )  # No donor, keep estimate.


if __name__ == '__main__':
    unittest.main()
