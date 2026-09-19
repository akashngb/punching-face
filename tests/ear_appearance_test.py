"""A source ear can occlude the head, but cannot invalidate a clear view."""

import unittest
import numpy as np

from scripts.ear_appearance import hidden_scalp_completion


class EarAppearanceTests(unittest.TestCase):
    def test_any_number_of_masked_views_cannot_overwrite_clear_skin(self):
        hidden = np.array([2.201157, 20.0, 2000.0])
        result = hidden_scalp_completion(
            hidden,
            np.full(3, 0.833717),
            hidden,
            np.ones(3),
            np.zeros(3),
            np.full(3, 0.601146),
        )
        np.testing.assert_array_equal(result, 0)

    def test_inadequate_grazing_photo_fades_without_promoting_masked_evidence(self):
        support = np.linspace(0, 0.12, 101)
        result = hidden_scalp_completion(
            np.ones(101),
            np.zeros(101),
            np.ones(101),
            np.ones(101),
            np.zeros(101),
            support,
        )
        self.assertEqual(result[0], 1)
        self.assertEqual(result[-1], 0)
        self.assertTrue((np.diff(result) <= 0).all())
        self.assertLess(np.max(np.abs(np.diff(result))), 0.02)

    def test_visible_oblique_hair_support_also_protects_its_photograph(self):
        from scripts.hair_appearance import hair_photo_support

        facing = np.array([0.5])
        quality = facing**8
        support = hair_photo_support(quality, facing, np.ones(1))
        self.assertLess(quality[0], 0.01)
        result = hidden_scalp_completion(
            np.ones(1), quality, np.ones(1), np.ones(1), np.zeros(1), support
        )
        np.testing.assert_array_equal(result, 0)

    def test_no_ear_mask_or_nonhair_prior_has_no_hair_override(self):
        result = hidden_scalp_completion(
            np.array([0, 1, 1, 1]),
            np.zeros(4),
            np.ones(4),
            np.array([1, 0, 1, 1]),
            np.array([0, 0, 3, 4]),
            np.zeros(4),
        )
        np.testing.assert_array_equal(result, 0)

    def test_missing_observation_retains_existing_completion_strength(self):
        hidden = np.array([0.0, 0.5, 1.0])
        result = hidden_scalp_completion(
            hidden, 1 - hidden, hidden, np.full(3, 0.8), np.zeros(3), np.zeros(3)
        )
        np.testing.assert_allclose(result, [0, 0.4, 0.8])


if __name__ == '__main__':
    unittest.main()
