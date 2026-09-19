"""Camera preference cannot manufacture evidence at an oblique jaw."""
import unittest
import numpy as np
from scripts.view_blending import frontal_preference


class ViewBlendingTests(unittest.TestCase):
    def test_clear_profile_outweighs_grazing_front_at_central_jaw(self):
        physical = np.array([.503, .953]) ** 8
        preferences = np.array([frontal_preference(1., .503, True),
                                frontal_preference(1., .503, False)])
        weights = physical * preferences
        self.assertGreater(weights[1] / weights.sum(), .99)
        colors = np.array([[.626, .523, .461], [.430, .327, .281]])
        composite = np.sum(colors * weights[:, None], axis=0) / weights.sum()
        self.assertLess(np.linalg.norm(composite - colors[1]), .003)
        self.assertLess(physical[0], .01)
        self.assertGreater(physical[1], .6)

    def test_frontal_features_keep_a_coherent_front_exposure(self):
        front = .95**8 * frontal_preference(1., .95, True)
        side = .8**8 * frontal_preference(1., .95, False)
        self.assertGreater(front / (front + side), .999)
        self.assertEqual(frontal_preference(0., .95, True), 1)
        self.assertEqual(frontal_preference(0., .95, False), 1)

    def test_protected_facial_features_keep_the_registered_exposure_on_curved_sides(self):
        self.assertEqual(frontal_preference(1., .5, True, release_region=0.), 31.)
        self.assertAlmostEqual(frontal_preference(1., .5, False, release_region=0.), .03)
        self.assertEqual(frontal_preference(1., .5, True, release_region=1.), 1.)

    def test_preference_fades_continuously_and_stays_bounded(self):
        facing = np.linspace(-1, 1, 10001)
        front = frontal_preference(np.ones_like(facing), facing, True)
        side = frontal_preference(np.ones_like(facing), facing, False)
        self.assertTrue(np.all(np.diff(front) >= 0))
        self.assertTrue(np.all(np.diff(side) <= 0))
        self.assertGreaterEqual(front.min(), 1)
        self.assertLessEqual(front.max(), 31)
        self.assertGreaterEqual(side.min(), .03 - 1e-12)
        self.assertLessEqual(side.max(), 1)
        self.assertLess(np.abs(np.diff(front)).max(), .04)
