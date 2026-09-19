"""Regression tests for crown pinching and protected captured geometry."""

import unittest, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.head_material import (
    missing_head_material,
    head_scalp_region,
    photographed_scalp_region,
)
from scripts.astra_head_completion import apply_shape_prior


class CompletionTests(unittest.TestCase):
    def scalp_fixture(self, count):
        p = np.zeros((470, 3))
        p[10, 1], p[152, 1] = 0.1, -0.1
        return p, np.tile([-0.11, 0.03, -0.1], (count, 1))

    def test_observed_temple_skin_does_not_become_fallback_hair(self):
        p, x = self.scalp_fixture(3)
        support = np.array([0.701, 0.73, 0.8]) ** 3
        self.assertTrue((head_scalp_region(x, p, {}) == 1).all())
        # The first view's RGB-selection score is below the old .08 cutoff,
        # despite a clearly visible skin observation. Hair stays hair too.
        self.assertLess(0.701**8, 0.08)
        result = photographed_scalp_region(
            x, p, {}, support * [0, 0.5, 1], support, support
        )
        np.testing.assert_allclose(result, [0, 0.5, 1])

    def test_missing_or_many_grazing_views_retain_prior(self):
        p, x = self.scalp_fixture(4)
        x[3, 1] = -0.1  # Missing lower-neck evidence must stay skin.
        result = photographed_scalp_region(
            x, p, {}, [0, 0, 0, 0], [0, 10, 0.14, 0], [0, 0.07, 0.14, 0]
        )
        np.testing.assert_array_equal(result, head_scalp_region(x, p, {}))

    def test_semantic_support_fades_without_a_material_switch(self):
        p, x = self.scalp_fixture(1001)
        support = np.linspace(0, 0.5, len(x))
        scalp = photographed_scalp_region(x, p, {}, support * 0, support, support)
        self.assertEqual(scalp[0], 1)
        self.assertEqual(scalp[-1], 0)
        self.assertTrue((np.diff(scalp) <= 1e-12).all())
        self.assertLess(np.max(np.abs(np.diff(scalp))), 0.01)

    def test_absent_hair_is_not_created_by_semantic_noise(self):
        p, x = self.scalp_fixture(1)
        scalp = photographed_scalp_region(
            x, p, {'hair': {'present': False}}, [1], [1], [1]
        )
        np.testing.assert_array_equal(scalp, [0])

    def test_crown_does_not_have_an_angular_texture_pole(self):
        p = np.zeros((470, 3))
        p[10, 1] = 0.1
        p[152, 1] = -0.1
        p[469] = [0.1, 0.2, -0.2]
        angle = np.linspace(0, 2 * np.pi, 100)
        x = np.c_[
            np.cos(angle) * 1e-7, np.full(100, 0.19), -0.06 + np.sin(angle) * 1e-7
        ]
        n = np.tile([0, 1, 0], (100, 1))
        # Rear reference has high spatial variation so angular wrapping would
        # visibly fail this continuity bound around the former pole.
        yy, xx = np.mgrid[:200, :200]
        rear = np.stack(
            [
                0.1 + 0.1 * np.sin(xx / 7),
                0.1 + 0.08 * np.cos(yy / 9),
                np.full_like(xx, 0.1, dtype=float),
            ],
            2,
        )
        rgb, scalp = missing_head_material(
            x, n, p, {'hair': {'present': True}}, np.array([0.6, 0.4, 0.3]), rear
        )
        self.assertTrue(np.isfinite(rgb).all())
        self.assertLess(np.ptp(rgb, axis=0).max(), 1e-4)
        self.assertTrue((scalp == 1).all())

    def test_ai_shape_cannot_move_measured_face_or_neck_cut(self):
        p = np.zeros((480, 3))
        p[:, 2] = -0.2
        p[:, 1] = 0.07
        p[10, 1] = 0.1
        p[152, 1] = -0.1
        p[479, 1] = -0.124
        spec = {
            'model': 'fixture',
            'head': {
                'posteriorDepthScale': 1.1,
                'posteriorWidthScale': 1.06,
                'crownLiftMm': 6,
                'occiputLiftMm': 5,
            },
        }
        q, info = apply_shape_prior(
            p, np.array([[468, 469, 470], [471, 472, 473]]), 1, spec, {}
        )
        np.testing.assert_array_equal(q[:471], p[:471])
        np.testing.assert_array_equal(q[479], p[479])
        self.assertGreater(info['maxDisplacementMm'], 0)
        q, info = apply_shape_prior(
            p, np.array([[468, 469, 470]]), 1, spec, {'completeOrbit': True}
        )
        np.testing.assert_array_equal(q, p)
        self.assertFalse(info['applied'])


if __name__ == '__main__':
    unittest.main()
