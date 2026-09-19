"""Short hair fallback respects source regions and preserves observed features."""

import unittest
import numpy as np
from scripts.rear_hair import short_hair_patch
from scripts.head_material import missing_head_material


class RearHairTests(unittest.TestCase):
    def test_short_patch_stays_inside_hair_and_excludes_ear_hole(self):
        rgb = np.full((160, 120, 3), 0.2)
        rgb[:65] = [0.6, 0.1, 0.1]  # Distinct long crown.
        mask = np.zeros((160, 120), bool)
        mask[10:150, 10:110] = True
        mask[80:130, 20:50] = False  # Ear/skin occlusion.
        rgb[~mask] = 1
        result = short_hair_patch(rgb, mask)
        self.assertIsNotNone(result)
        sample, (x0, y0, x1, y1) = result
        self.assertTrue(mask[y0:y1, x0:x1].all())
        np.testing.assert_allclose(sample, 0.2)
        self.assertIsNone(short_hair_patch(rgb, np.zeros_like(mask)))

    def test_short_hair_replaces_only_lower_fallback_and_preserves_crown_skin(self):
        p = np.zeros((468, 3))
        p[10, 1], p[152, 1] = 0.1, -0.1
        p[0] = [0.1, 0.18, -0.2]
        x = np.array([[-0.1, 0.025, -0.18], [-0.02, 0.18, -0.16], [-0.1, -0.06, -0.18]])
        normals = np.tile([-1, 0, 0], (3, 1))
        spec = {'hair': {'lengthMm': 90, 'sideLengthMm': 15}}
        skin = np.array([0.55, 0.35, 0.25])
        long = np.full((20, 20, 3), 0.04)
        short = np.full((20, 20, 3), 0.18)
        args = (x, normals, p, spec, skin)
        old, _ = missing_head_material(
            *args, swatch=long, scalp_override=np.array([1, 1, 0])
        )
        new, _ = missing_head_material(
            *args,
            swatch=long,
            scalp_override=np.array([1, 1, 0]),
            short_swatches={'rear': short},
        )
        np.testing.assert_allclose(new[0], 0.18)
        np.testing.assert_array_equal(new[1:], old[1:])
        np.testing.assert_array_equal(new[2], skin)


if __name__ == '__main__':
    unittest.main()
