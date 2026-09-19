import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from reconstruction import decode
import json


class ReconstructionTests(unittest.TestCase):
    def test_actual_gaussian_fields(self):
        data = Path('public/reference/face-surface-fixture.ply').read_bytes()
        xyz, color, normals, thinness = decode(data)
        self.assertGreater(len(xyz), 50000)
        self.assertLess(thinness, 0.1)
        self.assertTrue(np.isfinite(xyz).all())
        self.assertTrue(np.all((color >= 0) & (color <= 1)))
        self.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5))

    def test_saved_reconstruction_is_real_triangle_surface(self):
        mesh = json.loads(Path('public/reference/face-poisson.json').read_text())
        p = np.array(mesh['positions']).reshape(-1, 3)
        f = np.array(mesh['indices']).reshape(-1, 3)
        self.assertGreater(len(f), 10000)
        self.assertTrue(np.isfinite(p).all())
        self.assertGreaterEqual(f.min(), 0)
        self.assertLess(f.max(), len(p))
        area = (
            np.linalg.norm(
                np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]), axis=1
            )
            / 2
        )
        self.assertGreater(float(np.quantile(area, 0.01)), 1e-12)
        self.assertEqual(
            mesh['stats']['sourceSplats'],
            len(
                decode(Path('public/reference/face-surface-fixture.ply').read_bytes())[
                    0
                ]
            ),
        )

    def test_bad_splat_is_rejected(self):
        with self.assertRaises(ValueError):
            decode(b'not a splat', 'splat')

    def test_translucent_surface_samples_can_be_inspected_without_changing_source(self):
        # An overlapping translucent layer may be visually opaque even though
        # individual samples fall below the generic converter's opacity cutoff.
        dtype = np.dtype(
            [
                ('xyz', '<f4', 3),
                ('scale', '<f4', 3),
                ('rgba', 'u1', 4),
                ('rotation', 'u1', 4),
            ]
        )
        points = np.zeros(100, dtype=dtype)
        points['xyz'][:, 0] = np.linspace(-0.1, 0.1, 100)
        points['scale'] = 0.003
        points['rgba'] = [150, 110, 90, 10]
        points['rotation'] = [255, 128, 128, 128]
        source = points.tobytes()
        with self.assertRaisesRegex(ValueError, 'Too few valid Gaussian samples'):
            decode(source, 'splat')
        self.assertEqual(len(decode(source, 'splat', min_opacity=0.01)[0]), 100)
        self.assertEqual(source, points.tobytes())
        for invalid in [-0.01, 1, float('nan')]:
            with self.assertRaises(ValueError):
                decode(source, 'splat', min_opacity=invalid)


if __name__ == '__main__':
    unittest.main()
