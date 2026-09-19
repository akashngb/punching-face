import unittest
from types import SimpleNamespace
import numpy as np
from scripts.hair_flow import (
    orientation_field,
    lift_direction,
    trace_locks,
    unit,
    bind_curve_points,
)
from scripts.hair_recognition import hair_completion


class HairFlowTests(unittest.TestCase):
    def test_ridge_orientation_follows_hair_not_cross_strand_gradient(self):
        y, x = np.mgrid[:160, :160]
        theta = np.deg2rad(32)
        transverse = -np.sin(theta) * x + np.cos(theta) * y
        gray = np.uint8(80 + 55 * np.sin(transverse * 0.65))
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        flow, confidence = orientation_field(rgb)
        target = np.array([np.cos(theta), np.sin(theta)])
        agreement = abs(flow[12:-12, 12:-12] @ target)
        self.assertGreater(np.median(agreement), 0.995)
        self.assertGreater(np.median(confidence[12:-12, 12:-12]), 0.8)
        _, flat = orientation_field(np.full_like(rgb, 50))
        self.assertEqual(float(flat.max()), 0)

    def test_image_flow_lift_preserves_projection_and_surface_tangency(self):
        cam = SimpleNamespace(cam_from_img=lambda xy: xy / 500)
        pose = SimpleNamespace(rotation=SimpleNamespace(matrix=lambda: np.eye(3)))
        xy = np.array([[25.0, 60.0], [-70.0, -20.0]])
        direction = unit(np.array([[1.0, 0.3], [0.2, 1.0]]))
        normal = unit(np.array([[0.2, 0.1, 1.0], [-0.3, 0.2, 1.0]]))
        tangent = lift_direction(cam, pose, xy, direction, normal, np.eye(3))
        np.testing.assert_allclose(np.sum(tangent * normal, axis=1), 0, atol=1e-8)
        points = np.c_[xy / 500, np.ones(2)]
        shifted = points + tangent * 0.0001
        projected = unit(shifted[:, :2] / shifted[:, 2:] * 500 - xy)
        np.testing.assert_allclose(projected, direction, atol=1e-8)

    def test_lock_tracing_follows_surface_and_stops_at_hair_boundary(self):
        x, z = np.meshgrid(np.linspace(-0.03, 0.03, 45), np.linspace(-0.02, 0.02, 35))
        roots = np.c_[x.ravel(), np.full(x.size, 0.14), z.ravel()]
        normals = np.tile([0.0, 1.0, 0.0], (len(roots), 1))
        flow = np.tile([1.0, 0.0, 0.0], (len(roots), 1))
        # Opposing signs encode the same line; the tracer must not collapse.
        flow[::2] *= -1
        curves, cn, lengths = trace_locks(
            roots,
            normals,
            flow,
            np.ones(len(roots)),
            {'lengthMm': 75, 'sideLengthMm': 14},
            0.1,
        )
        points = roots[:, None] + curves
        np.testing.assert_allclose(points[:, :, 1], 0.14, atol=1e-8)
        self.assertLess(abs(points[:, :, 0]).max(), 0.037)
        middle = np.argmin(np.linalg.norm(roots - [0, 0.14, 0], axis=1))
        self.assertGreater(np.ptp(points[middle, :, 0]), 0.035)
        self.assertTrue(np.isfinite(cn).all())
        self.assertTrue(np.isfinite(lengths).all())

    def test_hair_analysis_never_replaces_eyewear_or_mutates_completion(self):
        completion = {
            'hair': {'type': 'straight'},
            'glasses': {'present': True},
            'views': [{'filename': 'a', 'imageLeftLens': [[0.1, 0.2]]}],
        }
        result = hair_completion(
            completion,
            {
                'hair': {'type': 'wavy'},
                'model': 'vision',
                'version': 1,
                'framesSent': 1,
                'evidence': 'waves',
                'views': [
                    {
                        'filename': 'a',
                        'hairRegions': [],
                        'flowPaths': [],
                        'confidence': 0.9,
                    }
                ],
            },
        )
        self.assertEqual(completion['hair']['type'], 'straight')
        self.assertEqual(result['hair']['type'], 'wavy')
        self.assertEqual(result['glasses'], completion['glasses'])
        self.assertEqual(result['views'][0]['imageLeftLens'], [[0.1, 0.2]])

    def test_curve_stations_bind_to_local_surface_not_only_the_lock_center(self):
        p = np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])
        f = np.array([[0, 2, 1], [1, 2, 3]])
        points = np.array([[0.1, 0.001, 0.1], [0.9, 0.001, 0.9]])
        binding = bind_curve_points(p, f, points)
        triangles = np.array(binding['triangles']).reshape(-1, 3)
        weights = np.array(binding['weights']).reshape(-1, 3)
        before = np.einsum('ij,ijk->ik', weights, p[triangles])
        p[3, 1] += 0.01
        after = np.einsum('ij,ijk->ik', weights, p[triangles])
        self.assertAlmostEqual(after[0, 1], before[0, 1])
        self.assertGreater(after[1, 1] - before[1, 1], 0.007)


if __name__ == '__main__':
    unittest.main()
