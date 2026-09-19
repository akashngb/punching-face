"""Physical visibility and mesh-connected registration regressions."""

import unittest
from types import SimpleNamespace
import numpy as np
from scripts.texture_registration import plane_hit_visibility, ear_registration_weights
from scripts.texture_registration import posterior_view_weight


class VisibilityTests(unittest.TestCase):
    def test_second_surface_two_mm_behind_ear_is_not_visible(self):
        cp = np.array([[0, 0, 1], [0, 0, 1.002]])
        visible = plane_hit_visibility(
            cp, np.tile([0, 0, 1], (2, 1)), np.tile([0, 0, 1], (2, 1)), np.ones(2), 1
        )
        np.testing.assert_array_equal(visible, [True, False])

    def test_sloped_visible_facet_compares_at_same_pixel_center(self):
        # z=1+x: the texel and raster pixel differ by 10mm in raw depth.
        visible = plane_hit_visibility(
            np.array([[0.01, 0, 1.01]]),
            np.array([[-1, 0, 1]]),
            np.array([[0, 0, 1]]),
            np.array([1.0]),
            1,
        )
        self.assertTrue(visible[0])

    def test_no_depth_or_parallel_ray_has_no_photo_evidence(self):
        cp = np.tile([0, 0, 1], (3, 1))
        normals = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 1]])
        self.assertFalse(
            plane_hit_visibility(
                cp, normals, cp, np.array([np.inf, 1, np.nan]), 1
            ).any()
        )


class RegistrationTests(unittest.TestCase):
    def test_photo_warp_is_continuous_into_scalp_and_pins_measured_face(self):
        from scripts.ear_fit import ear_sample_coordinates

        p = np.array([[0.0, 0, 1], [0, 10, 1], [10, 5, 1]])
        pose = SimpleNamespace(
            rotation=SimpleNamespace(matrix=lambda: np.eye(3)), translation=np.zeros(3)
        )
        im = SimpleNamespace(
            name='profile.png',
            projection_center=lambda: np.array([-1.0, 0, 0]),
            cam_from_world=lambda: pose,
        )
        cam = SimpleNamespace(img_from_cam=lambda x: x[:, :2])
        ear = dict(
            visible=True,
            confidence=1,
            top=[0.1, 0],
            bottom=[0.1, 0.1],
            tragus=[0.2, 0.05],
        )
        semantics = dict(
            crops={'profile.png': [0, 0, 100, 100]},
            views=[
                dict(
                    filename='profile.png',
                    imageLeftEar=ear,
                    imageRightEar=dict(visible=False),
                )
            ],
        )
        xy = np.tile([5.0, 5.0], (4, 1))
        weights = {'-1': np.array([1.0, 0.99, 0.5, 0])}
        result = ear_sample_coordinates(
            xy,
            np.array([3, 0, 0, 0]),
            p,
            {'-1': {'anchors': dict(top=0, bottom=1, tragus=2)}},
            semantics,
            im,
            cam,
            np.zeros(3),
            np.eye(3),
            dict(scale=1),
            1,
            weights,
        )
        np.testing.assert_allclose(result[:, 0], [15, 14.9, 10, 5], atol=1e-6)
        np.testing.assert_array_equal(result[-1], xy[-1])
        np.testing.assert_array_equal(xy, np.tile([5.0, 5.0], (4, 1)))

    def test_coherent_posterior_preference_does_not_revive_occluded_samples(self):
        points = np.tile([-0.1, 0.03, -0.15], (4, 1))
        facing = np.array([0.65, 0.95, 0.9, 0.8])
        quality = facing**8
        quality[2] = 0
        weight = posterior_view_weight(
            points,
            facing,
            quality,
            np.array([-1, 0.03, -0.4]),
            np.array([True, True, True, False]),
        )
        self.assertLess(weight[1] / weight[0], 3)
        self.assertEqual(weight[2], 0)
        self.assertEqual(weight[3], quality[3])

    def test_falloff_stays_on_connected_surface_and_pins_face(self):
        p = np.zeros((478, 3))
        p[468:476] = [[i * 0.012, j * 0.006, 0] for i in range(4) for j in range(2)]
        p[476:] = [[0, 0, 0.001], [0.006, 0.006, 0.001]]
        f = np.array(
            [
                [474, 475, 473],
                [468, 469, 470],
                [469, 471, 470],
                [470, 471, 472],
                [471, 473, 472],
                [472, 473, 474],
                [0, 476, 477],
            ]
        )
        binding = {
            'triangles': f,
            'triangleIds': np.arange(len(f)),
            'weights': np.tile([1.0, 0, 0], (len(f), 1)),
        }
        binding['weights'][-1] = [0, 1, 0]
        saved = p.copy()
        result = ear_registration_weights(
            p, f, 1, {'-1': {'coreVertices': [468, 469]}}, binding
        )['-1']
        self.assertEqual(result[1], 1)
        self.assertGreater(result[3], result[5])
        self.assertGreater(result[5], 0)
        self.assertEqual(result[0], 0)  # Measured face.
        self.assertEqual(result[6], 0)  # Nearby disconnected sheet/cage.
        np.testing.assert_array_equal(p, saved)


if __name__ == '__main__':
    unittest.main()
