import unittest, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.head_accessories import clean_view, paths_pixels, fit_temple
from types import SimpleNamespace
from scripts.hair_groom import build_hair_groom


class HairEyewearTests(unittest.TestCase):
    def spec(self, present=True):
        polygon = [
            [0.2, 0.3],
            [0.4, 0.3],
            [0.45, 0.35],
            [0.45, 0.5],
            [0.4, 0.55],
            [0.2, 0.55],
            [0.15, 0.5],
            [0.15, 0.35],
        ]
        return {
            'glasses': {'present': present, 'confidence': 0.2},
            'crops': {'front': [0, 0, 100, 100]},
            'views': [
                {
                    'filename': 'front',
                    'imageLeftLens': polygon,
                    'hairRegions': [[[0.1, 0.1], [0.8, 0.1], [0.8, 0.2]]],
                    'eyewearRegions': [],
                }
            ],
        }

    def test_frame_cleanup_preserves_eye_and_skin_detail_inside_clear_lenses(self):
        px = np.full((100, 100, 4), 255, np.uint8)
        px[:, :, :3] = [170, 120, 90]
        px[35:48, 23:38, :3] = [35, 25, 20]
        cleaned, mask, audit = clean_view(
            px, 'front', self.spec(), {}, return_details=True
        )
        self.assertEqual(mask[40, 30], 0)
        np.testing.assert_array_equal(cleaned[40, 30], px[40, 30])
        self.assertGreater(audit['excludedPixels'], 50)
        self.assertTrue(audit['preservedEyePhotographs'])
        self.assertEqual(
            set(paths_pixels(self.spec()['views'][0], [0, 0, 100, 100])),
            {'imageLeftLens'},
        )

    def test_rear_views_and_no_glasses_are_preserved(self):
        px = np.full((100, 100, 4), 255, np.uint8)
        np.testing.assert_array_equal(clean_view(px, 'rear', self.spec(), {}), px)
        np.testing.assert_array_equal(clean_view(px, 'front', self.spec(False), {}), px)

    def test_opaque_rim_cleanup_cannot_erase_landmark_brow_region(self):
        from scripts.photo_detail import BROWS, brow_mask

        px = np.full((100, 100, 4), 255, np.uint8)
        px[:, :, :3] = [175, 120, 85]
        landmarks = [{'x': 0.5, 'y': 0.5} for _ in range(468)]
        for group in BROWS:
            for j, i in enumerate(group):
                landmarks[i] = {'x': 0.2 + 0.025 * (j % 5), 'y': 0.28 + 0.04 * (j // 5)}
        frame = {'landmarks': landmarks}
        protected = brow_mask(px.shape, frame) > 0
        px[protected, :3] = [35, 25, 15]
        clean, mask, _ = clean_view(
            px,
            'front',
            self.spec(),
            frame,
            return_details=True,
            projected_mask=np.full((100, 100), 255, np.uint8),
        )
        np.testing.assert_array_equal(clean[protected], px[protected])
        self.assertFalse(mask[protected].any())

    def test_bald_and_zero_density_never_generate_a_hair_layer(self):
        p = np.zeros((468, 3))
        f = np.array([[0, 1, 2]])
        self.assertIsNone(build_hair_groom(p, f, {'hair': {'present': False}}))
        self.assertIsNone(
            build_hair_groom(p, f, {'hair': {'present': True, 'type': 'bald'}})
        )
        self.assertIsNone(
            build_hair_groom(p, f, {'hair': {'present': True, 'density': 0}})
        )

    def test_profile_contour_controls_temple_and_cannot_double_back_at_hinge(self):
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        camera = SimpleNamespace(
            cam_from_img=lambda xy: np.c_[-xy[:, 0], 0.5 - xy[:, 1]]
        )
        im = SimpleNamespace(
            name='side',
            camera_id=1,
            projection_center=lambda: np.array([0.5, 0, 0]),
            cam_from_world=lambda: SimpleNamespace(
                rotation=SimpleNamespace(matrix=lambda: rotation)
            ),
        )
        rec = SimpleNamespace(images={1: im}, cameras={1: camera})
        spec = {
            'views': [
                {
                    'filename': 'side',
                    'imageRightTemple': [
                        [0, 0.38],
                        [0.1, 0.38],
                        [0.2, 0.38],
                        [0.3, 0.4],
                    ],
                }
            ],
            'crops': {'side': [0, 0, 1, 1]},
        }
        hinge = np.array([0.08, 0.05, 0])
        p = np.array([[0.08, 0.05, -0.04]])
        q, audit = fit_temple(hinge, 1, spec, rec, np.zeros(3), np.eye(3), 1, p)
        self.assertEqual(audit['method'], 'profile-contour')
        np.testing.assert_array_equal(q[0], hinge)
        self.assertTrue((np.diff(q[:, 2]) < 0).all())
        self.assertLess(q[-1, 1], q[0, 1])
        self.assertGreater(q[0, 2] - q[-1, 2], 0.1)
        q, audit = fit_temple(
            hinge, 1, {'views': []}, rec, np.zeros(3), np.eye(3), 1, p
        )
        self.assertEqual(audit['method'], 'estimated-ear-hook')
        self.assertTrue(np.isfinite(q).all())


if __name__ == '__main__':
    unittest.main()
