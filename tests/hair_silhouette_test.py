import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from PIL import Image

from scripts.hair_silhouette import refine_hair_contours
from scripts.pipeline_accel import rasterize_depth
from scripts.hair_flow import conform_curve_points


class HairSilhouetteTests(unittest.TestCase):
    def test_hair_only_refit_cannot_reset_the_accepted_nape(self):
        from scripts.rebuild_hair import refit_domain

        p = np.zeros((472, 3))
        p[10, 1], p[152, 1] = 0.1, -0.1
        p[468:] = [
            [0, 0.15, -0.09],
            [0, -0.025, -0.20],
            [0, -0.065, -0.21],
            [0.025, 0.13, 0.0],
        ]
        faces = np.array([[0, 1, 471], [468, 469, 470]])
        domain = refit_domain(p, faces, 1, {'rearHairlineFraction': 0.9})
        self.assertGreater(domain[468], 0)
        np.testing.assert_array_equal(domain[[10, 152, 469, 470, 471]], 0)

    def test_local_contours_restore_asymmetric_quiff_without_moving_skin(self):
        mesh = trimesh.creation.uv_sphere(count=[24, 40])
        p = mesh.vertices * [0.095, 0.13, 0.095]
        faces = mesh.faces
        target = p.copy()
        target[:, 1] += (
            0.016
            * np.exp(-(((p[:, 0] + 0.034) / 0.035) ** 2))
            * np.clip(p[:, 1] / 0.09, 0, 1) ** 3
        )
        weight = (p[:, 1] > 0.025).astype(float)
        cam = SimpleNamespace(img_from_cam=lambda cp: cp[:, :2] / cp[:, 2:] * 640 + 128)
        views = []
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'images').mkdir()
            for i, angle in enumerate([-0.65, 0, 0.65]):
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, 0, -s], [0, -1, 0], [-s, 0, -c]])
                t = np.array([0, 0, 0.85])
                pose = SimpleNamespace(
                    rotation=SimpleNamespace(matrix=lambda R=R: R), translation=t
                )
                view = SimpleNamespace(
                    name=f'{i}.png',
                    camera_id=1,
                    cam_from_world=lambda pose=pose: pose,
                    projection_center=lambda R=R, t=t: -t @ R,
                )
                cp = target @ R.T + t
                mask = np.isfinite(
                    rasterize_depth(cam.img_from_cam(cp), cp[:, 2], faces, 256, 256)
                )
                image = np.full((256, 256, 4), 128, np.uint8)
                image[:, :, 3] = mask * 255
                Image.fromarray(image).save(folder / 'images' / view.name)
                views.append(view)
            rec = SimpleNamespace(cameras={1: cam})
            result, audit = refine_hair_contours(
                folder,
                p,
                faces,
                weight,
                np.ones(len(p)),
                views,
                rec,
                np.zeros(3),
                np.eye(3),
                {'scale': 1},
            )
            np.testing.assert_array_equal(result[weight == 0], p[weight == 0])
            self.assertTrue(audit['available'])
            self.assertGreater(audit['directlyConstrainedVertices'], 20)
            self.assertLess(
                audit['finalContourResidualMedianPx'],
                audit['initialContourResidualMedianPx'],
            )

            def mismatch(vertices):
                value = 0
                for view in views:
                    pose = view.cam_from_world()
                    cp = vertices @ pose.rotation.matrix().T + pose.translation
                    mask = np.isfinite(
                        rasterize_depth(cam.img_from_cam(cp), cp[:, 2], faces, 256, 256)
                    )
                    reference = (
                        np.asarray(Image.open(folder / 'images' / view.name))[:, :, 3]
                        > 200
                    )
                    value += np.sum(mask != reference)
                return value

            self.assertLess(mismatch(result), mismatch(p) * 0.8)
            self.assertLess(np.linalg.norm(result - p, axis=1).max(), 0.0221)
            self.assertTrue(np.isfinite(result).all())

    def test_underobserved_surface_is_unchanged(self):
        p = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
        q, audit = refine_hair_contours(
            None,
            p,
            np.array([[0, 1, 2]]),
            np.ones(3),
            np.ones(3),
            [],
            None,
            None,
            None,
            None,
        )
        np.testing.assert_array_equal(q, p)
        self.assertFalse(audit['available'])

    def test_fibers_conform_to_real_triangles_before_binding(self):
        p = np.array([[0.0, 0, 0], [0.01, 0, 0], [0, 0, 0.01], [0.01, 0.006, 0.01]])
        f = np.array([[0, 2, 1], [1, 2, 3]])
        traced = np.array([[0.002, -0.001, 0.002], [0.008, 0.001, 0.008]])
        normal = np.asarray(trimesh.Trimesh(p, f, process=False).vertex_normals)
        surface, normals, binding = conform_curve_points(p, f, traced, normal)
        ids = np.array(binding['triangles']).reshape(-1, 3)
        weights = np.array(binding['weights']).reshape(-1, 3)
        np.testing.assert_allclose(
            surface, np.einsum('ij,ijk->ik', weights, p[ids]), atol=1e-12
        )
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
        self.assertGreater(np.linalg.norm(surface - traced, axis=1).min(), 0.0001)


if __name__ == '__main__':
    unittest.main()
