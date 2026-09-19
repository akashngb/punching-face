"""Regressions for semantic surface ownership and bounded ear corrections."""

import json, hashlib, math, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pycolmap
from PIL import Image
from scripts.ear_fit import (
    fit_ears,
    ear_vertex_labels,
    source_ear_mask,
    triangulate_ears,
    refine_ear_ownership,
    KEYS,
)
from scripts.head_accessories import semantic_mask
from scripts.photo_cameras import matches_supplied_fov, recover
from scripts.frame_evidence import choose_views
from scripts.glasses_reference import (
    composite_reference,
    generate_reference,
    surface_coverage,
    current_reference_region,
    load_references,
)
from scripts.ear_deformation import surface_quality, bounded_surface_step
from scripts.skin_continuation import continue_lower_skin, continue_ear_skin
from scripts.surface_intersections import new_crossings
from scripts.photo_geometry import (
    measured_face_footprint,
    OVAL,
    raster_atlas,
    prepare_texture_view,
    blend_estimated_colour,
)
from scripts.photo_hair import fit_template_hair
from scripts.cleanup_lighting import match_cleanup_lighting


class EarTests(unittest.TestCase):
    def test_cleanup_matches_boundary_illumination_without_copying_the_hidden_rim(self):
        points = np.array(
            [
                [x, y, 0.0]
                for x in np.linspace(-0.018, 0.018, 25)
                for y in np.linspace(-0.018, 0.018, 25)
            ]
        )
        radius = np.linalg.norm(points, axis=1)
        n = len(points)
        normal = np.tile([0.0, 0.0, 1.0], (n, 1))
        parts = np.zeros(n, int)
        coverage = np.where(radius < 0.011, 1.0, np.where(radius < 0.017, 0.2, 0.0))
        clean = np.tile([0.45, 0.30, 0.25], (n, 1))
        photo = clean + [0.18, 0.12, 0.08]
        # The old photograph is dark under the opaque rim. Only outer
        # feather observations may determine the smooth lighting correction.
        photo[radius < 0.011] = 0.03
        center = int(np.argmin(radius))
        clean[center] += [0.025, 0.015, 0.01]
        protected = [center - 1, center + 1]
        parts[protected[0]] = 3
        side = np.ones(n)
        side[protected[1]] = 0
        result, count = match_cleanup_lighting(
            points,
            normal,
            parts,
            photo,
            clean,
            coverage,
            np.ones(n),
            coverage > 0,
            side,
        )
        np.testing.assert_allclose(
            result[center], clean[center] + [0.18, 0.12, 0.08], atol=1e-8
        )
        np.testing.assert_array_equal(result[protected], clean[protected])
        np.testing.assert_array_equal(result[coverage == 0], clean[coverage == 0])
        self.assertGreater(count, 0)
        empty, count = match_cleanup_lighting(
            points,
            normal,
            parts,
            photo,
            clean,
            coverage,
            np.zeros(n),
            coverage > 0,
            side,
        )
        np.testing.assert_array_equal(empty, clean)
        self.assertEqual(count, 0)

    def test_near_limit_deformation_keeps_accuracy_without_relaxing_quality(self):
        rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        desired = rest.copy()
        angle = np.deg2rad(85)
        desired[2] = [0, np.cos(angle), np.sin(angle)]
        faces = np.array([[0, 1, 2]])
        self.assertLess(
            surface_quality(rest, desired, faces)['minimumNormalAgreement'], 0.1
        )
        candidate, quality = bounded_surface_step(rest, desired, faces)
        self.assertGreater(quality['appliedFraction'], 0.98)
        self.assertLess(quality['appliedFraction'], 1.0)
        self.assertGreater(quality['minimumNormalAgreement'], 0.1)
        self.assertEqual(quality['newCrossings'], 0)
        self.assertLess(np.linalg.norm(candidate - desired), 0.03)

    def test_hair_envelope_cannot_inflate_ears_even_above_its_hairline_threshold(self):
        p = np.zeros((476, 3))
        p[10, 1] = 0.1
        p[152, 1] = -0.1
        p[468:472] = [
            [0.07, 0.14, -0.04],
            [0.072, 0.14, -0.04],
            [0.072, 0.142, -0.04],
            [0.07, 0.142, -0.04],
        ]
        p[472:] = [
            [0, 0.175, -0.04],
            [0.01, 0.175, -0.04],
            [0.01, 0.18, -0.04],
            [0, 0.18, -0.04],
        ]
        faces = np.array(
            [[468, 469, 470], [468, 470, 471], [472, 473, 474], [472, 474, 475]]
        )
        regions = {'1': {'vertices': list(range(468, 472))}}
        pose = NS(
            rotation=NS(matrix=lambda: np.eye(3)), translation=np.array([0, 0, 1])
        )
        camera = NS(
            img_from_cam=lambda points: np.tile([100.0, 100.0], (len(points), 1))
        )
        im = NS(name='view.png', camera_id=1, cam_from_world=lambda: pose)
        landmarks = [{'x': 0.5, 'y': 0.5} for _ in range(468)]
        landmarks[10]['y'] = 0.25
        landmarks[152]['y'] = 0.75
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'images').mkdir()
            Image.fromarray(np.full((400, 200, 4), 255, np.uint8)).save(
                folder / 'images/view.png'
            )
            with (
                patch('scripts.frame_evidence.assess_frames', return_value={}),
                patch('scripts.frame_evidence.choose_views', return_value={'view.png'}),
            ):
                out, audit = fit_template_hair(
                    folder,
                    p,
                    faces,
                    0,
                    NS(cameras={1: camera}),
                    {'view.png': {'yaw': 0, 'landmarks': landmarks}},
                    [im],
                    np.zeros(3),
                    np.eye(3),
                    {'scale': 1},
                    regions,
                )
        np.testing.assert_array_equal(out[:472], p[:472])
        self.assertGreater(np.linalg.norm(out[472:] - p[472:]), 0.001)
        self.assertEqual(audit['earVerticesProtected'], 4)

    def test_unseen_ear_skin_uses_only_nearby_same_ear_facing_evidence(self):
        xy = np.array(
            [
                [x, y, 0]
                for x in (-0.006, -0.002, 0.002, 0.006)
                for y in (-0.004, 0, 0.004)
            ]
        )
        # Same spatial locations on the opposite fold, other ear and scalp
        # must not bleed into the missing visible helix patch.
        p = np.vstack([xy, xy, xy, xy, [[0, 0, 0], [0.05, 0, 0], [0, 0, 0]]])
        parts = np.r_[np.full(24, 3), np.full(12, 4), np.zeros(12), [3, 3, 0]]
        normal = np.tile([0.0, 0.0, 1.0], (len(p), 1))
        normal[12:24] *= -1
        confidence = np.ones(len(p))
        confidence[-3:] = 0
        color = np.tile([0.95, 0.8, 0.7], (len(p), 1))
        color[:12] = [0.36, 0.25, 0.21]
        out, count = continue_ear_skin(p, color, confidence, parts, normal)
        np.testing.assert_allclose(out[-3], [0.36, 0.25, 0.21])
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(out[:-3], color[:-3])
        np.testing.assert_array_equal(out[-2:], color[-2:])
        none, count = continue_ear_skin(p, color, np.zeros(len(p)), parts, normal)
        np.testing.assert_array_equal(none, color)
        self.assertEqual(count, 0)

    def test_changed_annotations_invalidate_mask_but_reuse_the_source_verified_local_edit(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'images').mkdir()
            cache = folder / 'glasses-reference/view'
            cache.mkdir(parents=True)
            raw = np.full((100, 100, 4), 180, np.uint8)
            raw[:, :, 3] = 255
            Image.fromarray(raw).save(folder / 'images/view.png')
            spec = {
                'glasses': {'present': True},
                'frontFilename': 'view.png',
                'views': [{'filename': 'view.png', 'bridge': [[0.4, 0.5], [0.6, 0.5]]}],
                'crops': {'view.png': [0, 0, 100, 100]},
            }
            path = folder / 'astra-head-completion.json'
            path.write_text(json.dumps(spec))
            box, mask = current_reference_region(folder, 'view.png', raw)
            meta = {
                'filename': 'view.png',
                'sourceHash': hashlib.sha256(raw.tobytes()).hexdigest(),
                'crop': box,
            }
            (cache / 'reference.json').write_text(json.dumps(meta))
            Image.fromarray(mask).save(cache / 'mask.png')
            Image.fromarray(raw[:, :, :3]).save(cache / 'registered.png')
            Image.fromarray(raw[:, :, :3]).save(cache / 'generated-original.png')
            self.assertIn('view.png', load_references(folder))
            spec['views'][0]['bridge'] = [[0.4, 0.3], [0.6, 0.3]]
            path.write_text(json.dumps(spec))
            self.assertEqual(load_references(folder), {})
            with (
                patch(
                    'scripts.glasses_reference.register',
                    return_value={'available': True},
                ) as local,
                patch('scripts.glasses_reference.urlopen') as remote,
            ):
                self.assertTrue(generate_reference(folder, spec)['available'])
                local.assert_called_once()
                remote.assert_not_called()

    def test_reference_feather_does_not_suddenly_switch_to_another_camera(self):
        raw = np.full((100, 100, 4), 204, np.uint8)
        raw[:, :, 3] = 255
        mask = np.zeros((100, 100), np.uint8)
        mask[20:80, 20:80] = 255
        reference = {
            'crop': [0, 0, 100, 100],
            'mask': mask,
            'pixels': np.full((100, 100, 3), 209, np.uint8),
            'medianRegistrationErrorPx': 0.5,
        }
        pixels, excluded, region, audit = prepare_texture_view(
            raw, 'view', None, {}, {}, reference
        )
        # Strong owning photo, weak differently lit second camera. The first
        # pixel inside the edit previously lost its entire owning photograph.
        own = 0.5 * (excluded[50, 19:22] == 0)
        other = 0.07
        color = (pixels[50, 19:22, :3] / 255 * own[:, None] + 0.2 * other) / (
            own + other
        )[:, None]
        self.assertLess(np.abs(np.diff(color, axis=0)).max(), 1 / 255)
        self.assertGreater(color.min(), 0.7)
        self.assertEqual(audit['excludedPixels'], 0)
        self.assertGreater(audit['estimatedReferencePixels'], 0)
        self.assertEqual(region[50, 20], 255)

    def test_estimated_colour_is_continuous_across_the_old_confidence_switch(self):
        ratio = np.linspace(0.06, 0.22, 161)
        observed = np.tile([0.2, 0.3, 0.4], (len(ratio), 1))
        estimate = np.tile([0.8, 0.7, 0.6], (len(ratio), 1))
        result, occluded = blend_estimated_colour(
            observed, ratio, estimate, np.ones_like(ratio)
        )
        self.assertLess(np.abs(np.diff(result, axis=0)).max(), 0.01)
        np.testing.assert_allclose(result[0], estimate[0])
        np.testing.assert_allclose(result[-1], observed[-1])
        self.assertTrue(occluded[0])
        self.assertFalse(occluded[-1])
        untouched, _ = blend_estimated_colour(
            observed, ratio, np.zeros_like(estimate), np.zeros_like(ratio)
        )
        np.testing.assert_array_equal(untouched, observed)

    def test_mixed_ear_triangle_ownership_does_not_depend_on_vertex_order(self):
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        n = np.tile([0.0, 0.0, 1.0], (3, 1))
        uv = p[:, :2]
        mapping = np.arange(3)
        for labels in (np.array([0, 3, 3]), np.array([0, 3, 4])):
            results = [
                raster_atlas(p, n, mapping, np.array([order]), uv, [True], 32, labels)
                for order in ([0, 1, 2], [1, 2, 0], [2, 0, 1])
            ]
            self.assertGreater(len(np.unique(results[0][-1])), 1)
            for result in results[1:]:
                np.testing.assert_array_equal(result[3], results[0][3])
                np.testing.assert_array_equal(result[-1], results[0][-1])

    def test_photographed_outline_removes_skull_vertices_from_template_ear_label(self):
        p = np.array(
            [
                [-0.25, -0.1, 0],
                [-0.1, -0.1, 0],
                [-0.175, 0.1, 0],
                [-0.7, 0.4, 0],
                [-0.55, 0.4, 0],
                [-0.625, 0.6, 0],
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        camera = pycolmap.Camera(
            model='SIMPLE_PINHOLE', width=100, height=100, params=[50, 50, 50]
        )
        pose = NS(
            rotation=NS(matrix=lambda: np.diag([1, -1, -1])),
            translation=np.array([0, 0, 1]),
        )
        im = NS(
            name='front',
            camera_id=1,
            projection_center=lambda: np.array([0, 0, 1]),
            cam_from_world=lambda: pose,
        )
        ear = {
            'visible': True,
            'confidence': 0.9,
            'outline': [[0.35, 0.35], [0.48, 0.35], [0.48, 0.65], [0.35, 0.65]],
        }
        semantics = {
            'crops': {'front': [0, 0, 100, 100]},
            'views': [
                {
                    'filename': 'front',
                    'imageLeftEar': ear,
                    'imageRightEar': {'visible': False, 'confidence': 0, 'outline': []},
                }
            ],
        }
        regions = {'-1': {'coreVertices': list(range(6))}}
        with (
            tempfile.TemporaryDirectory() as temp,
            patch('scripts.photo_geometry.zbuffer', return_value=np.ones((100, 100))),
        ):
            report = refine_ear_ownership(
                Path(temp),
                p,
                faces,
                regions,
                semantics,
                NS(images={1: im}, cameras={1: camera}),
                np.zeros(3),
                np.eye(3),
                {'scale': 1},
            )
        self.assertEqual(regions['-1']['coreVertices'], [0, 1, 2])
        self.assertEqual(report['-1']['reclassifiedAdjacentSurface'], 3)

    def test_face_coverage_cannot_count_unobserved_template_neck_as_missing_face(self):
        cage = np.zeros((468, 3))
        angle = np.linspace(0, 2 * np.pi, len(OVAL), endpoint=False)
        cage[OVAL, :2] = np.c_[0.08 * np.cos(angle), 0.10 * np.sin(angle)]
        samples = np.array(
            [
                [0, 0, 0],
                [0.04, 0.03, -0.01],
                [0, -0.104, 0],
                [0, 0.11, 0],
                [0.09, 0, 0],
                [0.07, 0.09, 0],
            ]
        )
        np.testing.assert_array_equal(
            measured_face_footprint(samples, cage),
            [True, True, False, False, False, False],
        )

    def test_new_surface_crossing_is_detected_even_when_topology_does_not_change(self):
        candidate = np.array(
            [
                [-1, -1, 0],
                [1, -1, 0],
                [0, 1, 0],
                [0, -0.5, -1],
                [0, -0.5, 1],
                [0, 0.5, 0.5],
            ],
            float,
        )
        rest = candidate.copy()
        rest[3:, 2] += 2
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        result = new_crossings(rest, candidate, faces)
        self.assertEqual(result['existingCrossings'], 0)
        self.assertEqual(result['newCrossings'], 1)

    def test_unknown_neck_color_continues_nearby_skin_without_overwriting_observations_or_ears(
        self,
    ):
        from scipy.spatial import Delaunay

        x, y = np.meshgrid(np.linspace(-0.07, 0.07, 20), np.linspace(-0.12, -0.085, 20))
        p = np.column_stack((x.ravel(), y.ravel(), np.full(400, -0.06)))
        face = np.vstack((np.zeros((468, 3)), p))
        face[152, 1] = -0.10
        triangles = Delaunay(p[:, :2]).simplices + 468
        # Each sample lies exactly on one original vertex of its own triangle.
        owner = []
        bary = []
        for vertex in range(468, 868):
            tri, corner = np.argwhere(triangles == vertex)[0]
            w = np.zeros(3)
            w[corner] = 1
            owner.append(tri)
            bary.append(w)
        binding = {
            'triangles': triangles,
            'triangleIds': np.array(owner),
            'weights': np.array(bary),
        }
        color = np.tile([0.62, 0.43, 0.33], (400, 1))
        color[:10] = [0.12, 0.12, 0.12]
        confidence = np.ones(400)
        confidence[:10] = 0
        parts = np.zeros(400, int)
        parts[0] = 3
        preserve = np.zeros(400, bool)
        preserve[1] = True
        out, count, audit = continue_lower_skin(
            p,
            color,
            confidence,
            face,
            parts,
            np.zeros(400),
            triangles,
            binding,
            preserve=preserve,
        )
        np.testing.assert_array_equal(out[10:], color[10:])
        np.testing.assert_array_equal(out[:2], color[:2])
        np.testing.assert_allclose(
            out[2:10], np.tile([0.62, 0.43, 0.33], (8, 1)), atol=1e-7
        )
        self.assertEqual(count, 8)
        self.assertTrue(audit['estimated'])
        self.assertEqual(audit['unresolvedTargetTexels'], 0)

    def test_collinear_ear_observations_retain_prior_without_crashing(self):
        p = np.zeros((472, 3))
        p[468:471] = [[0.08, 0.02, -0.08], [0.07, -0.03, -0.08], [0.075, 0, -0.06]]
        region = {
            'vertices': [468, 469, 470],
            'weights': [1, 1, 1],
            'coreVertices': [468, 469, 470],
            'anchors': dict(zip(('top', 'bottom', 'tragus'), range(468, 471))),
        }
        measured = {
            '1': {
                'landmarks': dict(
                    zip(
                        ('top', 'bottom', 'tragus'),
                        [
                            [0.09, 0.04, -0.10],
                            [0.09, -0.02, -0.10],
                            [0.09, 0.01, -0.10],
                        ],
                    )
                ),
                'audit': {},
            }
        }
        q, audit = fit_ears(p, np.array([[468, 469, 470]]), 0, {'1': region}, measured)
        np.testing.assert_array_equal(q, p)
        self.assertFalse(audit['1']['accepted'])

    def test_watertightness_is_not_a_substitute_for_triangle_orientation(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        f = np.array([[0, 1, 2]])
        self.assertAlmostEqual(
            surface_quality(p * 1e-6, p * 1e-6, f)['minimumNormalAgreement'], 1
        )
        q = p.copy()
        q[2, 1] = -1
        self.assertEqual(surface_quality(p, q, f)['reversedTriangles'], 1)

    def test_ear_triangulation_uses_independent_rays_and_rejects_an_outlier(self):
        target = np.array([0.11, 0.025, -0.13])
        camera = pycolmap.Camera(
            model='SIMPLE_PINHOLE', width=1000, height=1000, params=[700, 500, 500]
        )
        images = {}
        views = []
        crops = {}
        for i, origin in enumerate(
            [
                np.array([0.5, 0.06, 0.2]),
                np.array([0.5, 0.04, -0.3]),
                np.array([0.4, -0.1, 0]),
            ]
        ):
            forward = target - origin
            forward /= np.linalg.norm(forward)
            right = np.cross([0, 1, 0], forward)
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)
            R = np.stack([right, up, forward])
            pose = NS(rotation=NS(matrix=lambda R=R: R), translation=-R @ origin)
            name = f'view{i}'
            images[i] = NS(
                name=name,
                camera_id=1,
                projection_center=lambda origin=origin: origin,
                cam_from_world=lambda pose=pose: pose,
            )
            xy = camera.img_from_cam(R @ target + pose.translation) / 1000
            if i == 2:
                xy += np.array([0.16, -0.18])
            ear = {'visible': True, 'confidence': 0.9, **{k: xy.tolist() for k in KEYS}}
            views.append(
                {
                    'filename': name,
                    'imageLeftEar': ear,
                    'imageRightEar': {'visible': False, 'confidence': 0},
                }
            )
            crops[name] = [0, 0, 1000, 1000]
        with tempfile.TemporaryDirectory() as temp:
            result = triangulate_ears(
                Path(temp),
                NS(images=images, cameras={1: camera}),
                np.zeros(3),
                np.eye(3),
                {'scale': 1},
                {'views': views, 'crops': crops},
            )
        for key in KEYS:
            np.testing.assert_allclose(result['1']['landmarks'][key], target, atol=1e-8)
            self.assertEqual(result['1']['audit'][key]['views'], 2)

    def test_generated_edit_preserves_pixels_and_alpha_outside_supported_region(self):
        rng = np.random.default_rng(4)
        source = rng.integers(0, 255, (100, 100, 4), dtype=np.uint8)
        mask = np.zeros((80, 80), np.uint8)
        mask[20:60, 20:60] = 255
        reference = {
            'crop': [10, 10, 90, 90],
            'mask': mask,
            'pixels': np.full((80, 80, 3), 180, np.uint8),
        }
        result, used = composite_reference(source, reference)
        np.testing.assert_array_equal(result[used == 0], source[used == 0])
        np.testing.assert_array_equal(result[:, :, 3], source[:, :, 3])
        self.assertGreater(np.count_nonzero(result != source), 0)

    def test_clean_reference_restores_enclosed_frame_holes_without_expanding_silhouette(
        self,
    ):
        source = np.zeros((160, 160, 4), np.uint8)
        source[20:140, 20:140, 3] = 255
        source[70:75, 70:75, 3] = 0  # Enclosed rim hole inside verified cleanup.
        source[30:35, 30:35, 3] = 0  # Enclosed, but outside cleanup.
        source[90:95, :80, 3] = 0  # Background connected to the silhouette edge.
        mask = np.zeros((160, 160), np.uint8)
        mask[40:120, 40:120] = 255
        reference = {
            'crop': [0, 0, 160, 160],
            'mask': mask,
            'pixels': np.full((160, 160, 3), 180, np.uint8),
        }
        result, used = composite_reference(source, reference)
        self.assertTrue(np.all(result[70:75, 70:75, 3] == 255))
        np.testing.assert_array_equal(result[used == 0], source[used == 0])
        np.testing.assert_array_equal(result[90:95, :80, 3], source[90:95, :80, 3])
        self.assertTrue(np.all(result[70:75, 70:75, :3] == 180))

    def test_no_glasses_cannot_trigger_image_generation(self):
        with patch('scripts.glasses_reference.urlopen') as remote:
            result = generate_reference(
                Path('/unused'), {'glasses': {'present': False}}
            )
            self.assertFalse(result['available'])
            remote.assert_not_called()

    def test_visible_profile_cleanup_owns_pixels_despite_weak_detail_weight(self):
        # This moderate-angle source previously had facing**8 < .002, so an
        # unmasked rim in another camera overwhelmed its verified cleanup.
        facing = np.array([0.44, 0.44, 0.44, 0.0, 0.44])
        usable = np.array([1.0, 0.0, 0.01, 1.0, 1.0])
        result = surface_coverage(
            np.ones(5), facing, usable, np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        )
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], 0.0)  # Occluded/background samples stay out.
        self.assertLess(result[2], 0.01)  # Almost transparent cutout boundary.
        self.assertEqual(result[3], 0.0)
        self.assertEqual(
            result[4], 0.0
        )  # Profile edits cannot replace the central brow/forehead.

    def test_clear_frame_can_win_nearby_angle_without_losing_rear_coverage(self):
        views = [NS(name=x) for x in ('blur', 'sharp', 'rear')]
        frames = {
            'blur': {'cameraYaw': 0},
            'sharp': {'cameraYaw': 5},
            'rear': {'cameraYaw': 179},
        }
        quality = {
            'blur': {'quality': 0.1},
            'sharp': {'quality': 0.9},
            'rear': {'quality': 0.5},
        }
        self.assertEqual(
            choose_views(views, frames, [0, -180], quality), {'sharp', 'rear'}
        )

    def test_bounded_local_fit_moves_ears_and_pins_facial_controls(self):
        p = np.zeros((475, 3))
        p[468:471] = [[0.08, 0.02, -0.08], [0.07, -0.03, -0.08], [0.075, 0, -0.06]]
        p[471:474] = p[468:471] + [0, 0, -0.004]
        p[474] = [0, 0.1, -0.08]
        region = {
            'vertices': list(range(468, 474)),
            'weights': [1] * 6,
            'coreVertices': list(range(468, 474)),
            'anchors': dict(zip(('top', 'bottom', 'tragus'), range(468, 471))),
        }
        targets = p[468:471] * [1, 1.2, 1] + [-0.01, 0.005, -0.02]
        measured = {
            '1': {
                'landmarks': dict(zip(('top', 'bottom', 'tragus'), targets.tolist())),
                'audit': {},
            }
        }
        q, audit = fit_ears(
            p,
            np.array([[0, 1, 2], [468, 469, 470], [471, 472, 473]]),
            1,
            {'1': region},
            measured,
        )
        self.assertTrue(audit['1']['accepted'])
        self.assertLess(audit['1']['residualMm'], 1)
        np.testing.assert_array_equal(q[:468], p[:468])
        np.testing.assert_array_equal(q[474], p[474])
        self.assertGreater(np.linalg.norm(q[468] - p[468]), 0.01)
        self.assertTrue(np.all(ear_vertex_labels(len(p), {'1': region})[468:474] == 4))
        measured['1']['landmarks'].pop('tragus')
        q, audit = fit_ears(p, np.array([[0, 1, 2]]), 1, {'1': region}, measured)
        np.testing.assert_array_equal(q, p)
        self.assertFalse(audit['1']['accepted'])

    def test_ear_photo_ownership_is_side_aware_at_native_resolution(self):
        ear = {
            'visible': True,
            'confidence': 0.9,
            'outline': [[0.2, 0.2], [0.4, 0.2], [0.4, 0.7], [0.2, 0.7]],
        }
        semantics = {
            'crops': {'side': [0, 0, 200, 200]},
            'views': [
                {
                    'filename': 'side',
                    'imageLeftEar': ear,
                    'imageRightEar': {'visible': False, 'confidence': 0, 'outline': []},
                }
            ],
        }
        mask, found = source_ear_mask(
            (100, 100), 'side', semantics, np.array([0.5, 0, -0.1]), 2
        )
        self.assertTrue(found)
        self.assertEqual(mask[40, 30], 4)
        self.assertEqual(mask[40, 70], 0)
        mask, found = source_ear_mask(
            (100, 100), 'unknown', semantics, np.array([0.5, 0, -0.1]), 2
        )
        self.assertFalse(found)
        self.assertFalse(mask.any())

    def test_ribbon_masks_preserve_lens_interior_and_reject_bad_polygons(self):
        semantics = {
            'crops': {'front': [0, 0, 100, 100]},
            'views': [
                {
                    'filename': 'front',
                    'accessoryConfidence': 0.9,
                    'opaqueGlassesRegions': [
                        [[0.2, 0.2], [0.7, 0.2], [0.7, 0.24], [0.2, 0.24]],
                        [[0, 0], [1, 0], [1, 1], [0, 1]],
                        [[-1, 0], [0.4, 0.3], [0.2, 0.2]],
                    ],
                }
            ],
        }
        mask = semantic_mask((100, 100), 'front', semantics)
        self.assertEqual(mask[22, 40], 255)
        self.assertEqual(mask[40, 40], 0)
        self.assertEqual(mask[90, 90], 0)


class CameraTests(unittest.TestCase):
    def test_calibrated_focal_is_checked_against_ground_truth(self):
        cam = pycolmap.Camera(
            model='SIMPLE_PINHOLE', width=1920, height=1080, params=[1250, 960, 540]
        )
        rec = NS(cameras={1: cam})
        fov = math.degrees(2 * math.atan(1920 / (2 * 1250)))
        self.assertTrue(matches_supplied_fov(rec, {'horizontalFovDegrees': fov}))
        cam.params[0] = 1701
        self.assertFalse(matches_supplied_fov(rec, {'horizontalFovDegrees': fov}))
        rec.cameras[1] = pycolmap.Camera(
            model='SIMPLE_RADIAL', width=1920, height=1080, params=[1250, 960, 540, 0]
        )
        self.assertFalse(matches_supplied_fov(rec, {'horizontalFovDegrees': fov}))

    def test_old_cache_and_legacy_cannot_override_supplied_fov(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            manifest = {
                'horizontalFovDegrees': 60,
                'frames': [{'filename': 'front.png'}],
            }
            (folder / 'capture.json').write_text(json.dumps(manifest))
            (folder / 'photo-cameras').mkdir()
            (folder / 'dataset/sparse/0').mkdir(parents=True)
            (folder / 'images').mkdir()
            Image.new('RGB', (100, 100)).save(folder / 'images/front.png')
            (folder / 'photo-cameras.json').write_text(
                json.dumps(
                    {
                        'captureHash': hashlib.sha256(
                            (folder / 'capture.json').read_bytes()
                        ).hexdigest()
                    }
                )
            )
            old = NS(
                cameras={
                    1: pycolmap.Camera(
                        model='SIMPLE_RADIAL',
                        width=100,
                        height=100,
                        params=[180, 50, 50, -2],
                    )
                }
            )
            with (
                patch(
                    'scripts.photo_cameras.pycolmap.Reconstruction', return_value=old
                ) as read,
                patch(
                    'scripts.photo_cameras.pycolmap.extract_features',
                    side_effect=RuntimeError('recovery-started'),
                ) as extract,
            ):
                with self.assertRaisesRegex(RuntimeError, 'recovery-started'):
                    recover(folder, lambda *args: None)
                self.assertEqual(read.call_count, 1)
                options = extract.call_args.kwargs
                self.assertEqual(options['camera_model'], 'SIMPLE_PINHOLE')
                self.assertAlmostEqual(
                    float(options['reader_options'].camera_params.split(',')[0]),
                    100 / (2 * math.tan(math.pi / 6)),
                )


if __name__ == '__main__':
    unittest.main()
