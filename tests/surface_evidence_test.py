"""The rendered face must agree with source evidence independently of its cage."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pycolmap
from scripts.surface_evidence import (
    bind_landmarks,
    landmark_positions,
    surface_projection_error,
    validate_refinement_baseline,
)
from scripts.template_selection import choose_regularization, fit_selected_template


class SurfaceEvidenceTests(unittest.TestCase):
    def test_stale_baseline_cannot_restore_different_face_under_current_rig(self):
        import copy

        positions = np.zeros((472, 3))
        current = {
            'positions': positions.ravel().tolist(),
            'indices': [468, 469, 470],
            'stats': {'observedFaceTriangles': 1},
        }
        cage = {'positions': positions[:468].ravel().tolist()}
        baseline = copy.deepcopy(current)
        baseline['positions'][471 * 3] = 0.01  # An ear-only correction is allowed.
        validate_refinement_baseline(current, baseline, cage)
        for index in (0, 469):
            stale = copy.deepcopy(baseline)
            stale['positions'][index * 3] = 0.001
            with self.assertRaisesRegex(ValueError, 'facial surface'):
                validate_refinement_baseline(current, stale, cage)
        cage['positions'][0] = 0.001
        with self.assertRaisesRegex(ValueError, 'physics cage'):
            validate_refinement_baseline(current, baseline, cage)

    def test_surface_binding_follows_mesh_instead_of_correct_but_unused_cage(self):
        vertices = np.array([[0.0, 0.0, 0.0], [0.2, 0, 0], [0.4, 0, 0], [0.2, 0.2, 0]])
        binding = {
            'landmarkIds': [0],
            'vertices': [[1, 2, 3]],
            'weights': [[1.0, 0.0, 0.0]],
        }
        pose = NS(
            rotation=NS(matrix=lambda: np.eye(3)), translation=np.array([0, 0, 2])
        )
        im = NS(name='held', camera_id=1, cam_from_world=lambda: pose)
        rec = NS(
            cameras={
                1: pycolmap.Camera(
                    model='SIMPLE_PINHOLE', width=100, height=100, params=[50, 50, 50]
                )
            }
        )
        frames = {'held': {'landmarks': [{'x': 0.5, 'y': 0.5}]}}
        report = surface_projection_error(
            vertices, binding, rec, frames, [im], np.zeros(3), np.eye(3), {'scale': 1.0}
        )
        self.assertEqual(report['medianPx'], 5.0)
        self.assertTrue(report['usesRenderedSurface'])
        vertices[0] = [100, 100, 100]
        self.assertEqual(
            surface_projection_error(
                vertices,
                binding,
                rec,
                frames,
                [im],
                np.zeros(3),
                np.eye(3),
                {'scale': 1.0},
            )['medianPx'],
            5.0,
        )
        vertices[1:4, 2] = -3
        with self.assertRaises(ValueError):
            surface_projection_error(
                vertices,
                binding,
                rec,
                frames,
                [im],
                np.zeros(3),
                np.eye(3),
                {'scale': 1.0},
            )

    def test_barycentric_binding_preserves_semantic_location_through_deformation(self):
        v = np.array([[0.0, 0.0, 0.0], [1, 0, 0], [0, 1, 0]])
        f = np.array([[0, 1, 2]])
        indices, weights = bind_landmarks(np.array([[0.2, 0.3, 0.1]]), v, f)
        binding = {'vertices': indices, 'weights': weights}
        np.testing.assert_allclose(landmark_positions(v, binding), [[0.2, 0.3, 0]])
        v[:, 2] = [0, 1, 2]
        np.testing.assert_allclose(landmark_positions(v, binding), [[0.2, 0.3, 0.8]])

    def test_selection_views_never_enter_inner_geometry_fit(self):
        frames = {str(i): {'yaw': i} for i in range(20)}
        images = [NS(name=str(i)) for i in range(20)]
        chosen = {im.name for im in images[1::5]}

        def rays(rec, frames, fit):
            self.assertFalse(chosen & {im.name for im in fit})
            self.assertEqual(len(fit), 16)
            return None, None

        def score(p, binding, rec, frames, views, *args):
            self.assertEqual(chosen, {im.name for im in views})
            strength = binding['strength']
            return {'medianPx': 2 + strength * 20, 'p95Px': 6 + strength * 10}

        with (
            patch('scripts.template_selection.camera_rays', side_effect=rays),
            patch(
                'scripts.template_selection.robust_landmarks', return_value=(None, None)
            ),
            patch(
                'scripts.template_selection.normalized_frame',
                return_value=(None, None, None, None),
            ),
            patch(
                'scripts.template_selection.fit_template',
                side_effect=lambda p, s: (
                    None,
                    None,
                    None,
                    {'landmarkBindings': {'strength': s}},
                ),
            ),
            patch(
                'scripts.template_selection.surface_projection_error', side_effect=score
            ),
        ):
            selected, audit = choose_regularization(None, frames, images)
        self.assertEqual(selected, 0.001)
        self.assertFalse(audit['outerWithheldUsed'])
        self.assertFalse(set(audit['trainingFrames']) & set(audit['selectionFrames']))

    def test_a_better_projection_cannot_publish_a_folded_surface(self):
        audit = {'admissibleRegularizations': [0.001, 0.005, 0.025]}

        def fitting(points, strength, check_regularization):
            self.assertTrue(check_regularization)
            if strength == 0.001:
                raise ValueError('Surface crossings')
            return None, None, None, {'regularization': strength}

        with (
            patch(
                'scripts.template_selection.choose_regularization',
                return_value=(0.001, audit),
            ),
            patch('scripts.template_selection.fit_template', side_effect=fitting),
        ):
            result = fit_selected_template(None, None, None, None)
        self.assertEqual(result[3]['regularization'], 0.005)
        self.assertIn(
            'Surface crossings',
            result[3]['selection']['rejectedBySurfaceQuality'][0]['reason'],
        )


if __name__ == '__main__':
    unittest.main()
