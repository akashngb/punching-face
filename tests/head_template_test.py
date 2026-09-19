"""Topology and regularity checks for the full-head identity fit."""

from pathlib import Path
import sys, unittest
import numpy as np, trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fit_head_template import fit_template


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lines = (
            (
                Path(__file__).resolve().parents[1]
                / 'public/models/canonical_face_model.obj'
            )
            .read_text()
            .splitlines()
        )
        points = np.array(
            [list(map(float, l.split()[1:4])) for l in lines if l.startswith('v ')]
        )
        points -= (points[10] + points[152]) / 2
        points *= 0.20 / (points[10, 1] - points[152, 1])
        cls.points = points
        cls.p, cls.f, cls.count, cls.info = fit_template(points)

    def test_full_head_has_closed_finite_geometry_and_meaningful_depth(self):
        m = trimesh.Trimesh(self.p, self.f, process=False)
        self.assertTrue(m.is_watertight)
        self.assertTrue(np.isfinite(self.p).all())
        self.assertGreater(np.ptp(self.p[:, 2]), 0.15)
        self.assertGreater(len(self.p), 15000)
        self.assertGreater(self.count, 1000)
        # The lower edge is a planar cut, not the sawtooth from whole-face crop.
        bottom = self.p[:, 1].min()
        self.assertGreater(np.sum(abs(self.p[:, 1] - bottom) < 1e-7), 20)

    def test_semantic_cage_order_survives_template_topology_changes(self):
        np.testing.assert_allclose(self.p[:468, :2], self.points[:, :2])
        self.assertGreaterEqual(self.f.min(), 468)
        self.assertEqual(self.info['template'], 'MakeHuman hm08, CC0')

    def test_validation_landmarks_are_attached_to_rendered_skin_triangles(self):
        binding = self.info['landmarkBindings']
        vertices = np.asarray(binding['vertices'])
        weights = np.asarray(binding['weights'])
        self.assertEqual(len(vertices), self.info['landmarkAnchors'])
        self.assertGreaterEqual(vertices.min(), 468)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        triangles = {tuple(sorted(face)) for face in self.f}
        self.assertTrue(all(tuple(sorted(face)) in triangles for face in vertices))


if __name__ == '__main__':
    unittest.main()
