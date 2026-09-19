"""Central facial features and separate surface parts keep their own appearance."""

import unittest
import numpy as np
from scripts.camera_texture import lower_lateral_region


class CameraTextureTests(unittest.TestCase):
    def test_central_features_and_separate_parts_are_excluded(self):
        face = np.zeros((468, 3))
        face[0, 0] = -0.1
        face[1] = [0, 0, 0]
        face[2, 0] = 0.1
        face[17, 1] = -0.05
        face[152, 1] = -0.1
        points = np.array(
            [
                [0, 0, 0],
                [0, -0.05, 0],
                [0.09, 0.05, 0],
                [0.09, -0.03, 0],
                [0.09, -0.03, 0],
            ]
        )
        result = lower_lateral_region(points, face, np.array([0, 0, 0, 0, 3]))
        np.testing.assert_array_equal(result[[0, 1, 2, 4]], 0)
        self.assertGreater(result[3], 0.99)


if __name__ == '__main__':
    unittest.main()
