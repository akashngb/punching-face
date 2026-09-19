"""Native sampling preserves timestamps, orientation and decoded source pixels."""

import base64
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from scripts.video_frames import decode


class VideoFramesTests(unittest.TestCase):
    def test_samples_match_the_source_at_the_reported_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = folder / 'source.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 20, (96, 160))
            self.assertTrue(writer.isOpened())
            for i in range(80):
                frame = np.zeros((160, 96, 3), np.uint8)
                frame[:] = [i * 2, 60, 190]
                frame[10:50, 20:60] = [0, 255, 0]
                writer.write(frame)
            writer.release()
            path.rename(folder / 'source-video')
            result = decode(folder)
            self.assertEqual(len(result['frames']), 48)
            times = [frame['timeSeconds'] for frame in result['frames']]
            self.assertEqual(times, sorted(set(times)))
            source = cv2.VideoCapture(str(folder / 'source-video'))
            try:
                for frame in result['frames']:
                    source.set(cv2.CAP_PROP_POS_FRAMES, round(frame['timeSeconds'] * 20))
                    ok, expected = source.read()
                    self.assertTrue(ok)
                    actual = cv2.imdecode(np.frombuffer(base64.b64decode(frame['image'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
                    self.assertEqual(actual.shape, (160, 96, 3))
                    np.testing.assert_array_equal(actual, expected)
            finally:
                source.release()

    def test_missing_video_fails_without_returning_partial_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                decode(Path(tmp))


if __name__ == '__main__':
    unittest.main()
