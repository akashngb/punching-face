"""Exposure evidence must survive segmentation gaps without reading background."""

import unittest
import numpy as np
from scripts.texture_evidence import (
    photographed_cheek_color,
    snapshot_supported_colors,
    validate_supported_colors,
    SupportedColorSnapshot,
    SupportedColorRegression,
)


class TextureEvidenceTests(unittest.TestCase):
    def fixture(self):
        image = np.full((80, 80, 4), [145, 101, 76, 255], dtype=np.uint8)
        landmarks = [{'x': 0.5, 'y': 0.5} for _ in range(468)]
        for index, (x, y) in zip(
            (50, 280, 205, 425), ((20, 20), (60, 20), (20, 60), (60, 60))
        ):
            landmarks[index] = {'x': x / 80, 'y': y / 80}
        return image, landmarks

    def test_transparent_dark_and_bright_pixels_cannot_bias_skin(self):
        # A nonuniform valid patch makes background contamination shift the
        # median even when valid pixels remain the majority.
        for background in ([0, 0, 0, 0], [255, 255, 255, 219]):
            image, landmarks = self.fixture()
            for x, y in ((20, 20), (60, 20), (20, 60), (60, 60)):
                patch = image[y - 5 : y + 6, x - 5 : x + 6]
                patch[:4] = background
                patch[4:, :, :3] = (
                    np.array([130, 70, 40]) + np.arange(7)[:, None, None] * 10
                )
            expected = np.array([160, 100, 70]) / 255.0
            self.assertFalse(
                np.array_equal(
                    np.median(patch[:, :, :3].reshape(-1, 3), axis=0) / 255.0, expected
                )
            )
            np.testing.assert_array_equal(
                photographed_cheek_color(image, landmarks), expected
            )

    def test_opaque_warm_highlights_and_dark_pixels_preserve_measured_median(self):
        # The actual capture's illuminated cheek has red250..252 without
        # clipping. Rejecting that patch changes the balance between cheeks.
        for palette in (
            [[250, 190, 165], [113, 77, 63], [224, 160, 134], [139, 91, 73]],
            [[3, 4, 5], [30, 25, 20], [80, 65, 50], [120, 100, 85]],
        ):
            image, landmarks = self.fixture()
            patches = []
            for index, (x, y) in enumerate(((20, 20), (60, 20), (20, 60), (60, 60))):
                patch = image[y - 5 : y + 6, x - 5 : x + 6]
                patch[:, :, :3] = (
                    np.asarray(palette[index]) + (np.arange(11) % 3)[:, None, None]
                )
                patches.append(patch[:, :, :3].reshape(-1, 3))
            expected = np.median(np.concatenate(patches), axis=0) / 255.0
            np.testing.assert_array_equal(
                photographed_cheek_color(image, landmarks), expected
            )

    def test_sparse_valid_pixels_and_image_edge_slivers_are_insufficient(self):
        image, landmarks = self.fixture()
        image[:, :, 3] = 0
        for x, y in ((20, 20), (60, 20), (20, 60), (60, 60)):
            image[y, x, 3] = 255
        self.assertIsNone(photographed_cheek_color(image, landmarks))
        image[:, :, 3] = 255
        for index in (50, 280, 205, 425):
            landmarks[index] = {'x': 0, 'y': 0}
        self.assertIsNone(photographed_cheek_color(image, landmarks))

    def test_invalid_required_positions_or_missing_landmarks_return_none(self):
        for value in (-0.01, 1.0, np.nan, np.inf, 'invalid'):
            image, landmarks = self.fixture()
            landmarks[50]['x'] = value
            self.assertIsNone(photographed_cheek_color(image, landmarks))
        image, landmarks = self.fixture()
        self.assertIsNone(photographed_cheek_color(image, landmarks[:280]))

    def test_valid_capture_matches_original_median_and_accepts_indexed_mapping(self):
        image, landmarks = self.fixture()
        patches = []
        for shift, (x, y) in enumerate(((20, 20), (60, 20), (20, 60), (60, 60))):
            image[y - 5 : y + 6, x - 5 : x + 6, :3] += shift * 3
            patches.append(image[y - 5 : y + 6, x - 5 : x + 6, :3].reshape(-1, 3))
        expected = np.median(np.concatenate(patches), axis=0) / 255.0
        np.testing.assert_array_equal(
            photographed_cheek_color(image, landmarks), expected
        )
        indexed = {str(index): landmarks[index] for index in (50, 280, 205, 425)}
        np.testing.assert_array_equal(
            photographed_cheek_color(image, indexed), expected
        )


class SupportedColorTests(unittest.TestCase):
    def snapshot(self, color, **overrides):
        n = len(color)
        args = dict(
            parts=np.zeros(n, np.uint8),
            best=np.full(n, 0.6),
            cleaned_coverage=np.zeros(n),
            bottom=np.zeros(n, bool),
        )
        args.update(overrides)
        return snapshot_supported_colors(color, **args)

    def test_repeated_masked_ears_cannot_silently_overwrite_one_clear_skin_view(self):
        # Recorded failure: one usable skin observation, several photographed
        # ears rejected as occluders, and an incorrect dark posterior-hair prior.
        skin = np.array([[0.754, 0.541, 0.464]])
        snapshot = self.snapshot(skin)
        for masked_ear_weight in (2.201157, 20.0, 2000.0):
            total_clean = 0.833717
            ratio = masked_ear_weight / (masked_ear_weight + total_clean)
            t = np.clip((ratio - 0.2) / 0.6, 0, 1)
            wrong_hair_ownership = t * t * (3 - 2 * t)
            faulty_result = (
                skin * (1 - wrong_hair_ownership)
                + np.array([[0.117, 0.088, 0.066]]) * wrong_hair_ownership
            )
            with self.subTest(masked_ear_weight=masked_ear_weight):
                with self.assertRaises(SupportedColorRegression) as caught:
                    validate_supported_colors(snapshot, faulty_result)
                d = caught.exception.diagnostic
                self.assertEqual(d['changedTexelCount'], 1)
                self.assertEqual(d['changedIds'], [0])
                self.assertEqual(d['samples'][0]['before'], [192, 137, 118])
                self.assertGreater(d['maximumChannelDifference'], 140)
        self.assertTrue(validate_supported_colors(snapshot, skin)['passed'])

    def test_unknown_cleanup_ear_eye_cap_and_late_interiors_are_not_guarded(self):
        color = np.full((8, 3), 0.6)
        snapshot = self.snapshot(
            color,
            parts=np.array([0, 0, 0, 3, 1, 0, 0, 0]),
            best=np.array([0.6, 0.1199, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),
            cleaned_coverage=np.array([0, 0, 0.001, 0, 0, 0, 0, 0]),
            bottom=np.array([False] * 5 + [True, False, False]),
        )
        final = color.copy()
        final[1:] = 0.03
        excluded = np.array([False] * 6 + [True, True])  # late mouth/interior ownership
        audit = validate_supported_colors(snapshot, final, excluded)
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['protectedTexelCount'], 3)
        self.assertEqual(audit['excludedTexelCount'], 2)
        self.assertEqual(audit['checkedTexelCount'], 1)

    def test_quantization_and_clipping_allow_invisible_changes_but_catch_one_byte(self):
        color = np.array([[128.25 / 255, 100.25 / 255, 80.25 / 255], [1.2, -0.1, 0.5]])
        snapshot = self.snapshot(color)
        final = color.copy()
        final[0] += 0.1 / 255
        final[1, :2] = [1.1, -0.2]
        self.assertTrue(validate_supported_colors(snapshot, final)['passed'])
        final[0, 0] += 1 / 255
        audit = validate_supported_colors(snapshot, final, raise_on_change=False)
        self.assertFalse(audit['passed'])
        self.assertEqual(audit['maximumChannelDifference'], 1)

    def test_snapshot_is_independent_and_diagnostics_are_bounded_across_chunks(self):
        color = np.full((65538, 3), 0.6)
        snapshot = self.snapshot(color)
        self.assertFalse(snapshot.ids.flags.writeable)
        self.assertFalse(snapshot.rgb.flags.writeable)
        color[[0, 65536, 65537]] = 0.1
        audit = validate_supported_colors(
            snapshot, color, raise_on_change=False, sample_limit=2
        )
        self.assertEqual(audit['changedTexelCount'], 3)
        self.assertEqual(audit['changedIds'], [0, 65536])
        self.assertEqual(len(audit['samples']), 2)
        import json

        json.dumps(audit)

    def test_empty_snapshot_is_valid_and_malformed_inputs_are_rejected(self):
        empty = self.snapshot(np.empty((0, 3)))
        self.assertTrue(validate_supported_colors(empty, np.empty((0, 3)))['passed'])
        color = np.full((2, 3), 0.5)
        for field, value in [
            ('parts', np.zeros(3, int)),
            ('best', [np.nan, 0.5]),
            ('cleaned_coverage', [0, 1.1]),
            ('bottom', [0, 1]),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.snapshot(color, **{field: value})
        snapshot = self.snapshot(color)
        for value in [np.zeros((2, 4)), np.full((2, 3), np.inf), np.zeros((3, 3))]:
            with self.assertRaises(ValueError):
                validate_supported_colors(snapshot, value)
        for bad in [
            SupportedColorSnapshot(2, [2], np.zeros((1, 3), np.uint8)),
            SupportedColorSnapshot(2, [0, 0], np.zeros((2, 3), np.uint8)),
            SupportedColorSnapshot(2, [0], np.zeros((1, 3), float)),
        ]:
            with self.assertRaises(ValueError):
                validate_supported_colors(bad, color)
        with self.assertRaises(ValueError):
            validate_supported_colors(snapshot, color, exclude=[False])
        with self.assertRaises(ValueError):
            validate_supported_colors(snapshot, color, sample_limit=65)


if __name__ == '__main__':
    unittest.main()
