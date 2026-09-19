"""Small, real artifact bundles exercise publication without a capture bake."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import head_artifacts as artifacts


def png(color):
    def chunk(kind, payload):
        return (
            struct.pack('>I', len(payload))
            + kind
            + payload
            + struct.pack('>I', zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(bytes([0] + color)))
        + chunk(b'IEND', b'')
    )


def bundle(folder, shift=0):
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype='<f4') + shift
    flat = positions.ravel().tolist()
    anchors = {'0': positions[0].tolist()}
    mesh = {'positions': flat, 'indices': [0, 1, 2], 'rigAnchors': anchors}
    cage = {'positions': flat, 'indices': [0, 1, 2], 'rigAnchors': anchors}
    binding = {
        'indices': [0, 1, 2] * 3,
        'weights': [1, 0, 0, 0, 1, 0, 0, 0, 1],
        'active': [1, 1, 1],
    }
    image = png([100 + shift, 80, 70])
    (folder / 'appearance.png').write_bytes(image)
    atlas = {
        'mapping': [0, 1, 2],
        'indices': [0, 1, 2],
        'uv': [0, 0, 1, 0, 0, 1],
        'texture': '/old-url',
        'positionsSha256': hashlib.sha256(positions.tobytes()).hexdigest(),
        'textureSha256': hashlib.sha256(image).hexdigest(),
    }
    for name, value in [
        ('mesh.json', mesh),
        ('physics-cage.json', cage),
        ('physics-binding.json', binding),
        ('texture-atlas.json', atlas),
    ]:
        (folder / name).write_text(json.dumps(value))


class HeadArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / ('a' * 32)
        self.folder.mkdir()
        bundle(self.folder)
        self.legacy = self.hashes(self.folder)

    def hashes(self, folder):
        return {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in artifacts.ARTIFACT_NAMES
            if (folder / name).is_file()
        }

    def publish(self, shift=0):
        with artifacts.HeadArtifactTransaction(self.folder) as tx:
            bundle(tx.stage, shift)
            return tx.commit()

    def test_failed_partial_staging_preserves_accepted_legacy(self):
        with self.assertRaisesRegex(ValueError, 'Missing required'):
            with artifacts.HeadArtifactTransaction(self.folder) as tx:
                stage = tx.stage
                (stage / 'mesh.json').write_text('{}')
                tx.commit()
        self.assertFalse(stage.exists())
        self.assertEqual(self.hashes(self.folder), self.legacy)
        self.assertEqual(
            artifacts.release_manifest(self.folder),
            {'version': 1, 'generation': 'legacy'},
        )
        self.assertTrue(artifacts.has_published_model(self.folder))
        with self.assertRaisesRegex(RuntimeError, 'compute'):
            with artifacts.HeadArtifactTransaction(self.folder, seed=True) as tx:
                raise RuntimeError('compute failed')
        self.assertEqual(self.hashes(self.folder), self.legacy)

    def test_pointer_write_failure_retains_old_release(self):
        old = self.publish()
        with artifacts.HeadArtifactTransaction(self.folder, seed=True) as tx:
            with patch.object(
                artifacts, '_publish_pointer', side_effect=OSError('disk error')
            ):
                with self.assertRaisesRegex(OSError, 'disk error'):
                    tx.commit()
            self.assertFalse(
                (self.folder / '.model-generations' / tx.generation).exists()
            )
        self.assertEqual(artifacts.release_manifest(self.folder), old)
        self.assertTrue(artifacts.has_published_model(self.folder))
        self.assertEqual(self.hashes(self.folder), self.legacy)

    def test_pinned_generation_remains_readable_after_next_commit(self):
        first = self.publish()
        first_path = artifacts.published_folder(self.folder, first['generation'])
        first_hashes = self.hashes(first_path)
        second = self.publish(1)
        self.assertNotEqual(first['generation'], second['generation'])
        self.assertEqual(self.hashes(first_path), first_hashes)
        self.assertEqual(
            artifacts.published_folder(self.folder).name, second['generation']
        )
        self.assertEqual(artifacts.published_folder(self.folder, 'legacy'), self.folder)
        self.assertEqual(self.hashes(self.folder), self.legacy)
        self.assertEqual(first['files'], first_hashes)
        atlas = json.loads((first_path / 'texture-atlas.json').read_text())
        self.assertIn('id=' + self.folder.name, atlas['texture'])
        self.assertIn('generation=' + first['generation'], atlas['texture'])
        (self.folder / 'status.json').write_text('{"status":"failed"}')
        self.assertTrue(artifacts.has_published_model(self.folder))

    def test_concurrent_writer_and_legacy_mutation_rejected(self):
        with artifacts.HeadArtifactTransaction(self.folder, seed=True) as first:
            with artifacts.HeadArtifactTransaction(self.folder, seed=True) as second:
                accepted = first.commit()
                with self.assertRaisesRegex(ValueError, 'changed during'):
                    second.commit()
        self.assertEqual(artifacts.release_manifest(self.folder), accepted)
        # A different capture still using root artifacts gets the same guard.
        other = self.folder.parent / ('b' * 32)
        other.mkdir()
        bundle(other)
        with artifacts.HeadArtifactTransaction(other, seed=True) as tx:
            (other / 'ear-measurements.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed during'):
                tx.commit()
        self.assertFalse((other / 'model-release.json').exists())

    def test_inconsistent_cage_geometry_and_texture_hash_rejected(self):
        for filename, field, value, error in (
            ('physics-cage.json', 'positions', [1] * 9, 'cage'),
            ('texture-atlas.json', 'positionsSha256', '0' * 64, 'geometry hash'),
            ('texture-atlas.json', 'textureSha256', '0' * 64, 'texture hash'),
            ('texture-atlas.json', 'indices', [2, 1, 0], 'topology'),
            ('physics-binding.json', 'weights', [1] * 9, 'weights'),
        ):
            with self.subTest(field=field):
                with artifacts.HeadArtifactTransaction(self.folder, seed=True) as tx:
                    path = tx.stage / filename
                    data = json.loads(path.read_text())
                    data[field] = value
                    path.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ValueError, error):
                        tx.commit()
                self.assertEqual(self.hashes(self.folder), self.legacy)

    def test_truncated_png_rejected_even_with_matching_hash(self):
        with artifacts.HeadArtifactTransaction(self.folder, seed=True) as tx:
            image = tx.stage / 'appearance.png'
            image.write_bytes(image.read_bytes()[:-3])
            atlas_path = tx.stage / 'texture-atlas.json'
            atlas = json.loads(atlas_path.read_text())
            atlas['textureSha256'] = hashlib.sha256(image.read_bytes()).hexdigest()
            atlas_path.write_text(json.dumps(atlas))
            with self.assertRaisesRegex(ValueError, 'Truncated PNG'):
                tx.commit()

    def test_nonfinite_bounds_and_missing_roughness_rejected(self):
        for field, value in (
            ('uv', [0, 0, 2, 0, 0, 1]),
            ('uv', [0, 0, float('nan'), 0, 0, 1]),
            ('roughnessTexture', '/missing.png'),
        ):
            with self.subTest(field=field, value=value):
                with artifacts.HeadArtifactTransaction(self.folder, seed=True) as tx:
                    path = tx.stage / 'texture-atlas.json'
                    data = json.loads(path.read_text())
                    data[field] = value
                    path.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):
                        tx.commit()

    def test_interruption_after_pointer_swap_does_not_delete_published_files(self):
        publish = artifacts._publish_pointer

        def interrupted(folder, manifest):
            publish(folder, manifest)
            raise KeyboardInterrupt()

        with artifacts.HeadArtifactTransaction(self.folder, seed=True) as tx:
            with patch.object(artifacts, '_publish_pointer', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    tx.commit()
            self.assertEqual(
                artifacts.published_folder(self.folder).name, tx.generation
            )
            self.assertTrue(artifacts.has_published_model(self.folder))

    def test_only_artifacts_sealed_and_baseline_checked(self):
        with artifacts.HeadArtifactTransaction(self.folder, seed=True) as tx:
            (tx.stage / 'images').symlink_to(self.folder, target_is_directory=True)
            positions = np.array(
                json.loads((tx.stage / 'mesh.json').read_text())['positions']
            ).reshape(-1, 3)
            np.savez(
                tx.stage / 'pre-ear-surface.npz',
                positions=positions,
                indices=[[0, 1, 2]],
            )
            release = tx.commit()
        path = artifacts.published_folder(self.folder)
        self.assertFalse((path / 'images').exists())
        self.assertIn('pre-ear-surface.npz', release['files'])
        with artifacts.HeadArtifactTransaction(self.folder, seed=True) as tx:
            np.savez(
                tx.stage / 'pre-ear-surface.npz',
                positions=positions + 1,
                indices=[[0, 1, 2]],
            )
            with self.assertRaisesRegex(ValueError, 'protected face'):
                tx.commit()

    def test_invalid_or_unsealed_generation_rejected(self):
        for token in ('../outside', 'A' * 32, '', 'not-a-generation'):
            with self.subTest(token=token):
                with self.assertRaises(ValueError):
                    artifacts.published_folder(self.folder, token)
        with self.assertRaises(ValueError):
            artifacts.published_folder(self.folder, 'c' * 32)
        for name in artifacts.REQUIRED_NAMES:
            with self.subTest(missing=name):
                path = self.folder / name
                content = path.read_bytes()
                path.unlink()
                self.assertFalse(artifacts.has_published_model(self.folder))
                path.write_bytes(content)


if __name__ == '__main__':
    unittest.main()
