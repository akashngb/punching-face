"""Accelerators must return exactly what they replace; prefetch must never add an AI request.

.venv/bin/python tests/pipeline_accel_test.py
.venv/bin/python tests/pipeline_accel_test.py .local/face-captures/CAPTURE_ID   # also checks a real recording
"""

import json, sys, tempfile, threading, time, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.pipeline_accel as accel
import scripts.photo_geometry as geometry
import scripts.photo_detail as detail

CAPTURE = None


def reference(module, name):
    return getattr(module, 'reference_' + name, getattr(module, name))


def sphere():
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.1)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


class RasterizerTests(unittest.TestCase):
    def test_depth_is_bit_identical_including_clipped_and_degenerate_triangles(self):
        p, f = sphere()
        rng = np.random.default_rng(3)
        # The camera sits inside the mesh: many triangles are behind it, cross the border or collapse.
        camera = p + [0.03, -0.02, 0.06]
        depth = camera[:, 2]
        projected = camera[:, :2] / np.where(np.abs(depth) < 1e-6, 1e-6, depth)[
            :, None
        ] * 180 + [160, 120]
        f = np.vstack([f, f[:40, [0, 0, 1]], rng.integers(0, len(p), (200, 3))])
        slow = reference(geometry, 'zbuffer')(projected, depth, f, 320, 240)
        fast = accel.rasterize_depth(projected, depth, f, 320, 240)
        self.assertGreater(np.isfinite(slow).sum(), 5000)
        self.assertTrue(np.array_equal(slow, fast))
        for chunk in (7, 1000):
            self.assertTrue(
                np.array_equal(
                    slow,
                    accel.rasterize_depth(projected, depth, f, 320, 240, chunk=chunk),
                )
            )

    def test_cached_depth_is_a_private_copy(self):
        p, f = sphere()
        depth = p[:, 2] + 0.3
        projected = p[:, :2] / depth[:, None] * 300 + [80, 60]
        first = accel.zbuffer(projected, depth, f, 160, 120)
        first[:] = 0
        self.assertTrue(
            np.array_equal(
                accel.zbuffer(projected, depth, f, 160, 120),
                reference(geometry, 'zbuffer')(projected, depth, f, 160, 120),
            )
        )

    def test_atlas_is_bit_identical_with_and_without_part_labels(self):
        import trimesh, xatlas

        p, f = sphere()
        normal = np.array(trimesh.Trimesh(p, f, process=False).vertex_normals)
        atlas = xatlas.Atlas()
        atlas.add_mesh(p.astype(np.float32), f.astype(np.uint32))
        pack = xatlas.PackOptions()
        pack.resolution = 256
        pack.padding = 2
        atlas.generate(pack_options=pack)
        mapping, indices, uv = atlas[0]
        is_face = np.arange(len(indices)) % 3 == 0
        labels = (np.arange(len(p)) % 4).astype(np.uint8)
        for args in [(labels,), ()]:
            slow = reference(geometry, 'raster_atlas')(
                p, normal, mapping, indices, uv, is_face, 256, *args
            )
            for chunk in (5, 1 << 13):
                fast = accel.raster_atlas(
                    p, normal, mapping, indices, uv, is_face, 256, *args, chunk=chunk
                )
                self.assertEqual(len(slow), len(fast))
                for a, b in zip(slow, fast):
                    self.assertTrue(np.array_equal(a, b))
        self.assertGreater(len(slow[0]), 20000)

    def test_atlas_binding_reconstructs_each_texel_on_the_same_surface(self):
        p = np.array(
            [[0.0, 0.0, 0.0], [0.03, 0, 0.01], [0, 0.03, 0.02], [0.03, 0.03, 0.04]]
        )
        normal = np.tile([0.0, 0.0, 1.0], (4, 1))
        mapping = np.arange(4)
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
        labels = np.zeros(4, np.uint8)
        args = (p, normal, mapping, faces, uv, np.ones(2, bool), 40, labels)
        slow = reference(geometry, 'raster_atlas')(*args, return_binding=True)
        fast = accel.raster_atlas(*args, return_binding=True, chunk=1)
        for a, b in zip(slow[:-1], fast[:-1]):
            np.testing.assert_array_equal(a, b)
        for k in slow[-1]:
            np.testing.assert_array_equal(slow[-1][k], fast[-1][k])
        binding = slow[-1]
        vertices = binding['triangles'][binding['triangleIds']]
        reconstructed = np.sum(p[vertices] * binding['weights'][:, :, None], axis=1)
        np.testing.assert_allclose(reconstructed, slow[0], atol=3e-9)

    def test_threaded_sampling_and_tall_product_are_bit_identical(self):
        from scipy.ndimage import map_coordinates

        rng = np.random.default_rng(5)
        image = rng.random((90, 70))
        coordinates = [rng.random(500_000) * 120 - 15, rng.random(500_000) * 100 - 15]
        self.assertTrue(
            np.array_equal(
                map_coordinates(image, coordinates, order=1, mode='nearest'),
                accel.map_coordinates(image, coordinates, order=1, mode='nearest'),
            )
        )
        self.assertTrue(
            np.array_equal(
                map_coordinates(image, [c[:999] for c in coordinates]),
                accel.map_coordinates(image, [c[:999] for c in coordinates]),
            )
        )
        X = rng.random((300_000, 3))
        M = np.linalg.qr(rng.random((3, 3)))[0]
        self.assertTrue(np.array_equal(X @ M, accel.rows(X, M)))
        self.assertTrue(np.array_equal(X @ M.T, accel.rows(X, M.T)))


class PinTests(unittest.TestCase):
    def test_formatting_keeps_compatibility_but_changed_computation_does_not(self):
        from unittest.mock import patch

        sources = [
            'def reference(x): return x+1*2\n',
            'def reference(x):\n    # formatted reference\n    return x + 1 * 2\n',
            'def reference(x): return (x+1)*2\n',
        ]
        digests = []
        for source in sources:
            with patch('scripts.pipeline_accel.inspect.getsource', return_value=source):
                digests.append(accel._source_hash(lambda: None))
        self.assertEqual(digests[0], digests[1])
        self.assertNotEqual(digests[0], digests[2])

    def test_an_edited_reference_is_never_shadowed(self):
        pins = dict(accel.PINS)
        saved = {
            (m, n): getattr(m, n)
            for m, n in [
                (geometry, 'zbuffer'),
                (geometry, 'raster_atlas'),
                (detail, 'prepare_detail_frames'),
            ]
        }
        try:
            accel.PINS.update(zbuffer='edited', prepare_detail_frames='edited')
            active = accel.install_leaves(log=None)
            self.assertIs(geometry.zbuffer, reference(geometry, 'zbuffer'))
            self.assertNotIn('zbuffer', active)
            self.assertNotIn('prepare_detail_frames', active)
            accel.PINS.update(pins)
            active = accel.install_leaves(log=None)
            self.assertIs(geometry.zbuffer, accel.zbuffer)
            self.assertIs(geometry.raster_atlas, accel.raster_atlas)
            self.assertTrue(detail.prepare_detail_frames.locked)
        finally:
            accel.PINS.update(pins)
            for (module, name), value in saved.items():
                setattr(module, name, value)

    def test_current_references_match_their_pins(self):
        stale = [
            name
            for name, (_, function) in accel._references().items()
            if accel._source_hash(function) != accel.PINS[name]
        ]
        self.assertEqual(
            stale,
            [],
            'Port the edit into scripts/pipeline_accel.py, then run it with --pin.',
        )


class PrefetchTests(unittest.TestCase):
    """Stand-ins that cache the way the real stages do: by the ordered list of views."""

    def setUp(self):
        import scripts.astra_head_completion as completion_module, scripts.head_semantics as semantics_module

        self.modules = (completion_module, semantics_module)
        self.saved = (completion_module.request, semantics_module.analyze)
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        (self.folder / 'images').mkdir()
        self.names = [f'frame_{i:04d}.png' for i in range(4)]
        for name in self.names:
            pixels = np.zeros((40, 30, 4), np.uint8)
            pixels[8:30, 5:22] = 255
            Image.fromarray(pixels).save(self.folder / 'images' / name)
        (self.folder / 'capture.json').write_text(
            json.dumps(
                {
                    'captureRegion': 'head',
                    'frames': [{'filename': n} for n in self.names],
                }
            )
        )
        self.requests = []
        self.cache = {}
        self.lock = threading.Lock()

    def tearDown(self):
        self.modules[0].request, self.modules[1].analyze = self.saved
        self.temp.cleanup()

    def stage(self, name, fail_first=False):
        def run(folder, completion, *rest):
            key = (
                name,
                tuple(v['filename'] for v in completion['views']),
                json.dumps(completion['crops'], sort_keys=True),
            )
            with self.lock:
                if key in self.cache:
                    return self.cache[key]
                self.requests.append(name)
                first = self.requests.count(name) == 1
            time.sleep(0.4)
            if fail_first and first:
                raise ValueError('rate limited')
            with self.lock:
                self.cache[key] = {
                    'stage': name,
                    'views': [
                        {
                            'filename': n,
                            'hairRegions': [],
                            'flowPaths': [],
                            'confidence': 1,
                        }
                        for n in key[1]
                    ],
                }
            return self.cache[key]

        return run

    def build(self, shuffle=False, fail_hair=False):
        completion_module, semantics_module = self.modules

        def request(content, *a, **k):
            with self.lock:
                self.requests.append('complete')
            time.sleep(0.4)
            names = self.names[::-1] if shuffle else self.names
            return {'views': [{'filename': n} for n in names]}

        completion_module.request = request

        def complete(folder, evidence, frames=None):
            content = [{'type': 'input_text', 'text': 'prompt'}]
            for n in self.names:
                content += [
                    {
                        'type': 'input_text',
                        'text': f'Filename: {n}; estimated azimuth 0.0 degrees.',
                    },
                    {'type': 'input_image', 'image_url': 'data:'},
                ]
            result = completion_module.request(content)
            result['crops'] = {n: [5, 8, 22, 30] for n in self.names}
            return result

        def scan_eyes(folder, frames, use_astra=True):
            with self.lock:
                if 'eyes' in self.cache:
                    return self.cache['eyes']
                self.requests.append('eyes')
            time.sleep(0.4)
            self.cache['eyes'] = {'eyes': {}}
            return self.cache['eyes']

        semantics_module.analyze = self.stage('semantics')
        build = SimpleNamespace(
            recover=lambda folder, status: ('rec', {}),
            complete=complete,
            recognize_hair=self.stage('hair', fail_hair),
            scan_eyes=scan_eyes,
        )
        self.assertTrue(accel.install_prefetch(build, log=None))
        return build, semantics_module

    def run_pipeline(self, build, semantics_module):
        started = time.perf_counter()
        build.recover(self.folder, None)
        advice = build.complete(self.folder, {})
        hair = build.recognize_hair(self.folder, advice)
        eyes = build.scan_eyes(self.folder, [], True)
        semantics = semantics_module.analyze(self.folder, advice)
        return time.perf_counter() - started, advice, hair, semantics

    def test_four_calls_overlap_and_each_is_requested_once(self):
        seconds, advice, hair, semantics = self.run_pipeline(*self.build())
        self.assertEqual(
            sorted(self.requests), ['complete', 'eyes', 'hair', 'semantics']
        )
        self.assertLess(seconds, 1.0)  # 1.6 s one after another
        self.assertEqual([v['filename'] for v in hair['views']], self.names)

    def test_a_reordered_answer_still_finds_the_prefetched_result(self):
        seconds, advice, hair, semantics = self.run_pipeline(*self.build(shuffle=True))
        self.assertEqual([v['filename'] for v in advice['views']], self.names[::-1])
        self.assertEqual(
            sorted(self.requests), ['complete', 'eyes', 'hair', 'semantics']
        )
        self.assertLess(seconds, 1.0)

    def test_a_failed_prefetch_falls_back_to_the_serial_call(self):
        seconds, advice, hair, semantics = self.run_pipeline(
            *self.build(fail_hair=True)
        )
        self.assertEqual(self.requests.count('hair'), 2)
        self.assertEqual(hair['stage'], 'hair')

    def test_prefetch_can_be_switched_off(self):
        import os

        os.environ['CONTACT_PREFETCH'] = '0'
        try:
            self.assertFalse(accel.install_prefetch(SimpleNamespace(), log=None))
        finally:
            del os.environ['CONTACT_PREFETCH']


@unittest.skipUnless(
    len(sys.argv) > 1 and Path(sys.argv[-1]).is_dir(),
    'pass a capture folder to check a real recording',
)
class RecordingTests(unittest.TestCase):
    def test_streamed_native_frames_are_byte_identical(self):
        import hashlib, shutil

        source = Path(sys.argv[-1]).resolve()
        if not (source / 'source-video').exists():
            self.skipTest('this capture has no recording')
        with tempfile.TemporaryDirectory() as temp:
            outputs = []
            for label, function in [
                ('reference', reference(detail, 'prepare_detail_frames')),
                ('streamed', accel.stream_detail_frames),
            ]:
                folder = Path(temp) / label
                folder.mkdir()
                shutil.copy2(source / 'capture.json', folder / 'capture.json')
                shutil.copy2(source / 'source-video', folder / 'source-video')
                shutil.copytree(source / 'images', folder / 'images')
                started = time.perf_counter()
                audit = function(folder)
                print(f'{label}: {time.perf_counter()-started:.1f}s', flush=True)
                outputs.append(
                    (
                        audit,
                        {
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((folder / 'detail-images').iterdir())
                        },
                    )
                )
            self.assertEqual(outputs[0], outputs[1])
            self.assertGreater(len(outputs[0][1]), 0)


if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])
