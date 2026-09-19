"""Meshy engine: view choice, cutouts, the task lifecycle and its failure modes, against a local
stand-in for the Meshy API. No network, no real key, and nothing is imported from server.py.
"""

import base64, io, json, os, stat, sys, tempfile, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import meshy_backend as mb

KEY = 'msy_unit_test_key_0123456789abcdef'
CAPTURE = 'a' * 32
GLB = b'glTF' + b'\x02\x00\x00\x00' + b'\x00' * 64


def frame(yaw, name, tracked=True):
    return {
        'filename': name,
        'yaw': yaw if tracked else None,
        'landmarks': [{'x': 0.5, 'y': 0.5}] if tracked else None,
    }


class FakeMeshy(BaseHTTPRequestHandler):
    """Records every request; answers like api.meshy.ai for one scripted task."""

    log = []
    script = {}

    def log_message(self, *args):
        pass

    def send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(size))
        FakeMeshy.log.append(('POST', self.path, dict(self.headers), body))
        if FakeMeshy.script.get('create_error'):
            code, message = FakeMeshy.script['create_error']
            return self.send_json(code, {'message': message})
        self.send_json(202, {'result': 'task-0001-abcdef'})

    def do_GET(self):
        FakeMeshy.log.append(('GET', self.path, dict(self.headers), None))
        path = urlparse(self.path).path
        origin = 'http://127.0.0.1:%d' % self.server.server_address[1]
        if path == '/openapi/v1/balance':
            return self.send_json(200, {'balance': 1234})
        if path == '/assets/model.glb':
            self.send_response(200)
            self.send_header('Content-Length', str(len(GLB)))
            self.end_headers()
            return self.wfile.write(GLB)
        if path.endswith('/task-0001-abcdef'):
            polls = sum(1 for e in FakeMeshy.log if e[0] == 'GET' and e[1] == self.path)
            if FakeMeshy.script.get('fail'):
                return self.send_json(
                    200,
                    {
                        'status': 'FAILED',
                        'task_error': {'message': 'The image has no object.'},
                    },
                )
            if polls < 3:
                return self.send_json(
                    200, {'status': 'IN_PROGRESS', 'progress': 40 * polls}
                )
            return self.send_json(
                200,
                {
                    'status': 'SUCCEEDED',
                    'progress': 100,
                    'consumed_credits': 30,
                    'model_urls': {'glb': origin + '/assets/model.glb'},
                },
            )
        self.send_json(404, {'message': 'not found'})


class Store:
    def __init__(self, root):
        self.root = Path(root)

    def folder(self, identifier):
        folder = self.root / str(identifier)
        if not identifier or not folder.is_dir():
            raise ValueError('Face scan not found.')
        return folder


class Views(unittest.TestCase):
    def test_front_is_first_then_both_sides_then_the_far_side(self):
        frames = (
            [frame(8, 'a'), frame(-2, 'front'), frame(-31, 'b'), frame(-58, 'left')]
            + [frame(None, f'rear{i}', tracked=False) for i in range(5)]
            + [frame(61, 'right'), frame(24, 'c')]
        )
        views = mb.select_views(frames)
        self.assertEqual(
            [v['role'] for v in views], ['front', 'side-a', 'side-b', 'far']
        )
        self.assertEqual(
            [frames[v['index']]['filename'] for v in views],
            ['front', 'left', 'right', 'rear2'],
        )

    def test_a_front_only_capture_is_one_view(self):
        views = mb.select_views([frame(3, 'a'), frame(-6, 'b'), frame(12, 'c')])
        self.assertEqual([v['role'] for v in views], ['front'])

    def test_a_short_untracked_gap_is_not_the_far_side(self):
        frames = [
            frame(0, 'a'),
            frame(None, 'x', False),
            frame(None, 'y', False),
            frame(1, 'b'),
        ]
        self.assertEqual([v['role'] for v in mb.select_views(frames)], ['front'])

    def test_no_view_facing_the_camera_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'front of the head'):
            mb.select_views([frame(-40, 'a'), frame(35, 'b')])
        with self.assertRaisesRegex(ValueError, 'faces the camera'):
            mb.select_views([frame(None, 'a', False)])


class Cutout(unittest.TestCase):
    def test_square_centred_and_never_reveals_hidden_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            im = Image.new('RGBA', (640, 360), (0, 0, 0, 0))
            im.paste((200, 150, 120, 255), (500, 40, 630, 240))  # head near the edge
            path = Path(temp) / 'frame.png'
            im.save(path)
            out = Image.open(io.BytesIO(mb.prepare_view(path)))
        self.assertEqual(out.mode, 'RGBA')
        self.assertEqual(out.width, out.height)
        self.assertEqual(out.width, round(200 * 1.16))
        alpha = out.getchannel('A')
        self.assertEqual(alpha.getpixel((0, 0)), 0)
        self.assertEqual(alpha.getpixel((out.width // 2, out.height // 2)), 255)
        left, top, right, bottom = alpha.getbbox()
        self.assertLessEqual(right - left, 130)  # the matte only ever shrinks
        self.assertLessEqual(bottom - top, 200)
        self.assertAlmostEqual((left + right) / 2, out.width / 2, delta=2)
        self.assertAlmostEqual((top + bottom) / 2, out.height / 2, delta=2)

    def test_large_heads_are_limited_and_empty_masks_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            big = Path(temp) / 'big.png'
            Image.new('RGBA', (1600, 1600), (90, 80, 70, 255)).save(big)
            self.assertEqual(
                Image.open(io.BytesIO(mb.prepare_view(big))).size, (1024, 1024)
            )
            empty = Path(temp) / 'empty.png'
            Image.new('RGBA', (64, 64), (0, 0, 0, 0)).save(empty)
            with self.assertRaisesRegex(ValueError, 'empty head mask'):
                mb.prepare_view(empty)


class Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = ThreadingHTTPServer(('127.0.0.1', 0), FakeMeshy)
        threading.Thread(target=cls.api.serve_forever, daemon=True).start()
        cls.base = 'http://127.0.0.1:%d/openapi/v1' % cls.api.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.api.shutdown()
        cls.api.server_close()

    def setUp(self):
        FakeMeshy.log = []
        FakeMeshy.script = {}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.folder = root / CAPTURE
        (self.folder / 'images').mkdir(parents=True)
        frames = []
        for name, yaw in [('f0.png', 1.0), ('f1.png', -55.0), ('f2.png', 48.0)]:
            im = Image.new('RGBA', (320, 240), (0, 0, 0, 0))
            im.paste((180, 140, 110, 255), (110, 40, 210, 200))
            im.save(self.folder / 'images' / name)
            frames.append(frame(yaw, name))
        (self.folder / 'capture.json').write_text(json.dumps({'frames': frames}))
        for p in (
            patch.dict(os.environ, {'MESHY_API_KEY': KEY, 'MESHY_API_BASE': self.base}),
            patch.object(mb, 'CONFIG', root / 'secrets' / 'meshy.json'),
            patch.object(mb, 'POLL_SECONDS', 0.01),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.engine = mb.MeshyEngine(Store(root))

    def finish(self, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.engine.job(CAPTURE)
            if state['status'] != 'running':
                return state
            time.sleep(0.02)
        self.fail('The Meshy job did not finish.')

    def creates(self):
        return [e for e in FakeMeshy.log if e[0] == 'POST']

    def test_builds_a_model_from_the_front_first_views(self):
        self.assertEqual(self.engine.job(CAPTURE)['status'], 'idle')
        self.assertEqual(self.engine.train(CAPTURE)['status'], 'running')
        state = self.finish()
        self.assertEqual(state['status'], 'complete', state)
        self.assertTrue(state['model'])
        self.assertEqual((self.folder / 'meshy/model.glb').read_bytes(), GLB)
        self.assertEqual(state['result']['consumedCredits'], 30)
        self.assertEqual(
            [v['filename'] for v in state['views']], ['f0.png', 'f1.png', 'f2.png']
        )
        (create,) = self.creates()
        self.assertEqual(create[1], '/openapi/v1/multi-image-to-3d')
        self.assertEqual(create[2]['Authorization'], 'Bearer ' + KEY)
        self.assertEqual(len(create[3]['image_urls']), 3)
        sent = base64.b64decode(create[3]['image_urls'][0].split(',', 1)[1])
        self.assertEqual(sent, (self.folder / 'meshy/views/view-0.png').read_bytes())
        self.assertEqual(create[3]['target_formats'], ['glb'])
        # The signed asset host never sees the API key.
        (asset,) = [e for e in FakeMeshy.log if e[1] == '/assets/model.glb']
        self.assertNotIn('Authorization', asset[2])
        self.assertNotIn(KEY, json.dumps(state))

    def test_a_finished_model_is_reused_until_a_rebuild_is_asked_for(self):
        self.engine.train(CAPTURE)
        self.finish()
        self.assertTrue(self.engine.train(CAPTURE)['reused'])
        self.assertEqual(len(self.creates()), 1)
        self.engine.train(CAPTURE, rebuild=True)
        self.assertEqual(self.finish()['status'], 'complete')
        self.assertEqual(len(self.creates()), 2)

    def test_a_single_view_uses_the_single_image_endpoint(self):
        frames = json.loads((self.folder / 'capture.json').read_text())['frames'][:1]
        (self.folder / 'capture.json').write_text(json.dumps({'frames': frames}))
        self.engine.train(CAPTURE)
        self.assertEqual(self.finish()['status'], 'complete')
        (create,) = self.creates()
        self.assertEqual(create[1], '/openapi/v1/image-to-3d')
        self.assertIn('image_url', create[3])

    def test_a_restarted_server_follows_the_same_task_without_new_credits(self):
        (self.folder / 'meshy').mkdir()
        (self.folder / 'meshy/job.json').write_text(
            json.dumps(
                {
                    'status': 'running',
                    'taskId': 'task-0001-abcdef',
                    'endpoint': 'multi-image-to-3d',
                    'requestedAt': time.time(),
                }
            )
        )
        state = self.finish()
        self.assertEqual(state['status'], 'complete', state)
        self.assertEqual(self.creates(), [])

    def test_an_interrupted_upload_is_reported_as_not_charged(self):
        (self.folder / 'meshy').mkdir()
        (self.folder / 'meshy/job.json').write_text(json.dumps({'status': 'running'}))
        state = self.engine.job(CAPTURE)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('Nothing was charged', state['message'])

    def test_a_failed_task_keeps_the_reason_and_leaves_no_model(self):
        FakeMeshy.script = {'fail': True}
        self.engine.train(CAPTURE)
        state = self.finish()
        self.assertEqual(state['status'], 'failed')
        self.assertIn('The image has no object.', state['message'])
        self.assertFalse(state['model'])

    def test_provider_errors_are_explained_and_never_echo_a_key(self):
        FakeMeshy.script = {'create_error': (402, 'Insufficient funds for ' + KEY)}
        self.engine.train(CAPTURE)
        state = self.finish()
        self.assertEqual(state['status'], 'failed')
        self.assertIn('out of credits', state['message'])
        self.assertNotIn(KEY, state['message'])

    def test_missing_key_and_unknown_scans_are_refused_before_any_request(self):
        with self.assertRaisesRegex(ValueError, 'not found'):
            self.engine.train('b' * 32)
        with patch.dict(os.environ, {'MESHY_API_KEY': ''}):
            with self.assertRaisesRegex(ValueError, 'not configured'):
                self.engine.train(CAPTURE)
            self.assertFalse(self.engine.status()['configured'])
        self.assertEqual(FakeMeshy.log, [])

    def test_status_reports_the_key_source_and_credit_balance(self):
        status = self.engine.status(balance=True)
        self.assertEqual(
            (status['configured'], status['source'], status['balance']),
            (True, 'environment', 1234),
        )
        self.assertNotIn(KEY, json.dumps(status))

    def test_a_saved_key_is_private_and_malformed_keys_are_refused(self):
        with patch.dict(os.environ, {'MESHY_API_KEY': ''}):
            for bad in (None, '', 'sk-not-a-meshy-key-000000000', 'msy_short'):
                with self.assertRaises(ValueError):
                    mb.configure(bad)
            mb.configure(KEY)
            self.assertEqual(mb.api_key(), KEY)
            self.assertEqual(mb.key_source(), 'saved')
            self.assertEqual(stat.S_IMODE(mb.CONFIG.stat().st_mode), 0o600)


class Routes(unittest.TestCase):
    """The HTTP surface server.py exposes, behind the same local-only rule as the face routes."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / CAPTURE / 'meshy').mkdir(parents=True)
        (root / CAPTURE / 'meshy/model.glb').write_bytes(GLB)
        engine = mb.MeshyEngine(Store(root))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, data):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                engine.handle(self, urlparse(self.path))

            do_POST = do_GET

        for p in (
            patch.dict(os.environ, {'MESHY_API_KEY': ''}),
            patch.object(mb, 'CONFIG', root / 'secrets' / 'meshy.json'),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = 'http://127.0.0.1:%d' % self.server.server_address[1]

    def get(self, path, headers=None):
        return urlopen(Request(self.base + path, headers=headers or {}), timeout=10)

    def post(self, path, data):
        return urlopen(
            Request(
                self.base + path,
                data=json.dumps(data).encode(),
                headers={'Content-Type': 'application/json'},
            ),
            timeout=10,
        )

    def test_status_job_and_model_are_served(self):
        self.assertFalse(json.load(self.get('/api/meshy-status'))['configured'])
        job = json.load(self.get('/api/meshy-job?id=' + CAPTURE))
        self.assertEqual((job['status'], job['model']), ('idle', True))
        reply = self.get(f'/api/meshy-asset?id={CAPTURE}&asset=model.glb')
        self.assertEqual(reply.headers['Content-Type'], 'model/gltf-binary')
        self.assertEqual(reply.read(), GLB)

    def test_a_key_saved_from_the_dialog_is_never_returned(self):
        reply = json.load(self.post('/api/meshy-config', {'apiKey': KEY}))
        self.assertEqual((reply['configured'], reply['source']), (True, 'saved'))
        self.assertNotIn(KEY, json.dumps(reply))

    def test_bad_requests_are_refused(self):
        for path in (
            '/api/meshy-asset?id=' + CAPTURE + '&asset=../capture.json',
            '/api/meshy-asset?id=' + CAPTURE + '&asset=thumbnail.png',
            '/api/meshy-job?id=../../etc',
        ):
            with self.assertRaises(HTTPError) as caught:
                self.get(path)
            self.assertEqual(caught.exception.code, 422)
        with self.assertRaises(HTTPError) as caught:
            self.post('/api/meshy-train', {'id': CAPTURE})
        self.assertIn('not configured', json.load(caught.exception)['error'])
        with self.assertRaises(HTTPError) as caught:
            self.get('/api/meshy-status', {'Host': 'evil.example'})
        self.assertEqual(caught.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
