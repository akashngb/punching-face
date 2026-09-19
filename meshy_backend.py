"""Meshy cloud reconstruction: the alternative engine for a saved head scan.

The PunchingFace pipeline (scripts/build_photo_face*.py) stays the default and runs on this computer.
This module sends up to four cropped views of one capture to Meshy's multi-image-to-3D API, follows
the task, and keeps the GLB beside the capture, so both engines can be compared on the same scan. It
never produces geometry itself. API reference: https://docs.meshy.ai/en/api/multi-image-to-3d
"""

import base64, io, json, os, re, socket, threading, time, traceback
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs
from urllib.request import Request, urlopen
from PIL import Image, ImageChops, ImageFilter

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / '.local/secrets/meshy.json'
ROUTES = {
    '/api/meshy-status',
    '/api/meshy-config',
    '/api/meshy-train',
    '/api/meshy-job',
    '/api/meshy-asset',
}
ASSETS = {
    'model.glb': 'model/gltf-binary',
    'thumbnail.png': 'image/png',
    **{f'view-{i}.png': 'image/png' for i in range(4)},
}
# Textured Meshy 6/7 image-to-3D task (https://docs.meshy.ai/en/api/pricing). The finished task reports the real figure.
ESTIMATED_CREDITS = 30
POLL_SECONDS = 3
TASK_TIMEOUT = 1200


class ApiError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Unreachable(Exception):
    pass


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, allow_nan=False))
    tmp.replace(path)


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def api_base():
    return os.environ.get('MESHY_API_BASE', 'https://api.meshy.ai/openapi/v1').rstrip(
        '/'
    )


def key_source():
    if os.environ.get('MESHY_API_KEY', '').strip():
        return 'environment'
    return 'saved' if read(CONFIG).get('apiKey') else None


def api_key():
    # Precedence matches the other providers here: process env > .env > .local/secrets.
    return (
        os.environ.get('MESHY_API_KEY', '').strip() or read(CONFIG).get('apiKey', '')
    ).strip()


def configure(key):
    if not isinstance(key, str) or not re.fullmatch(r'msy_[A-Za-z0-9_-]{16,250}', key):
        raise ValueError(
            'Enter a Meshy API key (it starts with msy_). It is saved only on this server.'
        )
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.parent.chmod(0o700)
    fd = os.open(CONFIG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump({'apiKey': key}, f)
    CONFIG.chmod(0o600)


def call(path, payload=None, timeout=60):
    key = api_key()
    if not key:
        raise ValueError(
            'Meshy is not configured. Add MESHY_API_KEY to .env or save a key under Meshy API settings.'
        )
    headers = {
        'Authorization': 'Bearer ' + key,
        'Accept': 'application/json',
        'User-Agent': 'punching-face/1.0',
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    try:
        with urlopen(
            Request(api_base() + path, data=data, headers=headers), timeout=timeout
        ) as r:
            body = r.read()
    except HTTPError as e:
        known = {
            401: 'Meshy rejected the API key. Replace it under Meshy API settings.',
            402: 'This Meshy account is out of credits.',
            403: 'This Meshy key is not allowed to use the API. Check the plan on the Meshy console.',
            429: 'Meshy rate limit reached. Wait a minute and try again.',
        }
        detail = ''
        try:
            detail = str(json.loads(e.read().decode('utf-8', 'ignore')).get('message'))
        except (ValueError, AttributeError, OSError):
            pass
        # Provider text is shown to the user, so it is shortened and can never carry a key back out.
        detail = re.sub(r'msy_[A-Za-z0-9_-]+', 'msy_…', detail)[:240]
        message = known.get(e.code) or f'Meshy returned HTTP {e.code}.'
        if detail and detail != 'None' and e.code not in (401,):
            message += ' ' + detail
        raise ApiError(e.code, message) from None
    except (URLError, TimeoutError, socket.timeout, ConnectionError):
        raise Unreachable('Meshy could not be reached from this computer.') from None
    try:
        reply = json.loads(body)
    except ValueError:
        reply = None
    if not isinstance(reply, dict):
        raise Unreachable('Meshy sent a reply this server could not read.')
    return reply


def asset_url(value):
    # Meshy hands out signed https URLs. Loopback http only ever occurs against the test double.
    local = 'http://127.0.0.1:'
    if isinstance(value, str) and (
        value.startswith('https://')
        or (value.startswith(local) and api_base().startswith(local))
    ):
        return value
    return None


def select_views(frames, limit=4):
    """Pick the views Meshy sees. The first must face the camera: Meshy treats it as the primary view.

    Tracked frames carry a yaw from the face landmarks (reliable to about 60 degrees). Frames without
    landmarks are profile and rear views whose angle is unknown until the local pipeline recovers cameras,
    so the far side is taken as the middle of the longest untracked run of a continuous turn.
    """
    tracked = [
        (i, f['yaw'])
        for i, f in enumerate(frames)
        if f.get('landmarks') and isinstance(f.get('yaw'), (int, float))
    ]
    if not tracked:
        raise ValueError(
            'Meshy needs at least one saved view that faces the camera. Record or import the front of the head first.'
        )
    front = min(tracked, key=lambda t: abs(t[1]))
    if abs(front[1]) > 20:
        raise ValueError(
            'No saved view faces the camera closely enough for Meshy. Capture the front of the head.'
        )
    views = [{'index': front[0], 'role': 'front', 'yaw': front[1]}]
    for role, sign in (('side-a', -1), ('side-b', 1)):
        side = [t for t in tracked if t[1] * sign >= abs(front[1]) + 20]
        if side:
            # The widest tracked turn adds the most new surface; 60 degrees is where tracking stays reliable.
            pick = min(side, key=lambda t: abs(t[1] * sign - 60))
            views.append({'index': pick[0], 'role': role, 'yaw': pick[1]})
    run, best = [], []
    for i, f in enumerate(frames):
        if f.get('landmarks'):
            run = []
            continue
        run.append(i)
        if len(run) > len(best):
            best = list(run)
    if len(best) >= 3:
        views.append({'index': best[len(best) // 2], 'role': 'far', 'yaw': None})
    return views[:limit]


def prepare_view(path, size=1024):
    """Square, centred, transparent-background PNG of one masked capture frame."""
    with Image.open(path) as source:
        im = source.convert('RGBA')
    alpha = im.getchannel('A')
    box = alpha.getbbox()
    if not box:
        raise ValueError('A selected view has an empty head mask.')
    # Saved mattes are binary and the hidden RGB is zero. Pull the edge in by a pixel and feather it so the
    # cutout has no hard black rim, without ever revealing a pixel the capture discarded.
    soft = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    im.putalpha(ImageChops.darker(soft, alpha))
    left, top, right, bottom = box
    side = max(64, int(round(max(right - left, bottom - top) * 1.16)))
    x0 = int(round((left + right) / 2 - side / 2))
    y0 = int(round((top + bottom) / 2 - side / 2))
    im = im.crop((x0, y0, x0 + side, y0 + side))  # out-of-frame area stays transparent
    if side > size:
        # Premultiplied resize: straight-alpha filtering would bleed the zeroed background into the rim.
        im = im.convert('RGBa').resize((size, size), Image.LANCZOS).convert('RGBA')
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return buf.getvalue()


def polycount():
    try:
        return min(300_000, max(100, int(os.environ.get('MESHY_TARGET_POLYCOUNT', ''))))
    except ValueError:
        return 30_000


def task_payload(images):
    uris = ['data:image/png;base64,' + base64.b64encode(raw).decode() for raw in images]
    payload = {
        'ai_model': os.environ.get('MESHY_AI_MODEL', 'latest'),
        'should_texture': True,
        'enable_pbr': os.environ.get('MESHY_ENABLE_PBR') == '1',
        'should_remesh': True,
        'topology': 'triangle',
        # Matches the single-photo flow in server.py: the browser impact rig deforms every vertex on the CPU.
        'target_polycount': polycount(),
        'target_formats': ['glb'],
    }
    if len(uris) == 1:
        return 'image-to-3d', {**payload, 'image_url': uris[0]}
    return 'multi-image-to-3d', {**payload, 'image_urls': uris}


class MeshyEngine:
    def __init__(self, store):
        self.store = (
            store  # FaceStore: validates identifiers and owns the capture folders
        )
        self.lock = threading.Lock()
        self.active = {}

    def status(self, balance=False):
        key = api_key()
        result = {
            'configured': bool(key),
            'source': key_source(),
            'aiModel': os.environ.get('MESHY_AI_MODEL', 'latest'),
            'estimatedCredits': ESTIMATED_CREDITS,
            'maxViews': 4,
        }
        if balance and key:
            try:
                value = call('/balance', timeout=20).get('balance')
                result['balance'] = value if isinstance(value, (int, float)) else None
                result['connected'] = True
            except (ValueError, Unreachable) as exc:
                result.update(connected=False, error=str(exc))
        return result

    def job(self, identifier):
        folder = self.store.folder(identifier)
        # Liveness first: a worker writes its final state before it leaves `active`, so a finished
        # job can never be read here as "running with no worker".
        with self.lock:
            alive = identifier in self.active
        state = read(folder / 'meshy/job.json')
        if state.get('status') == 'running' and not alive:
            # The server restarted mid-task. Meshy kept working, so follow the same task: no new credits.
            if state.get('taskId'):
                try:
                    self._start(identifier, folder, state)
                except ValueError:
                    pass  # another build holds the slot; the next status read tries again
            else:
                state.update(
                    status='failed',
                    stage='interrupted',
                    message='The server restarted before Meshy accepted the task. Nothing was charged; start again.',
                )
                atomic(folder / 'meshy/job.json', state)
        return {
            **state,
            'status': state.get('status', 'idle'),
            'model': (folder / 'meshy/model.glb').exists(),
            'result': read(folder / 'meshy/result.json') or None,
        }

    def train(self, identifier, rebuild=False):
        folder = self.store.folder(identifier)
        if not api_key():
            raise ValueError(
                'Meshy is not configured. Add MESHY_API_KEY to .env or save a key under Meshy API settings.'
            )
        state = read(folder / 'meshy/job.json')
        resumable = (
            state.get('taskId')
            and not rebuild
            and (state.get('status') == 'running' or state.get('resumable'))
        )
        if (
            not rebuild
            and not resumable
            and state.get('status') == 'complete'
            and (folder / 'meshy/model.glb').exists()
        ):
            return {'id': identifier, 'status': 'complete', 'reused': True}
        if not resumable:
            frames = read(folder / 'capture.json').get('frames', [])
            select_views(frames)  # fail before a worker starts, so the caller sees why
        self._start(identifier, folder, state if resumable else None)
        return {'id': identifier, 'status': 'running', 'resumed': bool(resumable)}

    def _start(self, identifier, folder, resume=None):
        with self.lock:
            if identifier in self.active:
                raise ValueError('Meshy is already building this scan.')
            if self.active:
                raise ValueError(
                    'Another Meshy build is running. Wait for it to finish.'
                )
            (folder / 'meshy').mkdir(exist_ok=True)
            state = (
                {**resume, 'status': 'running', 'resumable': False}
                if resume
                else {
                    'status': 'running',
                    'stage': 'prepare',
                    'message': 'Choosing head views for Meshy…',
                    'requestedAt': time.time(),
                }
            )
            atomic(folder / 'meshy/job.json', state)
            worker = threading.Thread(
                target=self._run, args=(identifier, folder, state), daemon=True
            )
            self.active[identifier] = worker
        worker.start()

    def _run(self, identifier, folder, state):
        def update(**changes):
            state.update(changes)
            if not folder.is_dir():
                raise FileNotFoundError  # the scan was deleted; stop quietly
            (folder / 'meshy').mkdir(exist_ok=True)
            atomic(folder / 'meshy/job.json', state)

        try:
            if not state.get('taskId'):
                frames = read(folder / 'capture.json').get('frames', [])
                views = select_views(frames)
                images = []
                (folder / 'meshy/views').mkdir(exist_ok=True)
                for old in (folder / 'meshy/views').glob('view-*.png'):
                    old.unlink()
                for n, view in enumerate(views):
                    name = frames[view['index']]['filename']
                    raw = prepare_view(folder / 'images' / name)
                    (folder / 'meshy/views' / f'view-{n}.png').write_bytes(raw)
                    images.append(raw)
                    view['filename'] = name
                endpoint, payload = task_payload(images)
                update(
                    stage='upload',
                    views=views,
                    endpoint=endpoint,
                    aiModel=payload['ai_model'],
                    message=f'Sending {len(images)} cropped head view{"s" if len(images) > 1 else ""} to Meshy…',
                )
                try:
                    created = call('/' + endpoint, payload, timeout=180)
                except Unreachable as exc:
                    raise ValueError(str(exc) + ' Nothing was charged.') from None
                task_id = created.get('result') or created.get('id')
                if not isinstance(task_id, str) or not re.fullmatch(
                    r'[A-Za-z0-9_-]{8,80}', task_id
                ):
                    raise ValueError('Meshy did not return a task id.')
                update(
                    taskId=task_id,
                    stage='queued',
                    progress=0,
                    message='Meshy accepted the views. Waiting for a worker…',
                )
            task = self._follow(state, update)
            self._download(folder, task, state, update)
        except FileNotFoundError:
            pass
        except ValueError as exc:
            self._fail(update, str(exc))
        except Exception:
            traceback.print_exc()
            self._fail(
                update, 'The Meshy build failed unexpectedly. See the server log.'
            )
        finally:
            with self.lock:
                self.active.pop(identifier, None)

    @staticmethod
    def _fail(update, message):
        try:
            update(status='failed', message=message)
        except OSError:
            pass

    def _follow(self, state, update):
        deadline = time.time() + TASK_TIMEOUT
        path = f"/{state['endpoint']}/{state['taskId']}"
        while True:
            try:
                task = call(path)
            except Unreachable:
                task = None
            except ApiError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                task = None
            status = (task or {}).get('status')
            if status == 'SUCCEEDED':
                return task
            if status in ('FAILED', 'CANCELED', 'EXPIRED'):
                reason = (task.get('task_error') or {}).get('message') or status.lower()
                raise ValueError(f'Meshy could not build this head: {reason}'[:300])
            if time.time() > deadline:
                update(resumable=True)
                raise ValueError(
                    'Meshy has not finished after 20 minutes. The task is still running there: '
                    'press Create again to keep following it without spending new credits.'
                )
            if task:
                progress = task.get('progress') or 0
                ahead = task.get('preceding_tasks') or 0
                update(
                    stage='queued' if status == 'PENDING' else 'generating',
                    progress=progress,
                    message=(
                        f'Queued at Meshy · {ahead} task{"s" if ahead != 1 else ""} ahead…'
                        if status == 'PENDING' and ahead
                        else f'Meshy is building your head · {progress}%'
                    ),
                )
            time.sleep(POLL_SECONDS)

    def _download(self, folder, task, state, update):
        url = asset_url((task.get('model_urls') or {}).get('glb'))
        if not url:
            raise ValueError('Meshy finished without a GLB model.')
        update(stage='download', progress=100, message='Downloading the Meshy model…')
        target = folder / 'meshy/model.glb'
        partial = folder / 'meshy/model.glb.part'
        try:
            # A signed asset URL on another host: the API key is never sent to it.
            with (
                urlopen(
                    Request(url, headers={'User-Agent': 'punching-face/1.0'}),
                    timeout=180,
                ) as r,
                partial.open('wb') as out,
            ):
                total = 0
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 200_000_000:
                        raise ValueError('The Meshy model is larger than 200 MB.')
                    out.write(chunk)
            with partial.open('rb') as check:
                if check.read(4) != b'glTF':
                    raise ValueError('Meshy returned an unexpected model file.')
            partial.replace(target)
        except (HTTPError, URLError, TimeoutError, socket.timeout, ConnectionError):
            update(resumable=True)
            raise ValueError(
                'The Meshy model could not be downloaded. Press Create again to retry without new credits.'
            ) from None
        finally:
            partial.unlink(missing_ok=True)
        thumbnail = asset_url(task.get('thumbnail_url'))
        if thumbnail:
            try:
                with urlopen(
                    Request(thumbnail, headers={'User-Agent': 'punching-face/1.0'}),
                    timeout=30,
                ) as r:
                    raw = r.read(8_000_000)
                with Image.open(io.BytesIO(raw)) as im:
                    im.convert('RGBA').save(folder / 'meshy/thumbnail.png')
            except Exception:
                pass  # a preview only
        seconds = round(time.time() - state.get('requestedAt', time.time()), 1)
        result = {
            'taskId': state['taskId'],
            'endpoint': state['endpoint'],
            'aiModel': state.get('aiModel'),
            'views': state.get('views', []),
            'consumedCredits': task.get('consumed_credits'),
            'seconds': seconds,
            'bytes': target.stat().st_size,
            'finishedAt': time.time(),
        }
        atomic(folder / 'meshy/result.json', result)
        update(
            status='complete',
            stage='complete',
            seconds=seconds,
            consumedCredits=result['consumedCredits'],
            message=f'Meshy head ready after {round(seconds)} s'
            + (
                f" · {result['consumedCredits']} credits used."
                if isinstance(result['consumedCredits'], (int, float))
                else '.'
            ),
        )

    def handle(self, handler, url):
        if handler.headers.get('Host', '').split(':')[0] not in (
            '127.0.0.1',
            'localhost',
        ):
            return handler.reply(
                403, {'error': 'This service accepts local requests only.'}
            )
        try:
            result = self.route(handler, url)
            if result is not None:
                handler.reply(*result)
        except ValueError as exc:
            handler.reply(422, {'error': str(exc)})
        except (KeyError, TypeError, OSError):
            handler.reply(422, {'error': 'The Meshy request could not be completed.'})

    def route(self, handler, url):
        query = parse_qs(url.query)
        identifier = query.get('id', [''])[0]
        if handler.command == 'GET':
            if url.path == '/api/meshy-status':
                return 200, self.status(query.get('balance', [''])[0] == '1')
            if url.path == '/api/meshy-job':
                return 200, self.job(identifier)
            if url.path == '/api/meshy-asset':
                asset = query.get('asset', [''])[0]
                if asset not in ASSETS:
                    raise ValueError('Unknown Meshy asset.')
                base = self.store.folder(identifier) / 'meshy'
                file = base / ('views' if asset.startswith('view-') else '') / asset
                if not file.exists():
                    raise ValueError('This Meshy asset is not available yet.')
                body = file.read_bytes()
                handler.send_response(200)
                handler.send_header('Content-Type', ASSETS[asset])
                handler.send_header('Cache-Control', 'no-store')
                handler.send_header('Content-Length', str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)
                return None
            return 404, {'error': 'Unknown Meshy endpoint.'}
        if handler.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            raise ValueError('Expected JSON.')
        size = int(handler.headers.get('Content-Length', '0'))
        if not 0 < size <= 4096:
            raise ValueError('Request is empty or too large.')
        data = json.loads(handler.rfile.read(size))
        if url.path == '/api/meshy-config':
            configure(data.get('apiKey'))
            return 200, self.status()
        if url.path == '/api/meshy-train':
            return 202, self.train(data.get('id', ''), data.get('rebuild') is True)
        return 404, {'error': 'Unknown Meshy endpoint.'}
