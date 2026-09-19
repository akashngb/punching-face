"""Exact accelerators for the photo pipeline, and concurrent prefetch of its independent AI calls.

Nothing here changes a result. The rasterizers, texel sampler and native-frame
decoder return what the functions they replace return, bit for bit
(tests/pipeline_accel_test.py). The AI calls are the pipeline's own functions
started early on threads; each already caches on disk under a hash of its
inputs, so the serial code that follows finds its answer waiting.

Each replacement is pinned to the SHA-256 of the parsed reference function it
was derived from. Formatting and comments do not invalidate a compatible leaf. When that reference is edited the replacement steps aside and
the reference runs, so an edit is never silently shadowed. After porting the
edit here: `python scripts/pipeline_accel.py --pin`. Status: `--status`.
Measurements and the remaining plan are in PIPELINE_SPEEDUP.md.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import ast
import textwrap
import inspect
import json
import os
import re
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

PINS = {
    'zbuffer': '5c87c8242c7a354a2e19c681b490fa148999a20db35d658a346ede54db883794',
    'raster_atlas': 'c7f924eb102509d823aabe622e7ff4a6163b6f795e24f8501e2cd5e752cc5eb0',
    'prepare_detail_frames': 'a37295636faa41fbb2280e6cb66dafd45fb70cd0491884664fdbc13529c67de0',
}
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))
_pool = ThreadPoolExecutor(WORKERS, thread_name_prefix='accel')


def rows(X, M):
    """X@M for a tall (N,3) X, bit-identical, ~30x faster on macOS.

    Accelerate's gemm takes ~0.9 s for a 6M-row X and serializes concurrent
    callers; the transposed product takes ~0.03 s and yields the same bits.
    """
    return (M.T @ X.T).T


def _candidates(lo, hi):
    """Every (triangle, pixel) pair inside each triangle's bounding box, in triangle order."""
    wx = hi[:, 0] - lo[:, 0] + 1
    wy = hi[:, 1] - lo[:, 1] + 1
    counts = wx * wy
    tri = np.repeat(np.arange(len(lo)), counts)
    local = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    w = wx[tri]
    return tri, lo[tri, 0] + local % w, lo[tri, 1] + local // w


def _barycentric(t, tri, px, py):
    # The same expressions, in the same order, as the per-triangle loops.
    xx = px + 0.5
    yy = py + 0.5
    t0x, t0y, t1x, t1y, t2x, t2y = (t[:, i, j][tri] for i in range(3) for j in range(2))
    d = (t1y - t2y) * (t0x - t2x) + (t2x - t1x) * (t0y - t2y)
    a = ((t1y - t2y) * (xx - t2x) + (t2x - t1x) * (yy - t2y)) / d
    b = ((t2y - t0y) * (xx - t2x) + (t0x - t2x) * (yy - t2y)) / d
    return a, b, 1 - a - b


def rasterize_depth(projected, depth, faces, width, height, chunk=1 << 15):
    result = np.full((height, width), np.inf)
    t = projected[faces]
    z = depth[faces]
    with np.errstate(invalid='ignore'):
        keep = ~(np.min(z, axis=1) <= 0)
        lo = np.maximum(np.floor(t.min(axis=1)).astype(int), 0)
        hi = np.minimum(np.ceil(t.max(axis=1)).astype(int), [width - 1, height - 1])
        keep &= ~np.any(lo > hi, axis=1)
        d = (t[:, 1, 1] - t[:, 2, 1]) * (t[:, 0, 0] - t[:, 2, 0]) + (
            t[:, 2, 0] - t[:, 1, 0]
        ) * (t[:, 0, 1] - t[:, 2, 1])
        keep &= ~(np.abs(d) < 1e-9)
    ids = np.where(keep)[0]
    flat = result.reshape(-1)
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        zp = z[part]
        tri, px, py = _candidates(lo[part], hi[part])
        with np.errstate(invalid='ignore', divide='ignore'):
            a, b, c = _barycentric(t[part], tri, px, py)
            inside = (a >= 0) & (b >= 0) & (c >= 0)
            tri = tri[inside]
            a = a[inside]
            b = b[inside]
            c = c[inside]
            value = 1 / np.maximum(
                a / zp[tri, 0] + b / zp[tri, 1] + c / zp[tri, 2], 1e-12
            )
        # A minimum does not depend on the order triangles are visited in.
        np.minimum.at(flat, py[inside] * width + px[inside], value)
    return result


_depth_cache = {}
_depth_lock = threading.Lock()


def zbuffer(projected, depth, faces, width, height):
    """photo_geometry.zbuffer. The groom and the bake rasterize one mesh from the same cameras, so results are shared."""
    digest = hashlib.blake2b(digest_size=16)
    for array in (projected, depth, faces):
        digest.update(np.ascontiguousarray(array).tobytes())
    key = (digest.hexdigest(), width, height)
    with _depth_lock:
        found = _depth_cache.get(key)
    if found is None:
        found = rasterize_depth(projected, depth, faces, width, height)
        with _depth_lock:
            if len(_depth_cache) >= 96:
                _depth_cache.pop(next(iter(_depth_cache)))
            _depth_cache[key] = found
    return found.copy()


def raster_atlas(
    p,
    n,
    mapping,
    indices,
    uv,
    is_face,
    size,
    part_labels=None,
    chunk=1 << 13,
    return_binding=False,
):
    """photo_geometry.raster_atlas."""
    from scripts.photo_geometry import interpolate_part_labels

    world = np.zeros((size, size, 3), np.float32)
    normal = np.zeros_like(world)
    covered = np.zeros((size, size), bool)
    observed = np.zeros_like(covered)
    parts = np.zeros((size, size), np.uint8) if part_labels is not None else None
    owner = np.zeros((size, size), np.int32) if return_binding else None
    bary = np.zeros_like(world) if return_binding else None
    t = uv[indices] * size
    lo = np.maximum(np.floor(t.min(axis=1)).astype(int), 0)
    hi = np.minimum(np.ceil(t.max(axis=1)).astype(int), size - 1)
    d = (t[:, 1, 1] - t[:, 2, 1]) * (t[:, 0, 0] - t[:, 2, 0]) + (
        t[:, 2, 0] - t[:, 1, 0]
    ) * (t[:, 0, 1] - t[:, 2, 1])
    ids = np.where(~np.any(lo > hi, axis=1) & ~(np.abs(d) < 1e-9))[0]
    fw = world.reshape(-1, 3)
    fn = normal.reshape(-1, 3)
    fc = covered.reshape(-1)
    fo = observed.reshape(-1)
    fp = parts.reshape(-1) if parts is not None else None
    last = np.full(size * size, -1, np.int64)
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        tri, px, py = _candidates(lo[part], hi[part])
        a, b, c = _barycentric(t[part], tri, px, py)
        inside = (a >= -1e-5) & (b >= -1e-5) & (c >= -1e-5)
        tri = tri[inside]
        pixel = py[inside] * size + px[inside]
        weights = np.stack([a[inside], b[inside], c[inside]], axis=-1)
        # A later triangle overwrites an earlier one on a shared edge, as in the sequential loop.
        order = np.arange(len(pixel))
        last[pixel] = -1
        np.maximum.at(last, pixel, order)
        win = last[pixel] == order
        tri = tri[win]
        pixel = pixel[win]
        weights = weights[win]
        vertex = mapping[indices[part]][tri]
        fw[pixel] = np.einsum('ij,ijk->ik', weights, p[vertex])
        fn[pixel] = np.einsum('ij,ijk->ik', weights, n[vertex])
        fc[pixel] = True
        fo[pixel] = is_face[part][tri]
        if fp is not None:
            fp[pixel] = interpolate_part_labels(weights, part_labels[vertex])
        if return_binding:
            owner.reshape(-1)[pixel] = part[tri]
            bary.reshape(-1, 3)[pixel] = weights
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-9)
    result = (world[covered], normal[covered], observed[covered], covered)
    if parts is not None:
        result += (parts[covered],)
    if return_binding:
        result += (
            {
                'triangles': mapping[indices],
                'triangleIds': owner[covered],
                'weights': bary[covered],
            },
        )
    return result


def map_coordinates(input, coordinates, *args, **kwargs):
    """scipy.ndimage.map_coordinates, with a long list of sample points split across threads.

    Every output depends on its own coordinate only, and SciPy releases the
    GIL, so the pieces are exact. Spline orders that prefilter stay serial.
    """
    from scipy.ndimage import map_coordinates as reference

    order = kwargs.get('order', args[1] if len(args) > 1 else 3)
    output = kwargs.get('output', args[0] if args else None)
    prefilters = order > 1 and kwargs.get(
        'prefilter', args[4] if len(args) > 4 else True
    )
    coordinates = np.asarray(coordinates)
    nested = threading.current_thread().name.startswith('accel')
    if (
        output is not None
        or prefilters
        or nested
        or coordinates.ndim != 2
        or coordinates.shape[1] < 400_000
    ):
        return reference(input, coordinates, *args, **kwargs)
    bounds = np.linspace(0, coordinates.shape[1], WORKERS + 1).astype(int)
    pieces = _pool.map(
        lambda ab: reference(input, coordinates[:, ab[0] : ab[1]], *args, **kwargs),
        zip(bounds[:-1], bounds[1:]),
    )
    return np.concatenate(list(pieces))


_detail_locks = {}
_detail_guard = threading.Lock()


def locked_per_folder(function):
    """Concurrent callers wait for the first instead of decoding, and writing, the same frames twice."""

    def prepare_detail_frames(folder):
        with _detail_guard:
            lock = _detail_locks.setdefault(
                str(Path(folder).resolve()), threading.Lock()
            )
        with lock:
            return function(folder)

    prepare_detail_frames.locked = True
    return prepare_detail_frames


def stream_detail_frames(folder, workers=8):
    """photo_detail.prepare_detail_frames: one sequential decode instead of five seeks per view.

    Seeking an HEVC recording 255 times took 58 s on the benchmark capture;
    streaming it once takes 6 s and writes byte-identical PNGs and audit.
    """
    import cv2
    from PIL import Image
    from face_pipeline import atomic

    folder = Path(folder)
    manifest = folder / 'photo-detail.json'
    if manifest.exists():
        cached = json.loads(manifest.read_text())
        if cached.get('version') == 2:
            return cached
    video = folder / 'source-video'
    if not video.exists():
        return {'frames': [], 'source': 'Captured images'}
    frames = json.loads((folder / 'capture.json').read_text())['frames']
    decoder = cv2.VideoCapture(str(video))
    fps = decoder.get(cv2.CAP_PROP_FPS)
    rotation = int(decoder.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    decoder.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    output = folder / 'detail-images'
    output.mkdir(exist_ok=True)
    wanted = []
    for frame in frames:
        if frame.get('timeSeconds') is None or fps <= 0:
            continue
        target = int(round(frame['timeSeconds'] * fps))
        wanted.append((frame, range(max(0, target - 2), target + 3)))

    def build(frame, candidates):
        captured = np.asarray(
            Image.open(folder / 'images' / frame['filename']).convert('RGBA')
        )
        h, w = captured.shape[:2]
        valid = captured[:, :, 3] > 220
        best = None
        for index, bgr in candidates:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rotation:
                rgb = np.rot90(rgb, (360 - rotation) // 90).copy()
            if abs(rgb.shape[1] / rgb.shape[0] - w / h) > 0.005:
                continue
            small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            difference = small.astype(float) - captured[:, :, :3]
            difference -= np.median(difference[valid], axis=0)
            error = float(np.mean(np.abs(difference[valid])))
            if best is None or error < best[0]:
                best = (error, index, rgb)
        if best is None or best[0] > 12:
            return None
        error, index, rgb = best
        registered = cv2.resize(
            captured[:, :, :3],
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        native = rgb.astype(np.float32)
        sigma = 1.4 * rgb.shape[1] / w
        rgb = np.uint8(
            np.clip(
                cv2.GaussianBlur(registered, (0, 0), sigma)
                + native
                - cv2.GaussianBlur(native, (0, 0), sigma),
                0,
                255,
            )
        )
        alpha = cv2.resize(
            captured[:, :, 3],
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        rgba = np.dstack([rgb, alpha])
        rgba[alpha == 0, :3] = 0
        Image.fromarray(rgba).save(output / frame['filename'])
        return {
            'filename': frame['filename'],
            'requestedSeconds': frame['timeSeconds'],
            'decodedFrame': index,
            'matchError255': round(error, 3),
            'size': [rgb.shape[1], rgb.shape[0]],
            'registeredSize': [w, h],
        }

    # A view is dispatched once its last candidate has streamed past, so
    # only a few native frames are ever held in memory.
    pending = sorted(range(len(wanted)), key=lambda i: wanted[i][1].stop)
    needed = {index for _, indices in wanted for index in indices}
    held = {}
    jobs = {}
    position = 0
    cursor = 0
    last = max(needed, default=-1)

    def dispatch(pool, i):
        frame, indices = wanted[i]
        jobs[i] = pool.submit(
            build, frame, [(k, held[k]) for k in indices if k in held]
        )

    try:
        with ThreadPoolExecutor(workers) as pool:
            while position <= last and decoder.grab():
                if position in needed:
                    ok, bgr = decoder.retrieve()
                    if ok:
                        held[position] = bgr
                position += 1
                while (
                    cursor < len(pending)
                    and wanted[pending[cursor]][1].stop <= position
                ):
                    dispatch(pool, pending[cursor])
                    cursor += 1
                    alive = {k for j in pending[cursor:] for k in wanted[j][1]}
                    for k in [k for k in held if k not in alive]:
                        del held[k]
            for i in pending[cursor:]:
                dispatch(pool, i)
            audit = [jobs[i].result() for i in range(len(wanted))]
    finally:
        decoder.release()
    result = {
        'version': 2,
        'source': (
            'Original video detail at matched timestamps; colour matched to '
            'registered browser-decoded frames; no super-resolution'
        ),
        'frames': [a for a in audit if a],
    }
    atomic(manifest, result)
    return result


def _source_hash(function):
    source = textwrap.dedent(inspect.getsource(function))
    normalized = ast.dump(ast.parse(source), include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _references():
    import scripts.photo_detail as detail
    import scripts.photo_geometry as geometry

    return {
        'zbuffer': (geometry, getattr(geometry, 'reference_zbuffer', geometry.zbuffer)),
        'raster_atlas': (
            geometry,
            getattr(geometry, 'reference_raster_atlas', geometry.raster_atlas),
        ),
        'prepare_detail_frames': (
            detail,
            getattr(
                detail, 'reference_prepare_detail_frames', detail.prepare_detail_frames
            ),
        ),
    }


def install_leaves(log=print):
    """Swap in each replacement whose reference is unchanged. Returns the names now active."""
    import scripts.photo_detail as detail

    active = []
    # photo_detail first: other modules bind prepare_detail_frames by name when imported.
    reference = getattr(
        detail, 'reference_prepare_detail_frames', detail.prepare_detail_frames
    )
    detail.reference_prepare_detail_frames = reference
    if _source_hash(reference) == PINS['prepare_detail_frames']:
        detail.prepare_detail_frames = locked_per_folder(stream_detail_frames)
        active.append('prepare_detail_frames')
    else:
        detail.prepare_detail_frames = locked_per_folder(reference)
    import scripts.photo_geometry as geometry

    for name, replacement in [('zbuffer', zbuffer), ('raster_atlas', raster_atlas)]:
        reference = getattr(geometry, 'reference_' + name, getattr(geometry, name))
        if _source_hash(reference) == PINS[name]:
            setattr(geometry, 'reference_' + name, reference)
            setattr(geometry, name, replacement)
            active.append(name)
    for module in (
        'scripts.photo_geometry',
        'scripts.head_material',
        'scripts.eye_detail',
    ):
        __import__(module)
        if hasattr(sys.modules[module], 'map_coordinates'):
            sys.modules[module].map_coordinates = map_coordinates
    active.append('map_coordinates')
    for module in ('scripts.photo_geometry', 'scripts.head_semantics'):
        if module in sys.modules and hasattr(
            sys.modules[module], 'prepare_detail_frames'
        ):
            sys.modules[module].prepare_detail_frames = detail.prepare_detail_frames
    stale = [name for name in PINS if name not in active]
    if log:
        log(
            'pipeline_accel: active '
            + ', '.join(active)
            + (
                '; STALE (reference edited, running it unaccelerated) '
                + ', '.join(stale)
                if stale
                else ''
            ),
            flush=True,
        )
    return active


class _Task:
    """A daemon thread with a result. A failed build must never wait minutes for a speculative request."""

    def __init__(self, function):
        self.done = threading.Event()
        self.error = None

        def work():
            try:
                function()
            except Exception as error:
                self.error = error
            finally:
                self.done.set()

        threading.Thread(target=work, daemon=True, name='prefetch').start()


def install_prefetch(build, log=print):
    """Run the pipeline's independent AI calls together instead of one after another.

    hair_recognition, head_semantics and eye_detail need only the list of views
    that astra_head_completion chose, never its answer. Each is started as soon
    as that list exists and writes its usual cache file; the serial call that
    follows in build.run() reads it. Any prefetch failure is ignored, leaving
    that serial call to run, and raise, exactly as before.
    """
    if os.environ.get('CONTACT_PREFETCH') == '0':
        return False
    import scripts.astra_head_completion as completion_module
    from PIL import Image

    tasks = {}
    state = {}
    guard = threading.Lock()

    def start(name, function):
        with guard:
            if name not in tasks:
                tasks[name] = _Task(function)

    def wait(name):
        task = tasks.get(name)
        if task:
            task.done.wait()
            if task.error and log:
                log(
                    f'pipeline_accel: {name} prefetch failed ({task.error}); running it inline.',
                    flush=True,
                )

    def planned(completion):
        # The cache keys hash the views in order. A model may return them in
        # another order; asking again in the planned order finds the prefetch.
        names = state.get('names')
        views = {v['filename']: v for v in completion.get('views', [])}
        if names and set(names) == set(views):
            return {**completion, 'views': [views[name] for name in names]}
        return completion

    def start_view_calls(folder, names, crops):
        if (
            json.loads((folder / 'capture.json').read_text()).get('captureRegion')
            != 'head'
        ):
            return
        state.setdefault('names', list(names))
        plan = {
            'views': [{'filename': name} for name in state['names']],
            'crops': crops,
        }
        start('hair', lambda: real['recognize_hair'](folder, plan))
        if real.get('analyze'):
            start('semantics', lambda: real['analyze'](folder, plan))

    real = {
        name: getattr(build, name)
        for name in ('recover', 'complete', 'recognize_hair', 'scan_eyes')
    }
    try:
        import scripts.head_semantics as semantics_module

        real['analyze'] = semantics_module.analyze
    except ImportError:
        semantics_module = None
    real_request = completion_module.request

    def request(content, *args, **kwargs):
        # complete() has just chosen its views; nothing downstream needs more than their names and crops.
        try:
            folder = state.get('folder')
            names = [
                m.group(1)
                for item in content
                if item.get('type') == 'input_text'
                for m in [re.match(r'Filename: (\S+?);', item['text'])]
                if m
            ]
            if folder and names:
                crops = {
                    name: list(
                        Image.open(folder / 'images' / name)
                        .convert('RGBA')
                        .getchannel('A')
                        .getbbox()
                    )
                    for name in names
                }
                start_view_calls(folder, names, crops)
        except Exception as error:
            if log:
                log(
                    f'pipeline_accel: could not start view prefetch ({error}).',
                    flush=True,
                )
        return real_request(content, *args, **kwargs)

    def recover(folder, *args, **kwargs):
        # Native detail frames need only the recording, so they decode while cameras are recovered.
        import scripts.photo_detail as detail

        if getattr(detail.prepare_detail_frames, 'locked', False):
            start('detail', lambda: detail.prepare_detail_frames(Path(folder)))
        return real['recover'](folder, *args, **kwargs)

    def complete(folder, *args, **kwargs):
        state['folder'] = Path(folder)
        frames = json.loads((Path(folder) / 'capture.json').read_text())['frames']
        start('eyes', lambda: real['scan_eyes'](folder, frames, True))
        result = real['complete'](folder, *args, **kwargs)
        # A cached completion never reaches request(); start from its stored plan instead.
        try:
            start_view_calls(
                Path(folder), [v['filename'] for v in result['views']], result['crops']
            )
        except Exception:
            pass
        return result

    def recognize_hair(folder, completion, *args, **kwargs):
        wait('hair')
        return real['recognize_hair'](folder, planned(completion), *args, **kwargs)

    def scan_eyes(folder, *args, **kwargs):
        wait('eyes')
        return real['scan_eyes'](folder, *args, **kwargs)

    def analyze(folder, completion, *args, **kwargs):
        wait('semantics')
        return real['analyze'](folder, planned(completion), *args, **kwargs)

    completion_module.request = request
    build.recover = recover
    build.complete = complete
    build.recognize_hair = recognize_hair
    build.scan_eyes = scan_eyes
    if semantics_module:
        semantics_module.analyze = analyze
    return True


if __name__ == '__main__':
    current = {
        name: _source_hash(function) for name, (_, function) in _references().items()
    }
    if '--pin' in sys.argv:
        path = Path(__file__)
        text = path.read_text()
        for name, digest in current.items():
            text = re.sub(rf"('{name}': )'[^']*'", rf"\1'{digest}'", text, count=1)
        path.write_text(text)
        print('pinned', ', '.join(current))
    else:
        for name, digest in current.items():
            print(
                f"{name:<24}{'active' if PINS[name] == digest else 'STALE: reference edited; port the change, then --pin'}"
            )
