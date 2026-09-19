"""Transactional head-model publication; source images and caches stay at root.

Readers pin ``release_manifest(folder)['generation']`` for every asset request
and physics session. A transaction's caller MUST join its writers before commit
or context exit. Only ARTIFACT_NAMES are sealed; staged cache links are ignored.
Read-side helpers use only the standard library; validation imports NumPy lazily.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from urllib.parse import urlencode
import uuid
import zlib

ARTIFACT_NAMES = (
    'mesh.json',
    'texture-atlas.json',
    'appearance.png',
    'appearance-roughness.png',
    'physics-cage.json',
    'physics-binding.json',
    'surface-validation.json',
    'pre-ear-surface.npz',
    'eyewear-mask-audit.json',
    'frame-evidence.json',
    'ear-measurements.json',
)
REQUIRED_NAMES = (
    'mesh.json',
    'texture-atlas.json',
    'appearance.png',
    'physics-cage.json',
    'physics-binding.json',
)
_POINTER = 'model-release.json'
_SEAL = 'release.json'
_GENERATIONS = '.model-generations'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)

    def finite(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)

    return json.loads(path.read_text(), parse_constant=invalid, parse_float=finite)


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _token(generation):
    _require(
        isinstance(generation, str) and re.fullmatch('[a-f0-9]{32}', generation),
        'Invalid head-model generation.',
    )
    return generation


def _manifest(data, generation=None):
    _require(
        isinstance(data, dict) and data.get('version') == 1,
        'Invalid head-model release manifest.',
    )
    _token(data.get('generation'))
    _require(
        generation is None or data['generation'] == generation,
        'Head-model generation does not match its seal.',
    )
    _require(
        data.get('photoModel') is True and isinstance(data.get('files'), dict),
        'Head-model release is not sealed.',
    )
    _require(
        all(name in data['files'] for name in REQUIRED_NAMES),
        'Head-model release is missing required assets.',
    )
    _require(
        all(
            name in ARTIFACT_NAMES
            and isinstance(digest, str)
            and re.fullmatch('[a-f0-9]{64}', digest)
            for name, digest in data['files'].items()
        ),
        'Invalid head-model file manifest.',
    )
    return data


def release_manifest(folder):
    """Current accepted release, independent of running/failed attempt status."""
    path = Path(folder) / _POINTER
    if not path.exists():
        return {'version': 1, 'generation': 'legacy'}
    return _manifest(_json(path))


def published_folder(folder, generation=None):
    """Resolve a pinned sealed generation, or explicit legacy root artifacts."""
    folder = Path(folder)
    current = release_manifest(folder) if generation is None else None
    generation = current['generation'] if current is not None else generation
    if generation == 'legacy':
        return folder
    path = folder / _GENERATIONS / _token(generation)
    _require(
        not path.is_symlink() and path.is_dir(), 'Head-model generation is unavailable.'
    )
    seal = _manifest(_json(path / _SEAL), generation)
    _require(
        current is None or current == seal,
        'Release pointer differs from its generation seal.',
    )
    return path


def has_published_model(folder):
    """Cheap presence check for the accepted core bundle; never reads status.json."""
    try:
        path = published_folder(folder)
        return all(
            (path / name).is_file() and (path / name).stat().st_size > 0
            for name in REQUIRED_NAMES
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


@contextmanager
def _publication_lock(folder):
    with (folder / '.model-publication.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _snapshot(folder):
    pointer = folder / _POINTER
    if pointer.exists():
        return ('release', pointer.read_bytes())
    return (
        'legacy',
        tuple(
            (name, _sha(folder / name) if (folder / name).exists() else None)
            for name in ARTIFACT_NAMES
        ),
    )


def _write_json(path, data):
    with path.open('w') as stream:
        json.dump(data, stream, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_pointer(folder, manifest):
    """Single atomic visibility point; separate for fault-injection tests."""
    temporary = folder / ('.model-release-' + uuid.uuid4().hex + '.tmp')
    try:
        _write_json(temporary, manifest)
        temporary.replace(folder / _POINTER)
    finally:
        # Cleanup cannot turn a successful pointer swap into a failed commit.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _png(path):
    """Check complete PNG framing/CRC and pixel payload without image libraries."""
    raw = path.read_bytes()
    _require(raw[:8] == b'\x89PNG\r\n\x1a\n', 'Invalid PNG: ' + path.name)
    offset, header, compressed, ended = 8, None, [], False
    while offset < len(raw):
        _require(offset + 12 <= len(raw), 'Truncated PNG: ' + path.name)
        size = struct.unpack('>I', raw[offset : offset + 4])[0]
        kind = raw[offset + 4 : offset + 8]
        end = offset + 12 + size
        _require(end <= len(raw), 'Truncated PNG: ' + path.name)
        payload = raw[offset + 8 : end - 4]
        crc = struct.unpack('>I', raw[end - 4 : end])[0]
        _require(
            zlib.crc32(kind + payload) & 0xFFFFFFFF == crc, 'PNG checksum mismatch.'
        )
        if header is None:
            _require(kind == b'IHDR' and size == 13, 'Missing PNG header.')
            header = struct.unpack('>IIBBBBB', payload)
        elif kind == b'IDAT':
            compressed.append(payload)
        elif kind == b'IEND':
            _require(size == 0 and end == len(raw), 'Invalid PNG ending.')
            ended = True
            break
        offset = end
    _require(ended and compressed, 'Incomplete PNG.')
    width, height, depth, color, compression, filtering, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    _require(
        width > 0
        and height > 0
        and channels
        and depth == 8
        and compression == filtering == interlace == 0,
        'Expected noninterlaced 8-bit head texture PNG.',
    )
    expected = height * (1 + width * channels)
    _require(expected <= 256 * 1024 * 1024, 'Head texture is too large.')
    decoder = zlib.decompressobj()
    pixels = decoder.decompress(b''.join(compressed), expected + 1)
    _require(
        len(pixels) == expected and decoder.eof and not decoder.unused_data,
        'Invalid PNG pixel payload.',
    )
    _require(
        all(pixels[i] <= 4 for i in range(0, expected, 1 + width * channels)),
        'Invalid PNG row filter.',
    )
    return width, height


def _validate(stage, capture):
    import numpy as np

    def array(data, key, columns=None):
        value = np.asarray(data[key], dtype=float)
        _require(
            value.ndim == 1 and value.size > 0 and np.isfinite(value).all(),
            'Invalid finite array: ' + key,
        )
        if columns:
            _require(value.size % columns == 0, 'Invalid shape: ' + key)
            value = value.reshape(-1, columns)
        return value

    def indices(data, key, limit, columns=None):
        value = array(data, key, columns)
        _require(
            np.equal(value, np.floor(value)).all()
            and (value >= 0).all()
            and (value < limit).all(),
            'Invalid index bounds: ' + key,
        )
        return value.astype(np.int64)

    for name in REQUIRED_NAMES:
        _require((stage / name).is_file(), 'Missing required head artifact: ' + name)
    for name in ARTIFACT_NAMES:
        _require(
            not (stage / name).is_symlink(),
            'Head artifacts must not be symlinks: ' + name,
        )
        if (stage / name).exists() and name.endswith('.json'):
            _json(stage / name)
    mesh, atlas, cage, binding = [
        _json(stage / name)
        for name in (
            'mesh.json',
            'texture-atlas.json',
            'physics-cage.json',
            'physics-binding.json',
        )
    ]
    positions = array(mesh, 'positions', 3)
    faces = indices(mesh, 'indices', len(positions), 3)
    _require(
        np.isfinite(positions.astype('<f4')).all(), 'Positions exceed float32 bounds.'
    )
    for key in ('normals', 'colors'):
        if key in mesh:
            _require(
                array(mesh, key, 3).shape == positions.shape,
                'Mesh attribute size mismatch: ' + key,
            )
    coarse = array(cage, 'positions', 3)
    _require(
        len(coarse) <= len(positions)
        and np.array_equal(
            coarse.astype('<f4'), positions[: len(coarse)].astype('<f4')
        ),
        'Physics cage does not match mesh positions.',
    )
    indices(cage, 'indices', len(coarse), 3)
    bind_indices = indices(binding, 'indices', len(coarse), 3)
    weights = array(binding, 'weights', 3)
    active = array(binding, 'active')
    _require(
        bind_indices.shape == weights.shape == (len(positions), 3)
        and len(active) == len(positions),
        'Physics binding size mismatch.',
    )
    _require(
        (weights >= 0).all()
        and (weights <= 1).all()
        and np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-5)
        and (active >= 0).all()
        and (active <= 1).all(),
        'Invalid physics weights.',
    )
    for key, anchor in cage.get('rigAnchors', {}).items():
        _require(
            str(key).isdigit() and int(key) < len(coarse), 'Invalid rig anchor index.'
        )
        value = np.asarray(anchor, dtype='<f4')
        _require(
            value.shape == (3,)
            and np.array_equal(value, coarse[int(key)].astype('<f4')),
            'Rig anchor differs from cage.',
        )
        if 'rigAnchors' in mesh:
            _require(
                np.array_equal(
                    value, np.asarray(mesh['rigAnchors'].get(key), dtype='<f4')
                ),
                'Mesh rig anchor differs from cage.',
            )
    mapping = indices(atlas, 'mapping', len(positions))
    atlas_faces = indices(atlas, 'indices', len(mapping), 3)
    uv = array(atlas, 'uv', 2)
    _require(
        len(uv) == len(mapping) and (uv >= -1e-6).all() and (uv <= 1 + 1e-6).all(),
        'Invalid atlas UV bounds.',
    )
    _require(
        np.array_equal(mapping[atlas_faces], faces),
        'Atlas topology does not match mesh.',
    )
    position_hash = hashlib.sha256(positions.astype('<f4').tobytes()).hexdigest()
    _require(
        atlas.get('positionsSha256') == position_hash, 'Atlas geometry hash mismatch.'
    )
    _require(
        atlas.get('textureSha256') == _sha(stage / 'appearance.png'),
        'Atlas texture hash mismatch.',
    )
    dimensions = _png(stage / 'appearance.png')
    roughness = stage / 'appearance-roughness.png'
    if atlas.get('roughnessTexture'):
        _require(roughness.is_file(), 'Atlas references missing roughness texture.')
    if roughness.exists():
        _require(_png(roughness) == dimensions, 'Roughness texture dimensions differ.')
    baseline = stage / 'pre-ear-surface.npz'
    if baseline.exists():
        with np.load(baseline, allow_pickle=False) as saved:
            rest = np.asarray(saved['positions'], dtype='<f4')
            _require(
                rest.shape == positions.shape
                and np.isfinite(rest).all()
                and np.array_equal(saved['indices'], faces),
                'Baseline topology/positions invalid.',
            )
            count = mesh.get('stats', {}).get('observedFaceTriangles', 0)
            _require(
                isinstance(count, int) and 0 <= count <= len(faces),
                'Invalid observed-face count.',
            )
            protected = np.unique(np.r_[np.arange(len(coarse)), faces[:count].ravel()])
            _require(
                np.array_equal(rest[protected], positions.astype('<f4')[protected]),
                'Baseline does not match protected face positions.',
            )
            if (capture / 'capture.json').exists():
                _require(
                    str(saved['captureHash']) == _sha(capture / 'capture.json'),
                    'Baseline capture hash mismatch.',
                )
    return position_hash


class HeadArtifactTransaction:
    """Stage and publish one immutable model; workers must join before exit.

    ``commit()`` returns the release manifest. Context exit removes only staging;
    committed generations persist so already-pinned clients can finish reading.
    A rejected/failed commit leaves the accepted pointer and legacy files intact.
    """

    def __init__(self, folder, seed=False):
        self.folder = Path(folder).resolve()
        self.seed = seed
        self.generation = uuid.uuid4().hex
        self.stage = None
        self._committed = False

    def __enter__(self):
        _require(self.stage is None, 'Transaction cannot be entered twice.')
        root = self.folder / _GENERATIONS
        root.mkdir(exist_ok=True)
        with _publication_lock(self.folder):
            self._before = _snapshot(self.folder)
            source = published_folder(self.folder)
            self.stage = Path(tempfile.mkdtemp(prefix='.staging-', dir=root))
            try:
                if self.seed:
                    for name in ARTIFACT_NAMES:
                        if (source / name).is_file():
                            shutil.copy2(source / name, self.stage / name)
            except BaseException:
                shutil.rmtree(self.stage)
                raise
        return self

    def commit(self):
        _require(
            self.stage is not None and self.stage.is_dir() and not self._committed,
            'Transaction is not active or is already committed.',
        )
        position_hash = _validate(self.stage, self.folder)
        atlas = _json(self.stage / 'texture-atlas.json')
        for field, name in (
            ('texture', 'appearance.png'),
            ('roughnessTexture', 'appearance-roughness.png'),
        ):
            if field == 'texture' or field in atlas:
                atlas[field] = '/api/face-asset?' + urlencode(
                    {
                        'id': self.folder.name,
                        'asset': name,
                        'generation': self.generation,
                    }
                )
        _write_json(self.stage / 'texture-atlas.json', atlas)
        manifest = {
            'version': 1,
            'generation': self.generation,
            'photoModel': True,
            'positionsSha256': position_hash,
            'files': {
                name: _sha(self.stage / name)
                for name in ARTIFACT_NAMES
                if (self.stage / name).is_file()
            },
        }
        destination = self.folder / _GENERATIONS / self.generation
        with _publication_lock(self.folder):
            _require(
                _snapshot(self.folder) == self._before,
                'Accepted head model changed during reconstruction; publication rejected.',
            )
            destination.mkdir()
            try:
                for name in manifest['files']:
                    shutil.copy2(self.stage / name, destination / name)
                    _require(
                        _sha(destination / name) == manifest['files'][name],
                        'Artifact changed while sealing: ' + name,
                    )
                    with (destination / name).open('rb') as stream:
                        os.fsync(stream.fileno())
                _write_json(destination / _SEAL, manifest)
                _publish_pointer(self.folder, manifest)
            except BaseException:
                # Never remove a generation if interruption arrived immediately
                # after its atomic pointer replacement succeeded.
                if release_manifest(self.folder).get('generation') == self.generation:
                    self._committed = True
                else:
                    shutil.rmtree(destination)
                raise
            self._committed = True
        return manifest

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stage is not None:
            shutil.rmtree(self.stage, ignore_errors=True)
        return False
