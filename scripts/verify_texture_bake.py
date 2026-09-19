"""Opt-in real-capture check of deterministic serial/threaded texture baking.

Usage: PYTHONPATH=. .venv/bin/python scripts/verify_texture_bake.py CAPTURE
Runs two full cached-input bakes in disposable .local folders. Nothing is
published, no generation/reconstruction is invoked, and network access fails.
"""

import argparse
import ast
from contextlib import contextmanager, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _json(path):
    return json.loads(path.read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def _offline_source_guard(source):
    """Audit Python writes through source symlinks, and reject network/process IO."""
    state = {'active': True}
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def protected(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            return resolved == source or source in resolved.parents
        return False

    def audit(event, args):
        if not state['active']:
            return
        if event in (
            'socket.connect',
            'socket.getaddrinfo',
            'socket.bind',
            'subprocess.Popen',
            'os.system',
        ):
            raise RuntimeError(
                'Texture verification is cached-only and offline: ' + event
            )
        paths = ()
        if event == 'open' and (args[2] or 0) & write_flags:
            paths = args[:1]
        elif event in (
            'os.remove',
            'os.rmdir',
            'os.mkdir',
            'os.chmod',
            'os.utime',
            'os.truncate',
        ):
            paths = args[:1]
        elif event in ('os.rename', 'os.link'):
            paths = args[:2]
        elif event == 'os.symlink':
            paths = args[1:2]  # The destination, not the read-only source target.
        if any(protected(path) for path in paths):
            raise RuntimeError('Verifier attempted to modify source capture: ' + event)

    sys.addaudithook(audit)
    try:
        yield
    finally:
        # Python cannot remove audit hooks; the inert closure remains harmless.
        state['active'] = False


def _stage(source, destination):
    from head_artifacts import ARTIFACT_NAMES, published_folder

    excluded = (
        'before-',
        'head-detail-stage-',
        'verify-texture-',
        'material-stage-',
        '.pending-',
    )
    output_metadata = {'texture-atlas.json', 'eyewear-mask-audit.json'}
    source_directories = {
        'images',
        'masks',
        'detail-images',
        'photo-cameras',
        'eye-detail',
        'glasses-reference',
        'rear-prediction',
    }
    for item in sorted(source.iterdir()):
        if item.name.startswith(excluded):
            continue
        target = destination / item.name
        if item.is_dir() and item.name in source_directories:
            target.symlink_to(item.resolve(), target_is_directory=True)
        elif item.name == 'source-video':
            target.symlink_to(item.resolve())
        elif (
            item.is_file()
            and item.suffix == '.json'
            and item.name not in output_metadata
        ):
            shutil.copy2(item, target)
    accepted = published_folder(source)
    for name in ARTIFACT_NAMES:
        if (
            name.endswith('.json')
            and name not in output_metadata
            and (accepted / name).is_file()
        ):
            shutil.copy2(accepted / name, destination / name)
    # In particular do not copy previous output PNGs: absent new roughness must
    # not appear to pass merely because both stages inherited an old texture.


def _cached_detail(folder):
    path = folder / 'photo-detail.json'
    if path.exists():
        data = _json(path)
        _require(
            data.get('version') == 2,
            'Cached-only verification requires photo-detail version 2.',
        )
        for frame in data.get('frames', []):
            _require(
                (folder / 'detail-images' / frame['filename']).is_file(),
                'Missing cached native-detail image: ' + frame['filename'],
            )
        return data
    _require(
        not (folder / 'source-video').exists(),
        'Source video has no detail cache; prepare it before running cached-only verification.',
    )
    return {'frames': [], 'source': 'Captured images'}


def _run(stage, workers):
    import numpy as np
    import pycolmap
    import scripts.photo_geometry as geometry
    from scripts.hair_recognition import hair_completion

    mesh_snapshot = (stage / 'mesh.json').read_bytes()
    data = json.loads(mesh_snapshot)
    positions = np.asarray(data['positions']).reshape(-1, 3)
    faces = np.asarray(data['indices']).reshape(-1, 3)
    points = np.asarray(_json(stage / 'surface-validation.json')['landmarksWorld'])
    # Keep these transforms identical to the material refinement path; no
    # fresh triangulation or change to the capture's saved physical scale.
    center = (points[10] + points[152]) / 2
    right = points[263] - points[33]
    right /= np.linalg.norm(right)
    up = points[10] - points[152]
    up -= right * np.dot(up, right)
    up /= np.linalg.norm(up)
    basis = np.stack([right, up, np.cross(right, up)])
    reconstruction = pycolmap.Reconstruction(str(stage / 'photo-cameras'))
    frames = {
        frame['filename']: frame for frame in _json(stage / 'capture.json')['frames']
    }
    for image in reconstruction.images.values():
        local = (image.projection_center() - center) @ basis.T
        frames[image.name]['cameraYaw'] = float(
            np.degrees(np.arctan2(local[0], local[2]))
        )
    advice = (
        _json(stage / 'astra-head-completion.json')
        if (stage / 'astra-head-completion.json').exists()
        else None
    )
    if advice and (stage / 'hair-recognition.json').exists():
        advice = hair_completion(advice, _json(stage / 'hair-recognition.json'))
    eyes = (
        _json(stage / 'eye-detail.json')
        if (stage / 'eye-detail.json').exists()
        else None
    )
    semantics = (
        _json(stage / 'head-semantics.json')
        if (stage / 'head-semantics.json').exists()
        else None
    )
    regions = data['stats'].get('templateFit', {}).get('earRegions')
    detail = _cached_detail(stage)
    started = time.perf_counter()
    log_path = stage / 'bake.log'
    try:
        with (
            patch.dict(os.environ, {'CONTACT_BAKE_WORKERS': str(workers)}),
            patch.object(geometry, 'prepare_detail_frames', return_value=detail),
            log_path.open('w') as log,
            redirect_stdout(log),
        ):
            stats = geometry.bake_photographs(
                stage,
                positions,
                faces,
                data['stats']['observedFaceTriangles'],
                reconstruction,
                frames,
                list(reconstruction.images.values()),
                center,
                basis,
                data['transform'],
                advice,
                eyes,
                semantics,
                regions,
            )
    except Exception:
        if log_path.exists():
            print(
                'Bake log tail:\n' + '\n'.join(log_path.read_text().splitlines()[-12:]),
                file=sys.stderr,
            )
        raise
    elapsed = time.perf_counter() - started
    atlas = _json(stage / 'texture-atlas.json')
    expected_geometry = hashlib.sha256(
        np.asarray(positions, dtype='<f4').tobytes()
    ).hexdigest()
    _require(
        atlas.get('positionsSha256') == expected_geometry,
        f'{workers}-worker atlas geometry hash does not match the saved mesh.',
    )
    _require(atlas['stats'] == stats, f'{workers}-worker returned/saved stats differ.')
    _require(
        (stage / 'mesh.json').read_bytes() == mesh_snapshot,
        'Unexpected staged mesh change.',
    )
    return atlas, elapsed, expected_geometry


def _normalized(value, stage):
    if isinstance(value, dict):
        return {key: _normalized(item, stage) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item, stage) for item in value]
    if isinstance(value, str):
        return value.replace(
            '/api/face-asset?id=' + stage.name + '&', '/api/face-asset?id=<capture>&'
        )
    return value


def local_code_fingerprint(root, entrypoints):
    """Include local imported helpers, even imports inside bake functions.

    A hand-maintained module list can miss a new helper while its caller stays
    unchanged during verification. Walk syntax rather than importing modules:
    fingerprinting must not initialize models or run optional backends.
    """
    root = Path(root).resolve()
    pending = [Path(path).resolve() for path in entrypoints]
    result = {}

    def enqueue(parts):
        path = root.joinpath(*parts)
        candidates = [path.with_suffix('.py')]
        candidates.extend(
            root.joinpath(*parts[:length], '__init__.py')
            for length in range(1, len(parts) + 1)
        )
        for candidate in candidates:
            if candidate.is_file() and root in candidate.resolve().parents:
                pending.append(candidate.resolve())

    while pending:
        path = pending.pop()
        key = str(path.relative_to(root))
        if key in result:
            continue
        tree = ast.parse(path.read_text())
        result[key] = hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode()
        ).hexdigest()
        package = list(path.relative_to(root).parent.parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue(alias.name.split('.'))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        continue
                    base = package[: len(package) - node.level + 1]
                else:
                    base = []
                module = base + (node.module.split('.') if node.module else [])
                enqueue(module)
                for alias in node.names:
                    if alias.name != '*':
                        enqueue(module + alias.name.split('.'))
    return result


def verify(source, report=None):
    source = Path(source).resolve()
    _require(source.is_dir(), 'Capture directory does not exist.')
    from head_artifacts import published_folder, release_manifest

    release_snapshot = release_manifest(source)
    accepted = published_folder(source)
    snapshot = (accepted / 'mesh.json').read_bytes()
    local = ROOT / '.local'
    local.mkdir(exist_ok=True)
    from scripts.pipeline_accel import install_leaves

    install_leaves()
    tracked = [
        ROOT / 'scripts' / (name + '.py')
        for name in (
            'photo_geometry',
            'pipeline_accel',
            'head_material',
            'head_accessories',
            'eye_detail',
            'eye_shading',
            'photo_detail',
            'cleanup_lighting',
            'skin_continuation',
            'surface_completion',
            'ear_fit',
            'ear_appearance',
            'glasses_reference',
            'frame_evidence',
            'view_blending',
            'texture_evidence',
            'hair_recognition',
            'hair_appearance',
            'rear_hair',
            'camera_ownership',
            'camera_texture',
        )
    ] + [ROOT / 'head_artifacts.py']

    def code_fingerprint():
        return local_code_fingerprint(ROOT, tracked)

    code_before = code_fingerprint()
    try:
        with (
            _offline_source_guard(source),
            tempfile.TemporaryDirectory(
                prefix='verify-texture-serial-', dir=local
            ) as first,
            tempfile.TemporaryDirectory(
                prefix='verify-texture-threaded-', dir=local
            ) as second,
        ):
            stages = [Path(first), Path(second)]
            # Snapshot mutable metadata for both runs before starting either.
            for stage in stages:
                _stage(source, stage)
                _require(
                    (stage / 'mesh.json').read_bytes() == snapshot,
                    'Source mesh changed while staging the verifier.',
                )
            results = []
            for stage, workers in zip(stages, (1, 4)):
                print(
                    f'Verifying cached texture bake with {workers} worker(s)...',
                    flush=True,
                )
                results.append(_run(stage, workers))
                print(f'{workers} worker(s): {results[-1][1]:.2f}s', flush=True)
            a, b = results[0][0], results[1][0]
            for key in ('mapping', 'indices', 'uv', 'positionsSha256'):
                _require(a[key] == b[key], 'Serial/threaded atlas mismatch: ' + key)
            normalized = [
                _normalized(atlas, stage) for atlas, stage in zip((a, b), stages)
            ]
            _require(
                normalized[0]['stats'] == normalized[1]['stats'],
                'Serial/threaded appearance statistics differ.',
            )
            hashes = {}
            for name in ('appearance.png', 'appearance-roughness.png'):
                paths = [stage / name for stage in stages]
                _require(
                    paths[0].exists() == paths[1].exists(),
                    'Output presence mismatch: ' + name,
                )
                if name == 'appearance-roughness.png':
                    _require(
                        all(
                            ('roughnessTexture' in atlas) == path.exists()
                            for atlas, path in zip((a, b), paths)
                        ),
                        'Roughness asset presence does not match atlas metadata.',
                    )
                if not paths[0].exists():
                    _require(
                        name != 'appearance.png',
                        'The bake did not generate appearance.png.',
                    )
                    continue
                payload = paths[0].read_bytes()
                _require(
                    payload == paths[1].read_bytes(),
                    'Serial/threaded PNG bytes differ: ' + name,
                )
                hashes[name] = hashlib.sha256(payload).hexdigest()
            for atlas in (a, b):
                _require(
                    atlas.get('textureSha256') == hashes['appearance.png'],
                    'Atlas texture hash does not match the emitted PNG.',
                )
            _require(
                normalized[0] == normalized[1],
                'Serial/threaded atlas metadata differ beyond asset URLs.',
            )
            _require(
                (accepted / 'mesh.json').read_bytes() == snapshot,
                'Source mesh changed during verification.',
            )
            _require(
                code_fingerprint() == code_before,
                'Bake code changed during verification; results do not cover the current implementation.',
            )
            source_atlas = _json(accepted / 'texture-atlas.json')
            result = {
                'verified': True,
                'workers': [1, 4],
                'seconds': [round(item[1], 3) for item in results],
                'pngSha256': hashes,
                'positionsSha256': results[0][2],
                'atlasVertices': len(a['mapping']),
                'atlasTriangles': len(a['indices']) // 3,
                'sourceMeshUnchanged': True,
                'published': False,
                'sourceOutputMatches': {
                    name: (accepted / name).exists()
                    and hashlib.sha256((accepted / name).read_bytes()).hexdigest()
                    == digest
                    for name, digest in hashes.items()
                },
                'sourceAtlasMatches': all(
                    source_atlas.get(key) == a[key]
                    for key in ('mapping', 'indices', 'uv', 'positionsSha256')
                ),
            }
            if report is not None:
                destination = Path(report).resolve()
                _require(
                    destination != source and source not in destination.parents,
                    'Verification report must be outside the source capture.',
                )
                from face_pipeline import atomic

                atomic(
                    destination,
                    {
                        **result,
                        'codeSha256': code_before,
                        'appearanceStats': a['stats'],
                        'eyewearAudit': _json(stages[0] / 'eyewear-mask-audit.json'),
                    },
                )
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        _require(
            release_manifest(source) == release_snapshot,
            'Published release changed during verification.',
        )
        _require(
            (accepted / 'mesh.json').read_bytes() == snapshot,
            'Source mesh changed during verification; results are not valid.',
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'capture',
        type=Path,
        help='Existing completed capture; requires cached native detail when video is present.',
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Optional JSON evidence outside the source capture; no model artifacts are published.',
    )
    args = parser.parse_args()
    try:
        verify(args.capture, args.report)
    except Exception as error:
        print('Texture verification failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
