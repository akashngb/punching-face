"""build_photo_face.py with scripts/pipeline_accel.py installed. Same arguments, same outputs.

The stages, their order and every result file are build_photo_face.run()'s own.
Only two things differ: slow leaf functions are replaced by exact equivalents,
and the independent AI calls are started together instead of one after another.
CONTACT_PREFETCH=0 keeps the AI calls serial; CONTACT_SERIAL_PIPELINE=1 (read by
face_pipeline.py) bypasses this launcher entirely. See PIPELINE_SPEEDUP.md.
"""

from pathlib import Path
import argparse, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Leaves first: other modules bind these functions by name as they are imported.
import scripts.pipeline_accel as accel

accel.install_leaves()
import scripts.build_photo_face as build

accel.install_prefetch(build)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    # Optional Sentry (SPONSOR_SETUP.md): the same trace hook as build_photo_face.py.
    try:
        import sponsor_obs

        sponsor_obs.init('pipeline')
        sponsor_obs.patch_pipeline_timer()
        trace = sponsor_obs.continue_from_env('build_photo_face')
    except ImportError:
        from contextlib import nullcontext

        trace = nullcontext()
    with trace:
        build.run(args.folder.resolve(), not args.local_only)
