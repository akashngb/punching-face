"""Photographs -> recovered cameras -> fitted mesh -> photo texture -> rig cage."""

from pathlib import Path
import argparse, hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, trimesh
from scipy.spatial import cKDTree, Delaunay
from face_pipeline import atomic
from head_artifacts import HeadArtifactTransaction, has_published_model
from pipeline_timing import PipelineTimer
from scripts.photo_cameras import recover
from scripts.photo_geometry import (
    camera_rays,
    robust_landmarks,
    projection_error,
    make_surface,
    bake_photographs,
    eye_parts,
)
from scripts.astra_head_completion import complete, apply_shape_prior
from scripts.head_accessories import build_glasses, sample_frame_colors
from scripts.hair_groom import build_hair_groom
from scripts.hair_recognition import recognize_hair, hair_completion
from scripts.photo_hair import fit_template_hair
from scripts.template_selection import fit_selected_template
from scripts.surface_evidence import surface_projection_error
from scripts.eye_detail import scan_eyes, fit_eye_depth
from scripts.predict_rear import predict

ROOT = Path(__file__).resolve().parents[1]


def physics_binding(folder, positions, faces, observed_face_count):
    coarse = np.asarray(positions[:468])
    # Resample the simulation cage in the frontal plane. The rendering mesh
    # keeps all measured landmarks, but tiny eyelid/lip slivers are unsuitable
    # as volumetric finite elements. Keep separated nodes and reconnect them.
    boundary = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]
    selected = []
    for i in boundary + list(range(468)):
        if i not in selected and (
            not selected
            or np.min(np.linalg.norm(coarse[selected, :2] - coarse[i, :2], axis=1))
            > 0.003
        ):
            selected.append(i)
    triangles = np.asarray(selected)[Delaunay(coarse[selected, :2]).simplices]
    tri = coarse[triangles]
    reverse = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2] < 0
    triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
    tri = coarse[triangles]
    _, candidates = cKDTree(tri.mean(axis=1)).query(positions, k=12)
    nearby = tri[candidates]
    points = np.repeat(positions, 12, axis=0)
    closest = trimesh.triangles.closest_point(nearby.reshape(-1, 3, 3), points).reshape(
        -1, 12, 3
    )
    which = np.argmin(np.linalg.norm(closest - positions[:, None], axis=2), axis=1)
    chosen = candidates[np.arange(len(positions)), which]
    bary = trimesh.triangles.points_to_barycentric(
        tri[chosen], closest[np.arange(len(positions)), which]
    )
    bary = np.clip(bary, 0, 1)
    bary /= bary.sum(axis=1, keepdims=True)
    active = np.zeros(len(positions))
    active[np.unique(faces[:observed_face_count])] = 1
    bind = {
        'indices': triangles[chosen].ravel().tolist(),
        'weights': bary.ravel().tolist(),
        'active': active.tolist(),
    }
    cage = {
        'positions': coarse.ravel().tolist(),
        'indices': triangles.ravel().tolist(),
        'rigAnchors': {
            str(i): coarse[i].tolist()
            for i in [
                1,
                10,
                13,
                14,
                33,
                50,
                61,
                70,
                133,
                152,
                159,
                234,
                263,
                280,
                291,
                300,
                362,
                386,
                454,
            ]
        },
        'source': 'Triangulated photographic landmarks; no Gaussian input.',
    }
    atomic(folder / 'physics-cage.json', cage)
    atomic(folder / 'physics-binding.json', bind)
    return cage['rigAnchors']


def run(folder, use_astra=True):
    # The context outlives the worker pool: a failed parallel bake must finish
    # before its private output directory is removed.
    with HeadArtifactTransaction(folder) as publication:
        return _run(folder, use_astra, publication)


def _run(folder, use_astra, publication):
    output = publication.stage
    evidence = {'source': 'photographs', 'usesSplats': False}
    started = time.perf_counter()
    timer = PipelineTimer(folder)
    local_pool = ThreadPoolExecutor(2, thread_name_prefix="photo-local")

    def status(stage, message):
        timer.mark(stage)
        atomic(
            folder / 'status.json',
            {
                'status': 'running',
                'stage': stage,
                'message': message,
                'evidence': evidence,
                'photoModel': has_published_model(folder),
            },
        )

    try:
        status('photos', 'Building your 3D face from saved photographs…')
        rec, info = recover(folder, status)
        evidence.update(info)
        status(
            'geometry', 'Triangulating facial measurements and checking reserved views…'
        )
        manifest = json.loads((folder / 'capture.json').read_text())
        head_capture = manifest.get('captureRegion') == 'head'
        evidence['includesHairCapture'] = head_capture
        frames = {f['filename']: f for f in manifest['frames']}
        ordered = sorted(
            [im for im in rec.images.values() if frames[im.name].get('landmarks')],
            key=lambda im: frames[im.name]['yaw'],
        )
        if len(ordered) < 12:
            raise ValueError(
                'Camera recovery needs at least 12 registered facial views to fit the template.'
            )
        held = ordered[2::5]
        held_names = {im.name for im in held}
        train = [im for im in ordered if im.name not in held_names]
        points, _ = robust_landmarks(*camera_rays(rec, frames, train))
        test = projection_error(points, rec, frames, held)
        evidence['withheldLandmarks'] = {
            **test,
            'views': len(held),
            'usedForGeometryFit': False,
        }
        if test['medianPx'] > 4 or test['p95Px'] > 12:
            raise ValueError(
                'The surface does not explain withheld views. Capture sharper, neutral-expression photos.'
            )
        center = (points[10] + points[152]) / 2
        right = points[263] - points[33]
        right /= np.linalg.norm(right)
        up = points[10] - points[152]
        up -= right * np.dot(up, right)
        up /= np.linalg.norm(up)
        B = np.stack([right, up, np.cross(right, up)])
        if (points[1] - (points[33] + points[263]) / 2) @ B[2] < 0:
            raise ValueError('The recovered face orientation is inconsistent.')
        aligned = (points - center) @ B.T
        scale = 0.20 / (aligned[10, 1] - aligned[152, 1])
        transform = {'center': [0.0, 0.0, 0.0], 'scale': float(scale)}
        normalized = aligned * scale
        all_views = list(rec.images.values())
        angles = []
        for im in all_views:
            camera_local = (im.projection_center() - center) @ B.T
            angle = float(np.degrees(np.arctan2(camera_local[0], camera_local[2])))
            frames[im.name]['cameraYaw'] = angle
            angles.append(angle % 360)
        ordered_angles = np.sort(angles)
        span = float(
            360 - np.diff(np.r_[ordered_angles, ordered_angles[0] + 360]).max()
        )
        rear_views = sum(abs(frames[im.name]['cameraYaw']) > 115 for im in all_views)
        evidence['orbitCoverage'] = {
            'recoveredSpanDegrees': round(span, 1),
            'registeredRearViews': rear_views,
            'completeOrbit': span >= 300 and rear_views >= 3,
        }
        texture_views = all_views
        # Camera-derived template fitting does not depend on vision advice.
        template_job = local_pool.submit(
            fit_selected_template, normalized, rec, frames, train
        )
        advice = None
        if use_astra:
            status(
                'astra',
                'Astra is generating missing-head parameters and detecting 3D accessories from the saved frames…',
            )
            # Withheld views validate triangulation and the rendered surface.
            # Appearance and AI completion may use all registered photographs;
            # these are geometry-fit checks, not fully unseen-image evaluation.
            advice = complete(
                folder, evidence, {im.name: frames[im.name] for im in texture_views}
            )
            if head_capture and advice['hair']['present']:
                advice = hair_completion(advice, recognize_hair(folder, advice))
            evidence['astra'] = {
                'model': advice['model'],
                'framesSent': advice['framesSent'],
                'modelingPriorsOnly': True,
                'unobservedParts': advice['unobservedParts'],
                'completionVersion': advice['version'],
            }
        status(
            'eyes',
            (
                'Scanning the sharpest recorded eyes; generating missing detail with Astra when needed…'
                if use_astra
                else 'Checking recorded eyes and preparing labeled local eye estimates…'
            ),
        )
        eyes = scan_eyes(folder, manifest['frames'], use_astra)
        evidence['eyeDetail'] = {
            k: eyes[k]
            for k in [
                'summary',
                'model',
                'geometryEstimated',
                'generationMethod',
                'apiError',
            ]
        }
        evidence['eyeDetail']['eyes'] = {
            name: {k: spec[k] for k in ['mode', 'qualityReason', 'candidate']}
            for name, spec in eyes['eyes'].items()
        }
        status('surface', 'Fitting the smooth full-head template to the measured face…')
        p, f, face_count, template_info = template_job.result()
        evidence['templateFit'] = template_info
        semantics = None
        if advice and head_capture:
            from scripts.head_semantics import analyze
            from scripts.ear_fit import triangulate_ears, fit_ears, refine_ear_ownership

            status(
                'semantics',
                'Checking visible ears and opaque glasses against the source frames…',
            )
            semantics = analyze(folder, advice)
            if advice['glasses']['present']:
                from scripts.glasses_reference import generate_references

                status(
                    'accessory-cleanup',
                    'Estimating glasses-hidden skin and verifying alignment to the source photograph…',
                )
                try:
                    evidence['glassesCleanup'] = generate_references(
                        folder, advice, frames
                    )
                except ValueError as error:
                    evidence['glassesCleanup'] = {
                        'available': False,
                        'reason': str(error),
                        'fallback': 'Local opaque-frame cleanup; lens appearance remains captured.',
                    }
            measurements = triangulate_ears(
                folder, rec, center, B, transform, semantics
            )
            atomic(output / 'ear-measurements.json', measurements)
        hair_base = None
        if head_capture and (not advice or advice['hair']['present']):
            hair_base = p.copy()
            status(
                'hair',
                'Fitting the head and hair envelope to the captured silhouettes…',
            )
            p, hair = fit_template_hair(
                folder,
                p,
                f,
                face_count,
                rec,
                frames,
                texture_views,
                center,
                B,
                transform,
                template_info['earRegions'],
            )
            evidence['hair'] = hair
        if advice:
            p, evidence['headCompletion'] = apply_shape_prior(
                p, f, face_count, advice, evidence['orbitCoverage']
            )
        p, evidence['eyeDetail']['recessedMm'] = fit_eye_depth(p, eye_parts(p, f))
        pre_ear_surface = p.copy()
        if semantics:
            # Fit against the final head envelope. Later silhouette or AI
            # shape changes must not invalidate the ear quality checks.
            p, evidence['earFit'] = fit_ears(
                p, f, face_count, template_info['earRegions'], measurements
            )
            evidence['earOwnership'] = refine_ear_ownership(
                folder,
                p,
                f,
                template_info['earRegions'],
                semantics,
                rec,
                center,
                B,
                transform,
            )
        if hair_base is not None:
            from scripts.hair_transition import continue_hair_transition

            revised, evidence['hairTransition'] = continue_hair_transition(
                hair_base,
                pre_ear_surface,
                p,
                f,
                face_count,
                template_info['earRegions'],
            )
            # Keep ear fitting independent and repeatable on the revised nape.
            # The transition pins the entire fitted ear/attachment support.
            moved = np.any(revised != p, axis=1)
            pre_ear_surface[moved] += revised[moved] - p[moved]
            p = revised
        # Validate the final rendered skin after all shape stages. The earlier
        # triangulation gate alone cannot detect template fitting drift.
        surface_test = surface_projection_error(
            p,
            template_info['landmarkBindings'],
            rec,
            frames,
            held,
            center,
            B,
            transform,
        )
        evidence['withheldSurfaceLandmarks'] = {
            **surface_test,
            'usedForTemplateFit': False,
            'usedForParameterSelection': False,
            'appearanceAndCompletionMayUseTheseViews': True,
        }
        if surface_test['medianPx'] > 4 or surface_test['p95Px'] > 12:
            raise ValueError(
                (
                    'The fitted surface does not explain withheld face views. '
                    'Capture sharper, neutral-expression photos.'
                )
            )
        status('accessories', 'Fitting separate glasses and hair detail…')
        glasses = (
            build_glasses(p, advice, rec, frames, center, B, transform)
            if advice
            else None
        )
        if glasses:
            frame_color = sample_frame_colors(folder, advice)
            if frame_color:
                glasses.update(
                    frameColor=frame_color,
                    colorSource='Opaque photo frame samples; never part of the skin atlas',
                )
        evidence['accessories'] = {
            'glasses': bool(glasses),
            'source': (
                'Separate photo-fitted 3D eyewear'
                if glasses
                else 'No reconstructed accessory'
            ),
        }
        mesh = trimesh.Trimesh(p, f, process=False)
        mesh.fix_normals()
        p = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)
        if not mesh.is_watertight or not np.isfinite(p).all():
            raise ValueError('The photo model has invalid surface topology.')

        def bake():
            return bake_photographs(
                folder,
                p,
                f,
                face_count,
                rec,
                frames,
                texture_views,
                center,
                B,
                transform,
                advice,
                eyes,
                semantics,
                template_info['earRegions'],
                output_folder=output,
            )

        # The final surface is immutable here. Texture, hair fibers and the
        # physics cage use it independently. Rear-image generation, if needed,
        # must finish before baking, so that path retains its original order.
        texture_job = (
            local_pool.submit(bake)
            if not (head_capture and use_astra and rear_views < 3)
            else None
        )
        groom = (
            build_hair_groom(
                p,
                f,
                advice,
                rec,
                center,
                B,
                transform,
                folder,
                template_info['earRegions'],
            )
            if advice and head_capture
            else None
        )
        evidence['hairGroom'] = {
            'available': bool(groom),
            'type': advice['hair'].get('type') if advice else None,
            'roots': groom['rootCount'] if groom else 0,
            'estimatedFibers': True,
        }
        if groom:
            evidence['hairGroom'].update(
                version=groom['version'],
                recognition=groom.get('recognition'),
                **groom.get('evidence', {}),
            )
        status('rig', 'Binding the facial controls and tissue cage…')
        anchors = physics_binding(output, p, f, face_count)
        evidence.update(
            surfaceMode='photo-mesh',
            estimatedBackDepthMm=template_info['depthMm'],
            watertight=True,
        )
        stats = {
            **evidence,
            'source': 'Photo reconstruction',
            'vertices': len(p),
            'triangles': len(f),
            'observedFaceTriangles': face_count,
            'estimatedCraniumTriangles': len(f) - face_count,
            'method': template_info['method'],
            'scale': 'Hairline-to-chin height normalized to 20 cm; absolute scale is estimated.',
            'rig': 'Landmark facial controls with a Newton layered tetrahedral cage; tissue parameters are estimates.',
            'limitation': (
                (
                    'Visible hair uses captured shape and texture; hidden rear '
                    'head, individual strands and glasses-hidden skin are '
                    'estimated. Ear placement uses validated multiview landmarks '
                    'when available; inner-ear folds remain a template prior.'
                )
                if head_capture
                else (
                    'Gray rear head is estimated. Hair, ears, glasses geometry '
                    'and hidden surfaces were not fully captured.'
                )
            ),
        }
        data = {
            'positions': p.astype(np.float32).ravel().tolist(),
            'normals': np.asarray(mesh.vertex_normals)
            .astype(np.float32)
            .ravel()
            .tolist(),
            'indices': f.ravel().tolist(),
            'colors': np.tile([0.55, 0.43, 0.36], (len(p), 1)).ravel().tolist(),
            'transform': transform,
            'rigAnchors': anchors,
            'accessories': {'glasses': glasses, 'hair': groom},
            'stats': stats,
        }
        atomic(
            output / 'surface-validation.json',
            {
                'withheldFrames': [im.name for im in held],
                'trainingFrames': [im.name for im in train],
                'landmarksWorld': points.tolist(),
                'evidence': evidence,
            },
        )

        if head_capture and use_astra and rear_views < 3:
            status(
                'rear',
                'Predicting the rear head appearance from front and side photos…',
            )
            try:
                evidence['rearPrediction'] = predict(folder)
            except ValueError as error:
                evidence['rearPrediction'] = {
                    'available': False,
                    'error': str(error),
                    'fallback': 'Local photographic material continuation',
                }
            stats['rearPrediction'] = evidence['rearPrediction']
        status(
            'texture', 'Applying the photographs and fitted eye detail to the 3D mesh…'
        )
        stats['appearance'] = texture_job.result() if texture_job else bake()
        stats['seconds'] = round(time.perf_counter() - started, 2)
        # Retain the exact surface before ear fitting for deterministic future
        # detail rebuilds, including after template selection changes topology.
        baseline = output / '.pending-pre-ear-surface.npz'
        np.savez_compressed(
            baseline,
            positions=pre_ear_surface,
            indices=f,
            captureHash=hashlib.sha256(
                (folder / 'capture.json').read_bytes()
            ).hexdigest(),
            regularization=template_info['regularization'],
        )
        baseline.replace(output / 'pre-ear-surface.npz')
        atomic(output / 'mesh.json', data)
        publication.commit()
        timer.finish()
        atomic(
            folder / 'status.json',
            {
                'status': 'complete',
                'stage': 'ready',
                'message': 'Head and accessories fitted. Inspect the crown, profiles and 3D glasses. '
                + (
                    'Rear photos recovered.'
                    if rear_views >= 3
                    else 'Rear geometry remains a template estimate.'
                ),
                'evidence': evidence,
                'id': folder.name,
                'photoModel': True,
            },
        )
        print(json.dumps(stats, indent=2))
    except Exception as e:
        # No worker may publish after terminal failure is reported.
        local_pool.shutdown(wait=True)
        timer.finish('failed')
        atomic(
            folder / 'status.json',
            {
                'status': 'failed',
                'stage': 'failed',
                'message': str(e),
                'evidence': evidence,
                'photoModel': has_published_model(folder),
            },
        )
        raise
    finally:
        local_pool.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    # Optional Sentry (SPONSOR_SETUP.md): joins the request's trace; every PipelineTimer stage becomes a span. A no-op without a DSN.
    try:
        import sponsor_obs

        sponsor_obs.init('pipeline')
        sponsor_obs.patch_pipeline_timer()
        trace = sponsor_obs.continue_from_env('build_photo_face')
    except ImportError:
        from contextlib import nullcontext

        trace = nullcontext()
    with trace:
        run(args.folder.resolve(), not args.local_only)
