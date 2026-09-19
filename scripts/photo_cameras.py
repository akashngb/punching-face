"""Recover calibrated camera views from photos; no splat training step."""
from pathlib import Path
import hashlib,json,math,shutil
import numpy as np
import pycolmap
from PIL import Image

def recover(folder,status):
    manifest=json.loads((folder/'capture.json').read_text());frames=manifest['frames'];signature=hashlib.sha256((folder/'capture.json').read_bytes()).hexdigest();cache=folder/'photo-cameras';metadata=folder/'photo-cameras.json'
    if cache.exists() and metadata.exists():
        info=json.loads(metadata.read_text())
        if info['captureHash']==signature:return pycolmap.Reconstruction(str(cache)),info
    # Existing scans already have camera poses recovered directly from images.
    # Reuse that calibration; no radiance-field or Gaussian file is read.
    legacy=folder/'dataset/sparse/0'
    if legacy.exists():
        rec=pycolmap.Reconstruction(str(legacy));valid=all(im.name in {f['filename'] for f in frames} for im in rec.images.values())
        if valid and rec.num_reg_images()>=max(18,math.ceil(len(frames)*.65)):
            cache.mkdir(exist_ok=True);rec.write(str(cache));info={'captureHash':signature,'inputViews':len(frames),'registeredViews':rec.num_reg_images(),'reprojectionErrorPx':rec.compute_mean_reprojection_error(),'cameraIntrinsics':'Reused camera calibration recovered from the original photographs.'};metadata.write_text(json.dumps(info));return rec,info
    images=folder/'images';db=folder/'photo-features.db';sparse=folder/'photo-sparse'
    db.unlink(missing_ok=True)
    if sparse.exists():shutil.rmtree(sparse)
    sparse.mkdir();status('cameras','Matching photographs and recovering camera views…')
    reader=pycolmap.ImageReaderOptions();reader.mask_path=str(folder/'masks');calibrated=manifest.get('horizontalFovDegrees') is not None;w,h=Image.open(images/frames[0]['filename']).size
    if calibrated:
        focal=w/(2*math.tan(math.radians(manifest['horizontalFovDegrees'])/2));reader.camera_params=f'{focal},{w/2},{h/2}'
    extraction=pycolmap.FeatureExtractionOptions();extraction.num_threads=4;extraction.max_image_size=1280;extraction.sift.max_num_features=8192;extraction.sift.peak_threshold=.002
    pycolmap.extract_features(str(db),str(images),camera_mode=pycolmap.CameraMode.SINGLE,camera_model='SIMPLE_PINHOLE' if calibrated else 'SIMPLE_RADIAL',reader_options=reader,extraction_options=extraction,device=pycolmap.Device.cpu)
    matching=pycolmap.FeatureMatchingOptions();matching.num_threads=4
    if len(frames)>80:
        pairing=pycolmap.SequentialPairingOptions();pairing.overlap=15;pairing.quadratic_overlap=True;pairing.loop_detection=False;pairing.num_threads=4
        pycolmap.match_sequential(str(db),matching_options=matching,pairing_options=pairing,device=pycolmap.Device.cpu)
    else:pycolmap.match_exhaustive(str(db),matching_options=matching,device=pycolmap.Device.cpu)
    options=pycolmap.IncrementalPipelineOptions();options.num_threads=4;options.max_runtime_seconds=90;options.min_model_size=10;options.mapper.init_min_num_inliers=50
    if calibrated:options.ba_refine_focal_length=False;options.ba_refine_extra_params=False;options.mapper.abs_pose_refine_focal_length=False;options.mapper.abs_pose_refine_extra_params=False
    candidates=list(pycolmap.incremental_mapping(str(db),str(images),str(sparse),options=options).values());minimum=max(18,math.ceil(len(frames)*.65))
    if not calibrated and max([r.num_reg_images() for r in candidates],default=0)<minimum:
        for ratio in (1.,1.6,2.2):
            attempt=folder/f'photo-focal-{ratio}';attempt.mkdir(exist_ok=True);seed=attempt/'database.db';shutil.copy2(db,seed)
            with pycolmap.Database.open(str(seed)) as database:
                for cam in database.read_all_cameras():cam.params=np.array([max(w,h)*ratio,w/2,h/2,0]);cam.has_prior_focal_length=True;database.update_camera(cam)
            options.max_runtime_seconds=45;options.ba_refine_focal_length=False;options.ba_refine_extra_params=False;options.mapper.abs_pose_refine_focal_length=False;options.mapper.abs_pose_refine_extra_params=False
            for rec in pycolmap.incremental_mapping(str(seed),str(images),str(attempt/'sparse'),options=options).values():
                if rec.num_reg_images()>=minimum:
                    ba=pycolmap.BundleAdjustmentOptions();ba.solver_options.num_threads=4;ba.solver_options.max_solver_time_in_seconds=30;pycolmap.bundle_adjustment(rec,ba)
                candidates.append(rec)
    if not candidates:raise ValueError('Photos do not contain enough matching views to recover depth.')
    rec=max(candidates,key=lambda r:(r.num_reg_images(),-r.compute_mean_reprojection_error()))
    if rec.num_reg_images()<minimum or rec.compute_mean_reprojection_error()>2.5:raise ValueError('Camera recovery failed. Capture sharper, overlapping, neutral-expression views from both sides.')
    cache.mkdir(exist_ok=True);rec.write(str(cache));info={'captureHash':signature,'inputViews':len(frames),'registeredViews':rec.num_reg_images(),'reprojectionErrorPx':rec.compute_mean_reprojection_error(),'cameraIntrinsics':'Estimated from photographs; absolute scale is not measured.'};metadata.write_text(json.dumps(info));return rec,info
