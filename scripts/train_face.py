"""Masked photographs -> verified SfM -> Gaussian training -> editable surface.

An experimental reconstruction, not an anatomical model or inferred identity.
No synthesized fallback is used when geometric verification fails.
"""
from pathlib import Path
import argparse,json,subprocess,sys,shutil,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pycolmap
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from PIL import Image
from face_pipeline import atomic
from openai_capture import review
ROOT=Path(__file__).resolve().parents[1]

def triangulate(rec,frames,index):
    rays=[]
    for image in rec.images.values():
        p=frames[image.name]['landmarks'][index];cam=rec.cameras[image.camera_id]
        xy=cam.cam_from_img(np.array([p['x']*cam.width,p['y']*cam.height]));pose=image.cam_from_world();R=pose.rotation.matrix();center=-R.T@pose.translation
        d=R.T@np.array([*xy,1]);d/=np.linalg.norm(d);rays.append((center,d))
    A=sum(np.eye(3)-np.outer(d,d) for _,d in rays);b=sum((np.eye(3)-np.outer(d,d))@c for c,d in rays)
    if np.linalg.cond(A)>1e6:raise ValueError('Too little parallax to recover facial structure. Include both side views.')
    point=np.linalg.solve(A,b);residual=np.median([np.linalg.norm(np.cross(point-c,d)) for c,d in rays])
    return point,float(residual)

def seed_face_surface(rec,frames,images,eye_span):
    # Topology supplies adjacency only. Every 3D vertex is triangulated from
    # the captured 2D landmarks, never copied from a canonical face shape.
    points=[];valid=[]
    for index in range(468):
        try:point,error=triangulate(rec,frames,index);ok=error<eye_span*.05 and np.isfinite(point).all()
        except (ValueError,np.linalg.LinAlgError):point=np.zeros(3);ok=False
        points.append(point);valid.append(ok)
    points=np.asarray(points);valid=np.asarray(valid)
    faces=[]
    for line in (ROOT/'public/models/canonical_face_model.obj').read_text().splitlines():
        if line.startswith('f '):faces.append([int(v.split('/')[0])-1 for v in line.split()[1:]])
    faces=np.asarray(faces);faces=faces[np.all(valid[faces],axis=1)];tri=points[faces]
    edges=np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2);faces=faces[edges.max(axis=1)<eye_span*.5];tri=points[faces]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if len(faces)<300 or area.sum()<1e-9:raise ValueError('Too few consistent facial landmark triangles. Record a neutral expression and clearer overlapping views.')
    rng=np.random.default_rng(42);chosen=rng.choice(len(faces),18000,p=area/area.sum());uv=rng.random((len(chosen),2));flip=uv.sum(axis=1)>1;uv[flip]=1-uv[flip];bary=np.column_stack([1-uv.sum(axis=1),uv]);samples=(points[faces[chosen]]*bary[:,:,None]).sum(axis=1)
    color=np.zeros((len(samples),3),np.uint8);assigned=np.zeros(len(samples),bool)
    # Prefer the front observation, filling any masked profile samples from
    # other registered views. These are observed RGB samples, not generated skin.
    for im in sorted(rec.images.values(),key=lambda im:abs(frames[im.name]['yaw'])):
        camera=rec.cameras[im.camera_id];pose=im.cam_from_world();cam=samples@pose.rotation.matrix().T+pose.translation;xy=camera.img_from_cam(cam);safe=np.isfinite(xy).all(axis=1)&(cam[:,2]>0);xy=np.nan_to_num(xy,nan=-1,posinf=-1,neginf=-1);ij=np.floor(xy).astype(int);pixels=np.array(Image.open(images/im.name).convert('RGBA'));h,w=pixels.shape[:2]
        safe&=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)&~assigned;idx=np.where(safe)[0];idx=idx[pixels[ij[idx,1],ij[idx,0],3]>200];color[idx]=pixels[ij[idx,1],ij[idx,0],:3];assigned[idx]=True
    for xyz,rgb in zip(samples[assigned],color[assigned]):rec.add_point3D(xyz,pycolmap.Track(),rgb)
    return {'triangulatedLandmarks':int(valid.sum()),'seedTriangles':int(len(faces)),'surfaceSeedPoints':int(assigned.sum()),'surfacePrior':'Landmark topology with triangulated geometry; interpolation supplies density, not captured detail.'}

def run(folder,cloud=False):
    def status(stage,message,**extra):atomic(folder/'status.json',{'status':'running','stage':stage,'message':message,**extra})
    manifest=json.loads((folder/'capture.json').read_text());frames={f['filename']:f for f in manifest['frames']};evidence={'inputViews':len(frames),'cloudReview':None,'syntheticTestFixture':bool(manifest.get('testFixture'))}
    try:
        for name in ('face.ply','mesh.json','texture-atlas.json','appearance.png'):(folder/name).unlink(missing_ok=True)
        if cloud:
            status('review','Reviewing up to six masked frames with OpenAI…')
            try:evidence['cloudReview']=review(folder,manifest['frames'])
            except ValueError as exc:evidence['cloudReview']={'error':str(exc),'advisoryOnly':True}
            atomic(folder/'review.json',evidence['cloudReview'])
        images=folder/'images';db=folder/'database.db';sparse=folder/'sparse'
        # Reattempts rebuild only derived files; capture images remain unchanged.
        for p in (sparse,folder/'training',folder/'dataset',folder/'sfm-seeds'):
            if p.exists():shutil.rmtree(p)
        db.unlink(missing_ok=True);sparse.mkdir()
        status('features','Matching facial texture across the recorded views…',evidence=evidence)
        reader=pycolmap.ImageReaderOptions();reader.mask_path=str(folder/'masks')
        calibrated=manifest.get('horizontalFovDegrees') is not None
        if calibrated:
            w,h=manifest['imageSize'];focal=w/(2*math.tan(math.radians(manifest['horizontalFovDegrees']/2)));reader.camera_params=f'{focal},{w/2},{h/2}'
        evidence['cameraIntrinsics']='User-supplied horizontal FOV; pinhole approximation' if calibrated else 'Estimated jointly from images'
        opts=pycolmap.FeatureExtractionOptions();opts.max_image_size=1280;opts.num_threads=4;opts.sift.max_num_features=8192;opts.sift.peak_threshold=.002
        pycolmap.extract_features(str(db),str(images),camera_mode=pycolmap.CameraMode.SINGLE,camera_model='SIMPLE_PINHOLE' if calibrated else 'SIMPLE_RADIAL',reader_options=reader,extraction_options=opts,device=pycolmap.Device.cpu)
        matching=pycolmap.FeatureMatchingOptions();matching.num_threads=4
        pycolmap.match_exhaustive(str(db),matching_options=matching,device=pycolmap.Device.cpu)
        status('cameras','Recovering camera positions and checking overlap…',evidence=evidence)
        options=pycolmap.IncrementalPipelineOptions();options.num_threads=4;options.max_runtime_seconds=90;options.min_model_size=10;options.mapper.init_min_num_inliers=50
        if calibrated:
            options.ba_refine_focal_length=False;options.ba_refine_extra_params=False;options.mapper.abs_pose_refine_focal_length=False;options.mapper.abs_pose_refine_extra_params=False
        maps=pycolmap.incremental_mapping(str(db),str(images),str(sparse),options=options)
        candidates=list(maps.values());minimum=max(18,math.ceil(len(frames)*.65))
        if not calibrated and max([r.num_reg_images() for r in candidates],default=0)<minimum:
            # Small smooth faces can make early focal/distortion refinement
            # degenerate. Bootstrap fixed focal hypotheses, then refine jointly
            # once enough images constrain the solution. Never lower quality gates.
            status('cameras','Stabilizing camera calibration for the close-up face…',evidence=evidence)
            seeds=folder/'sfm-seeds';seeds.mkdir();w,h=Image.open(images/next(iter(frames))).size
            for ratio in (1.0,1.6,2.2):
                attempt=seeds/str(ratio);attempt.mkdir();seed_db=attempt/'database.db';shutil.copy2(db,seed_db)
                with pycolmap.Database.open(str(seed_db)) as database:
                    for cam in database.read_all_cameras():
                        cam.params=np.array([max(w,h)*ratio,w/2,h/2,0]);cam.has_prior_focal_length=True;database.update_camera(cam)
                options.max_runtime_seconds=45;options.ba_refine_focal_length=False;options.ba_refine_extra_params=False;options.mapper.abs_pose_refine_focal_length=False;options.mapper.abs_pose_refine_extra_params=False
                found=pycolmap.incremental_mapping(str(seed_db),str(images),str(attempt/'sparse'),options=options)
                for candidate in found.values():
                    if candidate.num_reg_images()>=minimum:
                        ba=pycolmap.BundleAdjustmentOptions();ba.solver_options.num_threads=4;ba.solver_options.max_solver_time_in_seconds=30
                        pycolmap.bundle_adjustment(candidate,ba)
                    candidates.append(candidate)
            evidence['cameraIntrinsics']='Focal hypothesis bootstrap followed by joint bundle adjustment; estimated, not measured'
        if not candidates:raise ValueError('Camera poses could not be recovered. Record a neutral, still face with slower motion and more overlapping side views.')
        rec=max(candidates,key=lambda r:(r.num_reg_images(),-r.compute_mean_reprojection_error()));registered=rec.num_reg_images()
        if rec.compute_mean_reprojection_error()>2.5:raise ValueError('Camera reprojection error is too large. Capture sharper overlapping views.')
        evidence.update(registeredViews=registered,reprojectionErrorPx=float(rec.compute_mean_reprojection_error()))
        if registered<max(18,math.ceil(len(frames)*.65)):raise ValueError(f'Only {registered}/{len(frames)} views registered. More overlapping, sharp views are needed.')
        landmarks={};residual=[]
        for index in [33,263,10,152,1,61,291]:landmarks[index],error=triangulate(rec,frames,index);residual.append(error)
        width=np.linalg.norm(landmarks[263]-landmarks[33]);evidence['landmarkResidualRelativeToEyeSpan']=max(residual)/max(width,1e-8)
        if evidence['landmarkResidualRelativeToEyeSpan']>.08:raise ValueError('Facial structure moved between views. Keep a neutral expression and turn the head as one rigid object.')
        center=(landmarks[10]+landmarks[152])/2;right=landmarks[263]-landmarks[33];right/=np.linalg.norm(right)
        up=landmarks[10]-landmarks[152];up-=right*np.dot(up,right);up/=np.linalg.norm(up);forward=np.cross(right,up);B=np.stack([right,up,forward])
        if np.dot(landmarks[1]-(landmarks[33]+landmarks[263])/2,forward)<0:raise ValueError('The recovered face orientation is inconsistent. Capture clearer front and side views.')
        views=np.array([(im.projection_center()-center)/np.linalg.norm(im.projection_center()-center) for im in rec.images.values()]);span=float(np.degrees(np.arccos(np.clip((views@views.T).min(),-1,1))));evidence['cameraSpanDegrees']=span
        if span<35:raise ValueError(f'Recovered camera coverage is only {span:.0f} degrees. Capture more of each cheek and profile.')
        evidence.update(seed_face_surface(rec,frames,images,width))
        dataset=folder/'dataset';(dataset/'images').mkdir(parents=True);(dataset/'sparse/0').mkdir(parents=True)
        # Only registered images enter training. Transparent pixels remove the background.
        for im in rec.images.values():shutil.copy2(images/im.name,dataset/'images'/im.name)
        rec.write(str(dataset/'sparse/0'))
        status('gaussians',f'{registered} views verified. Training the Gaussian face on the GPU…',evidence=evidence)
        training=folder/'training';training.mkdir();brush=ROOT/'.local/tools/brush/brush-app-aarch64-apple-darwin/brush_app'
        with (folder/'training.log').open('w') as log:
            result=subprocess.run([str(brush),str(dataset),'--total-steps','3000','--max-splats','150000','--max-resolution','960','--sh-degree','0','--refine-every','100','--growth-select-fraction','0.3','--export-every','3000','--export-path',str(training),'--export-name','raw.ply'],stdout=log,stderr=subprocess.STDOUT,timeout=900)
        if result.returncode or not (training/'raw.ply').exists():raise ValueError('Gaussian training did not finish. The capture is saved; inspect the local training log.')
        # Align the learned splats to the observed facial axes, including covariance.
        ply=PlyData.read(training/'raw.ply');v=ply['vertex'].data
        xyz=(np.column_stack([v[k] for k in 'xyz'])-center)@B.T
        for i,k in enumerate('xyz'):v[k]=xyz[:,i]
        rot=Rotation.from_quat(np.column_stack([v[f'rot_{i}'] for i in (1,2,3,0)]));q=Rotation.from_matrix(B@rot.as_matrix()).as_quat()
        for i,k in enumerate((1,2,3,0)):v[f'rot_{k}']=q[:,i]
        ply.write(folder/'face.ply');evidence['gaussians']=len(v)
        sizes=np.exp(np.clip(np.column_stack([v[f'scale_{i}'] for i in range(3)]),-20,10));sizes.sort(axis=1);thin=float(np.median(sizes[:,0]/np.maximum(sizes[:,1],1e-12)));evidence['medianSplatThinness']=thin
        if thin>.5:raise ValueError('The Gaussian is saved, but its samples are too volumetric for this Poisson surface extractor. Surface regularization is needed before a likeness mesh can be accepted.')
        status('surface','Extracting and checking the editable face surface…',evidence=evidence)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/convert.py'),str(folder/'face.ply'),str(folder/'mesh.json'),'--depth','8'],capture_output=True,text=True,timeout=120)
        if result.returncode:raise ValueError('The splat was trained, but surface extraction failed. No substitute face has been generated.')
        mesh=json.loads((folder/'mesh.json').read_text());points=np.array(mesh['positions']).reshape(-1,3);transform=mesh['transform'];anchors=(np.array(list(landmarks.values()))-center)@B.T;anchors=(anchors-transform['center'])*transform['scale'];distances=cKDTree(points).query(anchors)[0]
        evidence['maxLandmarkSurfaceDistanceMm']=float(max(distances)*1000)
        if max(distances)>.018:raise ValueError('The extracted surface deviates too far from the observed face landmarks. Keep the saved Gaussian for inspection; this mesh is not accepted.')
        mesh['stats'].update(evidence,source='Synthetic public head test' if manifest.get('testFixture') else 'Recorded multiview face',scale='Normalized to nominal 28 cm height; not measured anatomy.',rig='Heuristic control fields; not a fitted anatomical rig.')
        atomic(folder/'mesh.json',mesh)
        status('texture','Baking the learned Gaussian appearance onto the mesh…',evidence=evidence)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/bake_texture.py'),'--folder',str(folder)],capture_output=True,text=True,timeout=300)
        if result.returncode:raise ValueError('The mesh is saved, but texture baking failed. Inspect the local reconstruction before using it.')
        atomic(folder/'status.json',{'status':'complete','stage':'ready','message':'Face splat and textured mesh ready. Inspect likeness from several angles before use.','evidence':evidence,'id':folder.name})
    except Exception as exc:
        atomic(folder/'status.json',{'status':'failed','stage':'failed','message':str(exc),'evidence':evidence,'splatAvailable':(folder/'face.ply').exists()});raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--cloud-review',action='store_true');args=parser.parse_args();run(args.folder.resolve(),args.cloud_review)
