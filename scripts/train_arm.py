"""Real image -> COLMAP -> Brush Gaussian training -> experimental Poisson arm.

No proxy meshes are substituted on failure. Outputs retain the capture evidence.
"""
from pathlib import Path
import sys, json, subprocess, time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pycolmap
from plyfile import PlyData

folder=Path(sys.argv[1]).resolve()
root=Path(__file__).resolve().parents[1]
manifest=json.loads((folder/'capture.json').read_text())
status_path=folder/'status.json'
def status(state,message,**extra):
    payload={'status':state,'message':message,**extra}
    temp=folder/'status.tmp';temp.write_text(json.dumps(payload));temp.replace(status_path)

def triangulate_joint(rec,frames,key):
    lines=[]
    for image in rec.images.values():
        frame=frames.get(image.name)
        if not frame:continue
        p=frame['pose'][key] if isinstance(key,int) else frame['hand'][key[1]]
        cam=rec.cameras[image.camera_id]
        xy=cam.cam_from_img(np.array([p['x']*cam.width,p['y']*cam.height]))
        if xy is None:continue
        transform=image.cam_from_world();R=transform.rotation.matrix();t=transform.translation
        direction=R.T@np.array([xy[0],xy[1],1.]);direction/=np.linalg.norm(direction);center=-R.T@t
        lines.append((center,direction))
    if len(lines)<5:raise ValueError('Too few registered views to triangulate the arm joints.')
    A=np.zeros((3,3));b=np.zeros(3)
    for c,d in lines:
        P=np.eye(3)-np.outer(d,d);A+=P;b+=P@c
    point=np.linalg.lstsq(A,b,rcond=None)[0]
    return point,float(np.mean([np.linalg.norm(np.cross(point-c,d)) for c,d in lines]))

try:
    images=folder/'images';masks=folder/'masks';db=folder/'database.db';sparse=folder/'sparse';sparse.mkdir(exist_ok=True)
    frames={f['filename']:f for f in manifest['frames']}
    if len(frames)<12:raise ValueError('At least 12 separated, sharp arm views are required; aim for 60–100.')
    status('running','Finding features inside the captured arm masks…')
    reader=pycolmap.ImageReaderOptions();reader.mask_path=str(masks)
    opts=pycolmap.FeatureExtractionOptions();opts.max_image_size=1280;opts.num_threads=4;opts.sift.max_num_features=8192
    pycolmap.extract_features(str(db),str(images),camera_mode=pycolmap.CameraMode.SINGLE,camera_model='SIMPLE_RADIAL',reader_options=reader,extraction_options=opts,device=pycolmap.Device.cpu)
    status('running','Matching the same skin features across views…')
    matching=pycolmap.FeatureMatchingOptions();matching.num_threads=4
    pycolmap.match_exhaustive(str(db),matching_options=matching,device=pycolmap.Device.cpu)
    status('running','Recovering camera positions and checking scan consistency…')
    options=pycolmap.IncrementalPipelineOptions();options.num_threads=4;options.max_runtime_seconds=180;options.min_model_size=8;options.mapper.init_min_num_inliers=50
    maps=pycolmap.incremental_mapping(str(db),str(images),str(sparse),options=options)
    if not maps:raise ValueError('Camera poses could not be recovered. Keep the arm rigid, show more side views, and avoid blurred or textureless captures. Photos were retained; no substitute arm was generated.')
    rec=max(maps.values(),key=lambda r:r.num_reg_images())
    registered=rec.num_reg_images()
    if registered<max(10,int(len(frames)*.6)):raise ValueError(f'Only {registered}/{len(frames)} views registered. Recapture with overlapping views and a rigid arm.')
    main=sparse/'0';main.mkdir(exist_ok=True);rec.write(str(main))
    ids=[11,13,15] if manifest['side']=='left' else [12,14,16]
    joint_results=[triangulate_joint(rec,frames,key) for key in ids+[('hand',i) for i in range(21)]]
    joints=np.array([p for p,e in joint_results]);bone=np.linalg.norm(joints[1]-joints[2]);error=max(e for p,e in joint_results)/max(bone,1e-6)
    if error>.18:raise ValueError('Joint positions vary too much across views. The arm or fingers moved during capture; hold the pose rigid for reconstruction.')
    status('running',f'{registered}/{len(frames)} views registered. Training the Gaussian splat on the Apple GPU…')
    brush=root/'.local/tools/brush/brush-app-aarch64-apple-darwin/brush_app'
    if not brush.exists():raise ValueError('Brush trainer is not installed. Run the project setup script.')
    training=folder/'training';training.mkdir(exist_ok=True)
    with (folder/'training.log').open('w') as log:
        result=subprocess.run([str(brush),str(folder),'--total-steps','3000','--max-splats','200000','--max-resolution','960','--sh-degree','0','--export-every','3000','--export-path',str(training),'--export-name','arm.ply'],stdout=log,stderr=subprocess.STDOUT,timeout=900)
    ply=training/'arm.ply'
    if result.returncode or not ply.exists():raise ValueError('Gaussian training did not finish. Inspect training.log; your photographs and camera poses remain saved.')
    status('running','Extracting the arm surface and fitting captured joint anchors…')
    vertices=PlyData.read(ply)['vertex'].data
    scales=np.exp(np.clip(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]),-20,10))
    ordered=np.sort(scales,axis=1);thinness=float(np.median(ordered[:,0]/np.maximum(ordered[:,1],1e-12)))
    if thinness is not None and thinness>.5:raise ValueError('The trained splat is not sufficiently surface-aligned for a reliable mesh. The Gaussian PLY is saved; this capture needs surface regularization before rigging.')
    # Open3D and PyCOLMAP ship different OpenMP runtimes on macOS. Keep them in
    # separate processes; importing both caused a native crash in validation.
    mesh_path=folder/'mesh.json'
    conversion=subprocess.run([sys.executable,str(root/'scripts/convert.py'),str(ply),str(mesh_path),'--depth','8'],capture_output=True,text=True,timeout=90)
    if conversion.returncode or not mesh_path.exists():raise ValueError('Surface extraction failed. The trained Gaussian and captured images remain saved.')
    mesh=json.loads(mesh_path.read_text())
    transform=mesh['transform'];joints=(joints-np.array(transform['center']))*transform['scale']
    # Set forearm length using the median monocular pose estimate retained at
    # capture. This supplies approximate scale, not a physical measurement.
    measured=manifest.get('forearmCm');target_length=float(measured)/100 if measured else .27;
    if not .12<=target_length<=.5:raise ValueError('Elbow-to-wrist length must be 12–50 cm.')
    scale=target_length/max(np.linalg.norm(joints[2]-joints[1]),.01)
    positions=np.array(mesh['positions']).reshape(-1,3)*scale;joints*=scale
    mesh['positions']=positions.ravel().tolist()
    evidence={'registeredViews':registered,'inputViews':len(frames),'jointResidualRelativeToForearm':error,'medianSplatThinness':thinness,'surfaceMethod':mesh['stats']['method'],'scale':f'Forearm normalized to {target_length*100:g} cm; '+('user measurement.' if measured else 'unmeasured estimate.'),'limitation':'Experimental reconstruction. Inspect all views before accepting likeness. Auto skin weights are approximate.'}
    bundle={'format':'contact-arm','version':1,'side':manifest['side'],'mesh':mesh,'joints':joints.tolist(),'evidence':evidence}
    (folder/'arm-bundle.json').write_text(json.dumps(bundle))
    status('complete','Arm mesh reconstructed from the captured images. Inspect its shape and texture.',bundle=bundle)
except Exception as exc:
    status('failed',str(exc));raise
