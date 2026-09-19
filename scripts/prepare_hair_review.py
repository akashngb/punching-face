"""Prepare a local, ignored camera-matched photo/model review."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,pycolmap
from PIL import Image


def prepare(folder):
    out=Path(__file__).resolve().parents[1]/'public/generated/hair-review'
    out.mkdir(parents=True,exist_ok=True)
    mesh=json.loads((folder/'mesh.json').read_text())
    rec=pycolmap.Reconstruction(str(folder/'photo-cameras'))
    points=np.array(json.loads((folder/'surface-validation.json').read_text())['landmarksWorld'])
    center=(points[10]+points[152])/2
    right=points[263]-points[33];right/=np.linalg.norm(right)
    up=points[10]-points[152];up-=right*np.dot(up,right);up/=np.linalg.norm(up)
    B=np.stack([right,up,np.cross(right,up)]);scale=mesh['transform']['scale']
    frames={v['filename']:v for v in json.loads((folder/'capture.json').read_text())['frames']}
    selected=json.loads((folder/'hair-recognition.json').read_text())['views']
    images={im.name:im for im in rec.images.values()};views=[]
    for v in selected:
        im=images.get(v['filename'])
        if im is None:continue
        cam=rec.cameras[im.camera_id];pose=im.cam_from_world()
        original=np.asarray(Image.open(folder/'detail-images'/im.name).convert('RGBA'))
        factor=original.shape[1]/cam.width
        h,w=original.shape[:2]
        fx,fy=cam.focal_length_x*factor,cam.focal_length_y*factor
        cx,cy=cam.principal_point_x*factor,cam.principal_point_y*factor
        y,x=np.mgrid[:h,:w];rays=np.c_[((x-cx)/fx).ravel(),((y-cy)/fy).ravel(),np.ones(w*h)]
        uv=cam.img_from_cam(rays).reshape(h,w,2)*factor
        rectified=cv2.remap(original,uv[:,:,0].astype(np.float32),uv[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        box=Image.fromarray(rectified[:,:,3]).getbbox();left,top,right_edge,bottom=box
        pad=20;left=max(0,left-pad);top=max(0,top-pad);right_edge=min(w,right_edge+pad);bottom=min(h,bottom+pad)
        Image.fromarray(rectified[top:bottom,left:right_edge]).save(out/im.name)
        matrix=np.eye(4);flip=np.diag([1,-1,-1])
        matrix[:3,:3]=flip@pose.rotation.matrix()@B.T
        matrix[:3,3]=flip@(pose.rotation.matrix()@center+pose.translation)*scale
        camera_center=(im.projection_center()-center)@B.T
        views.append({'name':im.name,'time':frames[im.name]['timeSeconds'],
            'yaw':float(np.degrees(np.arctan2(camera_center[0],camera_center[2]))),
            'width':right_edge-left,'height':bottom-top,'fx':fx,'fy':fy,'cx':cx-left,'cy':cy-top,
            'viewMatrix':matrix.ravel().tolist(),'photo':im.name})
    for name in ['mesh.json','texture-atlas.json','appearance.png','appearance-roughness.png']:
        if (folder/name).exists():shutil.copy2(folder/name,out/name)
    (out/'review.json').write_text(json.dumps({'captureId':folder.name,'views':views,'groom':mesh['stats']['hairGroom'],
        'extraction':{'frames':len(frames),'registeredViews':len(rec.images),'detailSize':list(original.shape[1::-1])}}))
    print(out)


if __name__=='__main__':prepare(Path(sys.argv[1]).resolve())
