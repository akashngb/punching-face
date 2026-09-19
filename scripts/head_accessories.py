"""Fit independent eyeglass geometry and remove its projected ink from skin."""
import numpy as np
from scipy.ndimage import distance_transform_edt,gaussian_filter
from PIL import Image
import cv2

PATH_KEYS=('imageLeftLens','imageRightLens','bridge','imageLeftTemple','imageRightTemple')

def paths_pixels(view,crop):
    left,top,right,bottom=crop
    return {k:np.asarray(view[k],dtype=float)*[right-left,bottom-top]+[left,top]
            for k in PATH_KEYS if len(view.get(k,[]))>=2}


def clean_view(pixels,filename,spec,frame,frames=None,return_details=False,projected_mask=None):
    def result(value,mask=None,method="not-detected"):
        if mask is None:mask=np.zeros(pixels.shape[:2],np.uint8)
        audit={"method":method,"excludedPixels":int(np.count_nonzero(mask)),"hiddenSurfaceEstimated":bool(mask.any()),"preservedEyePhotographs":True}
        return (value,mask,audit) if return_details else value
    if not spec or not spec['glasses']['present']:return result(pixels)
    view=next((v for v in spec['views'] if v['filename']==filename),None)
    if (not view or not any(len(view.get(k,[]))>=2 for k in PATH_KEYS)) and frames and frame.get('landmarks'):
        candidates=[v for v in spec['views'] if frames.get(v['filename'],{}).get('landmarks') and any(len(v.get(k,[]))>=2 for k in PATH_KEYS)]
        if candidates:view=min(candidates,key=lambda v:abs((frames[v['filename']].get('yaw') or 0)-(frame.get('yaw') or 0)))
    if not view:return result(pixels,method='no-visible-contour')
    rgb=pixels[:,:,:3].copy();mask=np.zeros(rgb.shape[:2],np.uint8);paths=paths_pixels(view,spec['crops'][view['filename']])
    if view['filename']!=filename:
        ids=[33,133,159,145,263,362,386,374,168,6,70,300,105,334,107,336]
        h,w=rgb.shape[:2]
        def xy(f):return np.array([[f['landmarks'][i]['x']*w,f['landmarks'][i]['y']*h] for i in ids])
        matrix,_=cv2.findHomography(xy(frames[view['filename']]),xy(frame),cv2.RANSAC,3.)
        if matrix is None:return result(pixels)
        paths={k:cv2.perspectiveTransform(v[None].astype(np.float64),matrix)[0] for k,v in paths.items()}
    # Fill outside-alpha RGB first, so the cutout background cannot bleed into
    # an inpainted temple at the head silhouette.
    valid=pixels[:,:,3]>128
    if valid.any():
        _,nearest=distance_transform_edt(~valid,return_indices=True);rgb[~valid]=rgb[nearest[0][~valid],nearest[1][~valid]]
    # Narrow geometric masks remove only the actual opaque frame; the eyes,
    # eyebrows and all other measured identity features remain in the photo.
    eye_width=abs(frame['landmarks'][263]['x']-frame['landmarks'][33]['x'])*rgb.shape[1] if frame.get('landmarks') else (spec['crops'][view['filename']][2]-spec['crops'][view['filename']][0])*.45
    thickness=max(3,int(eye_width*.09))
    for name,path in paths.items():
        points=np.rint(path).astype(np.int32)
        cv2.polylines(mask,[points],name.endswith('Lens') and len(points)>=8,255,thickness,lineType=cv2.LINE_AA)
    if projected_mask is not None:mask=np.maximum(mask,projected_mask)
    if not mask.any():return result(pixels,method='no-visible-contour')
    rgb=cv2.inpaint(rgb,mask,max(3,thickness//2),cv2.INPAINT_TELEA)
    output=pixels.copy();output[:,:,:3]=rgb;return result(output,mask,'opaque-frame-inpaint')

def sample_frame_colors(folder,spec):
    """Sample opaque frame centerlines into the eyewear material ONLY."""
    samples=[]
    for view in spec.get('views',[]):
        pixels=np.asarray(Image.open(folder/'images'/view['filename']).convert('RGB'))/255.
        h,w=pixels.shape[:2]
        for path in paths_pixels(view,spec['crops'][view['filename']]).values():
            for a,b in zip(path[:-1],path[1:]):
                xy=a[None]+np.linspace(0,1,max(2,int(np.linalg.norm(b-a))))[:,None]*(b-a)[None]
                ij=np.rint(xy).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
                ij=ij[inside];samples.extend(pixels[ij[:,1],ij[:,0]])
    if not samples:return None
    samples=np.asarray(samples);base=np.asarray(spec['glasses']['frameColorSrgb'])
    # Reject skin/lens pixels where an approximate path is a few pixels off.
    distance=np.linalg.norm(samples-base,axis=1);good=samples[distance<.20]
    return np.median(good,axis=0).tolist() if len(good)>16 else None


def build_glasses(p,spec,rec,frames,center,B,transform):
    if not spec or not spec['glasses']['present'] or spec['glasses']['confidence']<.6:return None
    name=spec['frontFilename'];view=next(v for v in spec['views'] if v['filename']==name)
    paths=paths_pixels(view,spec['crops'][name]);im=next((im for im in rec.images.values() if im.name==name),None)
    if im is None:return None
    cam=rec.cameras[im.camera_id];pose=im.cam_from_world();origin=(im.projection_center()-center)@B.T*transform['scale']
    info=spec['glasses'];half_width=max(abs(p[234,0]),abs(p[454,0]));z0=p[168,2]+info['bridgeClearanceMm']*.001
    def lift(xy):
        rays=np.column_stack([cam.cam_from_img(xy),np.ones(len(xy))])@pose.rotation.matrix()@B.T
        rays/=np.linalg.norm(rays,axis=1)[:,None];depth=np.full(len(xy),z0)
        for _ in range(5):
            q=origin+rays*((depth-origin[2])/rays[:,2])[:,None]
            depth=z0-.016*(q[:,0]/half_width)**2
        q=origin+rays*((depth-origin[2])/rays[:,2])[:,None]
        return q
    rims=[lift(paths[k]) for k in ['imageLeftLens','imageRightLens'] if k in paths and len(paths[k])>=8]
    if len(rims)!=2:return None
    bridge=lift(paths['bridge']) if 'bridge' in paths else np.array([rims[0][np.argmax(rims[0][:,0])],rims[1][np.argmin(rims[1][:,0])]])
    temples=[]
    for rim in rims:
        sign=np.sign(rim[:,0].mean());outer=np.where(rim[:,0]*sign>(rim[:,0]*sign).max()-.005)[0];i=outer[np.argmax(rim[outer,1])];hinge=rim[i].copy()
        # Temple arms are a smooth, editable prior behind observed front rims.
        y=hinge[1];arm=[hinge]
        for depth,drop in [(-.025,0),(-.055,.001),(-.085,.003),(-.112,.007),(-.119,.020)]:
            height=y-drop;section=p[(np.abs(p[:,1]-height)<.009)&(np.abs(p[:,2]-depth)<.008)&(p[:,0]*sign>0)]
            outer=float(np.quantile(section[:,0]*sign,.92))+.003 if len(section) else half_width+.003
            # Follow the fitted temple rather than flaring immediately out to
            # the widest point of the cheek/ear.
            side=sign*max(abs(hinge[0]),outer)
            arm.append([side,height,depth])
        temples.append(np.asarray(arm))
    return {'type':'eyeglasses','source':'Astra photo contours + recovered frontal camera; depth and temple fit estimated','model':spec['model'],'rims':[r.tolist() for r in rims],'bridge':bridge.tolist(),'temples':[t.tolist() for t in temples],'frameColor':info['frameColorSrgb'],'radius':info['frameRadiusMm']*.001,'templeWidth':info['templeWidthMm']*.001,'lensTint':info['lensTint'],'estimated':True,'description':info['description']}
