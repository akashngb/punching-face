"""Photo-derived face geometry and texture. No radiance-field reconstruction."""
from pathlib import Path
import json
import hashlib
import numpy as np
import trimesh,xatlas
from PIL import Image
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import distance_transform_edt,map_coordinates
from face_pipeline import atomic
from scripts.head_material import rear_reference,missing_head_material
from scripts.head_accessories import clean_view,build_glasses
from scripts.eye_detail import apply_eye_material
from scripts.photo_detail import prepare_detail_frames,detail_image
import cv2
ROOT=Path(__file__).resolve().parents[1]
OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

def camera_rays(rec,frames,images):
    centers=[];directions=[]
    for im in images:
        cam=rec.cameras[im.camera_id];xy=np.array([[p['x']*cam.width,p['y']*cam.height] for p in frames[im.name]['landmarks']])
        xy=cam.cam_from_img(xy);pose=im.cam_from_world();rays=np.column_stack([xy,np.ones(len(xy))])@pose.rotation.matrix();rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        centers.append(im.projection_center());directions.append(rays)
    return np.asarray(centers),np.asarray(directions)

def robust_landmarks(centers,directions):
    projectors=np.eye(3)-directions[:,:,:,None]*directions[:,:,None,:]
    weight=np.ones(directions.shape[:2]);points=None
    for _ in range(8):
        A=np.einsum('vn,vnij->nij',weight,projectors);b=np.einsum('vn,vnij,vj->ni',weight,projectors,centers)
        if np.max(np.linalg.cond(A))>1e6:raise ValueError('Insufficient angular diversity for stable face depth.')
        points=np.linalg.solve(A,b);delta=points[None]-centers[:,None];errors=np.linalg.norm(np.cross(delta,directions),axis=-1)
        robust_scale=np.maximum(np.median(errors,axis=0)*1.4826,1e-7);weight=np.minimum(1,1.5*robust_scale/np.maximum(errors,1e-9))
    return points,errors

def make_surface(points,back_ratio):
    faces=[]
    for line in (ROOT/'public/models/canonical_face_model.obj').read_text().splitlines():
        if line.startswith('f '):faces.append([int(x.split('/')[0])-1 for x in line.split()[1:]])
    face_count=len(faces);verts=points.tolist();boundary=points[OVAL];previous=np.array(OVAL)
    width=np.ptp(boundary[:,0]);height=np.ptp(boundary[:,1]);rear_depth=width*back_ratio;middle=np.array([np.mean(boundary[:,0]),height*.20,0.])
    # This smooth cap is an explicit prior. It does not claim recovered ears,
    # hair, scalp or occipital shape, and receives a separate neutral texture.
    for theta in np.linspace(0,np.pi/2,20)[1:-1]:
        ring=middle+(boundary-middle)*np.cos(theta);ring[:,2]-=rear_depth*np.sin(theta)
        ring[:,1]+=height*.31*np.sin(2*theta)*np.clip((boundary[:,1]/height+.20)/.70,0,1)
        current=np.arange(len(verts),len(verts)+len(OVAL));verts.extend(ring.tolist())
        for j in range(len(OVAL)):
            k=(j+1)%len(OVAL);faces.extend([[int(previous[j]),int(current[j]),int(previous[k])],[int(previous[k]),int(current[j]),int(current[k])]])
        previous=current
    pole=len(verts);verts.append([middle[0],middle[1],middle[2]-rear_depth])
    for j in range(len(OVAL)):faces.append([int(previous[j]),pole,int(previous[(j+1)%len(OVAL)])])
    base=trimesh.Trimesh(verts,faces,process=False);base.fix_normals()
    if not base.is_watertight:raise ValueError('Head completion has an invalid boundary.')
    p,f=trimesh.remesh.subdivide_loop(base.vertices,base.faces,iterations=2)
    # Restore observed landmarks after smooth subdivision, without copying a
    # stock face shape. Anchor rear points to keep the correction local.
    pins=np.r_[np.arange(468),np.arange(468,len(base.vertices),12)]
    target=np.asarray(base.vertices)[pins];correction=target-p[pins]
    warp=RBFInterpolator(p[pins],correction,kernel='thin_plate_spline',smoothing=1e-10,neighbors=24)
    p+=warp(p);p[:468]=points
    return p,f,face_count*16,rear_depth

def projection_error(points,rec,frames,images):
    errors=[]
    for im in images:
        cam=rec.cameras[im.camera_id];pose=im.cam_from_world();predicted=cam.img_from_cam(points@pose.rotation.matrix().T+pose.translation)
        observed=np.array([[p['x']*cam.width,p['y']*cam.height] for p in frames[im.name]['landmarks']])
        errors.extend(np.linalg.norm(predicted-observed,axis=1).tolist())
    return {'medianPx':float(np.median(errors)),'p95Px':float(np.quantile(errors,.95))}

def raster_atlas(p,n,mapping,indices,uv,is_face,size,part_labels=None):
    world=np.zeros((size,size,3),np.float32);normal=np.zeros_like(world);covered=np.zeros((size,size),bool);observed=np.zeros_like(covered)
    parts=np.zeros((size,size),np.uint8) if part_labels is not None else None
    for i,triangle in enumerate(indices):
        t=uv[triangle]*size;lo=np.maximum(np.floor(t.min(axis=0)).astype(int),0);hi=np.minimum(np.ceil(t.max(axis=0)).astype(int),size-1)
        if np.any(lo>hi):continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5);d=(t[1,1]-t[2,1])*(t[0,0]-t[2,0])+(t[2,0]-t[1,0])*(t[0,1]-t[2,1])
        if abs(d)<1e-9:continue
        a=((t[1,1]-t[2,1])*(xx-t[2,0])+(t[2,0]-t[1,0])*(yy-t[2,1]))/d;b=((t[2,1]-t[0,1])*(xx-t[2,0])+(t[0,0]-t[2,0])*(yy-t[2,1]))/d;c=1-a-b;inside=(a>=-1e-5)&(b>=-1e-5)&(c>=-1e-5)
        w=np.stack([a,b,c],axis=-1);ids=mapping[triangle];region=np.s_[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        world[region][inside]=(w@p[ids])[inside];normal[region][inside]=(w@n[ids])[inside];covered[region]|=inside;observed[region][inside]=is_face[i]
        if parts is not None:parts[region][inside]=part_labels[ids[0]]
    normal/=np.maximum(np.linalg.norm(normal,axis=2,keepdims=True),1e-9)
    result=(world[covered],normal[covered],observed[covered],covered)
    return result+(parts[covered],) if parts is not None else result


def eye_parts(p,f):
    """Identify independent template eyeballs, without marking lids as eyes."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(p),len(p)))
    _,components=connected_components(graph,directed=False);labels=np.zeros(len(p),np.uint8)
    for label in np.unique(components[468:]):
        ids=np.where(components==label)[0]
        if not 100<=len(ids)<=3000:continue
        points=p[ids];extent=np.ptp(points,axis=0);center=points.mean(0)
        if np.any(extent<.015) or np.any(extent>.050):continue
        for part,corners in [(1,[33,133]),(2,[263,362])]:
            if np.linalg.norm(center-p[corners].mean(0))<.030:labels[ids]=part
    return labels


def zbuffer(projected,depth,faces,width,height):
    result=np.full((height,width),np.inf)
    for face in faces:
        t=projected[face];z=depth[face]
        if np.min(z)<=0:continue
        lo=np.maximum(np.floor(t.min(axis=0)).astype(int),0);hi=np.minimum(np.ceil(t.max(axis=0)).astype(int),[width-1,height-1])
        if np.any(lo>hi):continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5);d=(t[1,1]-t[2,1])*(t[0,0]-t[2,0])+(t[2,0]-t[1,0])*(t[0,1]-t[2,1])
        if abs(d)<1e-9:continue
        a=((t[1,1]-t[2,1])*(xx-t[2,0])+(t[2,0]-t[1,0])*(yy-t[2,1]))/d;b=((t[2,1]-t[0,1])*(xx-t[2,0])+(t[0,0]-t[2,0])*(yy-t[2,1]))/d;c=1-a-b;inside=(a>=0)&(b>=0)&(c>=0)
        interpolated=1/np.maximum(a/z[0]+b/z[1]+c/z[2],1e-12);tile=result[lo[1]:hi[1]+1,lo[0]:hi[0]+1];tile[inside]=np.minimum(tile[inside],interpolated[inside])
    return result


def project_eyewear_mask(glasses,cam,pose,center,B,transform,head_depth):
    """Only mask eyewear in front of the head: rear photos remain usable."""
    if glasses is None:return None
    h,w=head_depth.shape;mask=np.zeros((h,w),np.uint8)
    def project(points):
        world=np.asarray(points)/transform['scale']@B+center
        cp=world@pose.rotation.matrix().T+pose.translation
        return cp,cam.img_from_cam(cp)
    # Project only opaque frame paths. Clear lens interiors carry actual eye
    # evidence and must not be replaced by skin or generic eyeball priors.
    paths=[*(list(r)+[r[0]] for r in glasses['rims']),glasses['bridge'],*glasses['temples']]
    for path in paths:
        for a,b in zip(path[:-1],path[1:]):
            samples=np.asarray(a)[None]+np.linspace(0,1,40)[:,None]*(np.asarray(b)-a)[None]
            cp,xy=project(samples);ij=np.rint(np.nan_to_num(xy,nan=-1,posinf=-1,neginf=-1)).astype(int)
            inside=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)&(cp[:,2]>0)
            for j in np.where(inside)[0]:
                x,y=ij[j]
                if cp[j,2]<=head_depth[y,x]+.004/transform['scale']:
                    offset=samples[j]+[glasses['radius']+.002,0,0];_,projected=project([offset]);radius=max(3,int(np.linalg.norm(projected[0]-xy[j])))
                    cv2.circle(mask,(int(x),int(y)),min(radius,25),255,-1)
    return cv2.dilate(mask,np.ones((5,5),np.uint8))

def bake_photographs(folder,p,f,face_count,rec,frames,train,center,B,transform,completion=None,eyes=None):
    detail_audit=prepare_detail_frames(folder)
    head_capture=json.loads((folder/'capture.json').read_text()).get('captureRegion')=='head'
    size=3072;m=trimesh.Trimesh(p,f,process=False);normal=np.array(m.vertex_normals)
    atlas=xatlas.Atlas();atlas.add_mesh(p.astype(np.float32),f.astype(np.uint32));pack=xatlas.PackOptions();pack.resolution=size;pack.padding=4;atlas.generate(pack_options=pack);mapping,indices,uv=atlas[0]
    face_keys={tuple(sorted(face)) for face in f[:face_count]};is_face=np.array([tuple(sorted(face)) in face_keys for face in mapping[indices]])
    eye_labels=eye_parts(p,f)
    texel,tn,observed,covered,parts=raster_atlas(p,normal,mapping,indices,uv,is_face,size,eye_labels)
    world=(texel/transform['scale']+transform['center'])@B+center;world_n=tn@B;verts_world=(p/transform['scale']+transform['center'])@B+center
    # Neutral shading identifies the inferred cranium instead of inventing a
    # hair/scalp texture for regions the camera never captured.
    light=np.array([-.3,.7,1.]);light/=np.linalg.norm(light);shade=.6+.4*np.maximum(tn@light,0)
    color=np.array([.31,.38,.36])[None]*shade[:,None];best=np.zeros(len(texel));total=np.zeros(len(texel));accum=np.zeros_like(color);visible_any=np.zeros(len(texel),bool)
    estimated_total=np.zeros(len(texel));estimated_color=np.zeros_like(color);mask_audit=[]
    detail_best=np.zeros(len(texel));fine_detail=np.zeros_like(color)
    yaw=lambda im:frames[im.name].get('cameraYaw',frames[im.name].get('yaw') or 0)
    facial=[im for im in train if frames[im.name].get('landmarks')]
    front=min(facial,key=lambda im:abs(yaw(im)))
    targets=list(range(-180,180,20)) if head_capture else [-45,0,45]
    selected={min(train,key=lambda im:abs((yaw(im)-angle+180)%360-180)).name for angle in targets};selected.add(front.name)
    if completion and completion['glasses']['present']:
        annotated={v['filename'] for v in completion['views']}
        selected|={im.name for im in train if im.name in annotated}
        front=next((im for im in train if im.name==completion['frontFilename']),front)
    # One frontal exposure owns the eyes, nose and mouth. Side cameras enter
    # through broad smooth weights, never a per-texel winner that cuts a face
    # into mismatched photographic fragments.
    width=np.ptp(p[:468,0]);side_mix=np.clip((np.abs(texel[:,0])-width*.25)/(width*.22),0,1);side_mix=side_mix*side_mix*(3-2*side_mix)
    frontal_forehead=(texel[:,1]>p[10,1]-.025)&(texel[:,2]>-.045)
    central=(observed|frontal_forehead)*(1-side_mix)
    def cheek_color(im):
        frame=frames[im.name]
        if not frame.get('landmarks'):return None
        px=np.asarray(Image.open(folder/'images'/im.name).convert('RGB'))/255.;samples=[];h,w=px.shape[:2]
        for idx in [50,280,205,425]:
            lm=frame['landmarks'][idx];x=int(lm['x']*w);y=int(lm['y']*h);samples.extend(px[max(0,y-5):y+6,max(0,x-5):x+6].reshape(-1,3))
        return np.median(samples,axis=0)
    reference_color=cheek_color(front)
    glasses=build_glasses(p,completion,rec,frames,center,B,transform) if completion else None
    for im in [im for im in train if im.name in selected]:
        cam=rec.cameras[im.camera_id];pose=im.cam_from_world();R=pose.rotation.matrix();translation=pose.translation;cp=world@R.T+translation;xy=cam.img_from_cam(cp);ij=np.floor(np.nan_to_num(xy,nan=-1,posinf=-1,neginf=-1)).astype(int)
        h,w=cam.height,cam.width
        raw=detail_image(folder,im.name);pixel_scale=raw.shape[1]/w
        vertex_cp=verts_world@R.T+translation;depth=zbuffer(cam.img_from_cam(vertex_cp),vertex_cp[:,2],f,w,h)
        projected_mask=project_eyewear_mask(glasses,cam,pose,center,B,transform,depth)
        detail_completion=completion
        if pixel_scale!=1 and completion:
            detail_completion={**completion,'crops':{name:(np.asarray(crop)*pixel_scale).tolist() for name,crop in completion['crops'].items()}}
        if projected_mask is not None:projected_mask=cv2.resize(projected_mask,(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_NEAREST)
        pixels,accessory_mask,audit=clean_view(raw,im.name,detail_completion,frames[im.name],frames,return_details=True,projected_mask=projected_mask)
        mask_audit.append({'filename':im.name,**audit});inside=(ij[:,0]>=0)&(ij[:,0]<w-1)&(ij[:,1]>=0)&(ij[:,1]<h-1)&(cp[:,2]>0)&(observed|head_capture)
        ij[:,0]=np.clip(ij[:,0],0,w-1);ij[:,1]=np.clip(ij[:,1],0,h-1);x,y=ij.T
        visible=np.abs(cp[:,2]-depth[y,x])*transform['scale']<.006
        toward=im.projection_center()-world;toward/=np.maximum(np.linalg.norm(toward,axis=1,keepdims=True),1e-9);facing=np.maximum(np.sum(world_n*toward,axis=1),0)
        visible_any|=inside&visible&(facing>.05)
        preference=(1+central*30) if im.name==front.name else (1-central*.97)
        # Segmentation leaves a pale fringe at some neck/hair cutouts. Fade
        # projections before the cutout boundary so it cannot become a sharp
        # diagonal stripe when another view or the inferred material takes over.
        base_alpha=cv2.resize(pixels[:,:,3],(w,h),interpolation=cv2.INTER_NEAREST)
        boundary_distance=distance_transform_edt(base_alpha>128)
        edge_weight=np.clip(boundary_distance[y,x]/10.,0,1)
        edge_weight=edge_weight*edge_weight*(3-2*edge_weight)
        quality=facing**8*inside*visible*(base_alpha[y,x]/255)*preference*edge_weight
        sample_xy=np.nan_to_num(xy,nan=-1,posinf=-1,neginf=-1)*pixel_scale
        rgb=np.stack([map_coordinates(pixels[:,:,c].astype(float),[sample_xy[:,1]-.5,sample_xy[:,0]-.5],order=1,mode='nearest') for c in range(3)],axis=1)/255.
        # Blend exposure at low frequencies; retain fine hairs from one clear
        # camera instead of averaging misaligned eyebrow/beard/lock edges.
        smooth=cv2.GaussianBlur(pixels[:,:,:3].astype(np.float32)/255,(0,0),1.4*pixel_scale)
        low_rgb=np.stack([map_coordinates(smooth[:,:,c],[sample_xy[:,1]-.5,sample_xy[:,0]-.5],order=1,mode='nearest') for c in range(3)],axis=1)
        # Reject the bright cutout fringe without removing legitimate skin.
        fringe=(rgb.min(axis=1)>.76)&(np.ptp(rgb,axis=1)<.10);quality*=~((~observed)&fringe)
        median=cheek_color(im)
        if median is not None:
            correction=np.clip(reference_color/np.maximum(median,.05),.8,1.25)
            rgb*=correction;low_rgb*=correction
        masked=cv2.resize(accessory_mask,(w,h),interpolation=cv2.INTER_NEAREST)[y,x]>0
        # Clean unoccluded photographs always win. Only use estimated fills
        # when no camera can see the skin behind the accessory.
        # The small opaque frame fill replaces only those pixels. Lenses,
        # eyes and brows retain their observed photo appearance.
        estimate=quality*masked;estimated_total+=estimate;estimated_color+=rgb*estimate[:,None]
        quality*=~masked
        take=quality>detail_best;fine_detail[take]=(rgb-low_rgb)[take];detail_best=np.maximum(detail_best,quality)
        best=np.maximum(best,quality);total+=quality;accum+=low_rgb*quality[:,None]
        print('Projected capture',im.name,flush=True)
    supported=total>1e-7;color[supported]=accum[supported]/total[supported,None]+fine_detail[supported]
    occluded=(total<.00001)&(estimated_total>1e-7)
    color[occluded]=estimated_color[occluded]/estimated_total[occluded,None]
    rear_path=folder/'rear-prediction/rear.png'
    inferred=~observed & (best<.08)
    rear=rear_reference(rear_path) if head_capture and rear_path.exists() else None
    if head_capture:
        # Use real photographed hair for missing-crown material, even when no
        # generated rear reference exists. This is texture synthesis, not an
        # additional measured viewpoint.
        frame=frames[front.name];photo=Image.open(folder/'images'/front.name).convert('RGBA');box=photo.getchannel('A').getbbox();swatch=None
        if frame.get('landmarks') and box:
            forehead=frame['landmarks'][10]['y']*photo.height;hair_h=forehead-box[1];mid=(box[0]+box[2])*.5;span=(box[2]-box[0])*.23
            if hair_h>12:
                patch=np.asarray(photo.crop((int(mid-span),int(box[1]+hair_h*.08),int(mid+span),int(box[1]+hair_h*.70))))
                rgb=patch[:,:,:3]/255.;valid=(patch[:,:,3]>200)&(rgb.mean(2)<.45)
                if valid.any():
                    _,near=distance_transform_edt(~valid,return_indices=True);rgb[~valid]=rgb[near[0][~valid],near[1][~valid]];swatch=rgb
        inferred_rgb,scalp=missing_head_material(texel,tn,p,completion,reference_color,rear,swatch)
        # Discard grazing-angle projections smoothly: they carry little usable
        # texture resolution and otherwise smear the side and crown.
        weight=np.clip((.12-best)/.12,0,1)*(~observed)
        weight=weight*weight*(3-2*weight)
        color=color*(1-weight[:,None])+inferred_rgb*weight[:,None]
    # A complete template contains hidden mouth surfaces and the backs of the
    # eyeballs. Those are not failed facial photographs. Measure coverage on
    # the exposed surface and give hidden internal anatomy a neutral material.
    exposed=observed&visible_any&(parts==0);interior=observed&~visible_any&(parts==0)
    color[interior]=reference_color*.65
    mouth=interior&(texel[:,1]<p[13,1]+.006)&(texel[:,1]>p[152,1]+.01)&(np.abs(texel[:,0])<.045)
    color[mouth]=np.array([.14,.055,.045])
    missing=exposed&(total<.00001)&~occluded;low=float(np.mean(missing[exposed]))
    if low>.12:np.savez_compressed(folder/'texture-diagnostic.npz',texel=texel,observed=observed,missing=missing,total=total)
    if low>.12:raise ValueError(f'Photographic coverage is insufficient for {low:.0%} of the face surface.')
    if np.any(missing):
        supported=exposed&~missing;_,closest=cKDTree(texel[supported]).query(texel[missing]);color[missing]=color[supported][closest]
    eye_texels=0
    if eyes:color,eye_texels=apply_eye_material(folder,texel,parts,p,eye_labels,color,eyes)
    estimated_eye=(parts>0) if eyes else np.zeros(len(texel),bool)
    texture=np.zeros((size,size,3),np.uint8);texture[covered]=np.uint8(np.clip(color,0,1)*255);_,nearest=distance_transform_edt(~covered,return_indices=True);texture[~covered]=texture[nearest[0][~covered],nearest[1][~covered]]
    Image.fromarray(texture[::-1]).save(folder/'appearance.png')
    if eyes:
        rough=np.full((size,size),199,np.uint8);rough[covered]=np.where(parts>0,51,199)
        rough[~covered]=rough[nearest[0][~covered],nearest[1][~covered]]
        Image.fromarray(rough[::-1]).convert('RGB').save(folder/'appearance-roughness.png')
    # An explicit audit prevents detection alone being reported as successful
    # cleanup. Retain which views were masked, propagated or excluded.
    atomic(folder/'eyewear-mask-audit.json',{'views':mask_audit,'estimatedFaceTexels':int(np.count_nonzero(occluded&exposed)),'estimatedEyeTexels':int(estimated_eye.sum()),'lensInteriorsExcluded':False,'opaqueFramePixelsReplaced':True,'eyeDetail':eyes['summary'] if eyes else 'Unverified photographic projection','limitation':'Skin through clear lenses is retained, so lens tint/reflections may remain. Eyeballs use the separately audited eye material; a glasses-free reference is needed for verified clean hidden skin.'})
    metadata={'mapping':mapping.tolist(),'indices':indices.ravel().tolist(),'uv':uv.ravel().tolist(),'texture':f'/api/face-asset?id={folder.name}&asset=appearance.png','stats':{'textureSize':size,'sourceViews':len(selected),'lowConfidenceFraction':low,'hairTextureFromPhotos':head_capture,'rearAppearance':('Captured rear photographs with inferred gaps' if any(abs(yaw(im))>115 for im in train) else ('AI-predicted rear reference' if rear_path.exists() else ('Photographic material continuation' if head_capture else 'Unobserved gray'))),'photographedCapFraction':float(np.mean(best[~observed]>.001)),'material':'lit','astraCompletion':bool(completion),'eyewearRemovedFromSkin':bool(completion and completion['glasses']['present']),'crownMapping':'Cartesian triplanar photo swatch; no spherical pole','method':'Photographic face and hair with frontal feature ownership. Narrow opaque rim cleanup preserves observed eyes and brows; separate glasses retain their own material. Missing rear appearance is labeled estimated.'}}
    metadata['stats'].update(material='lit',eyewearRemovedFromSkin=False,opaqueFrameCleanupApplied=any(a['excludedPixels']>0 for a in mask_audit),eyewearMaskedViews=sum(a['excludedPixels']>0 for a in mask_audit),eyewearSkippedViews=sum(a['method']=='unverified-view-excluded' for a in mask_audit),estimatedOccludedFaceFraction=float(np.mean(occluded[exposed])),lensInteriorsExcluded=False,skinDetail='3072px photographic color; observed eyes, eyebrows and hair preserved')
    metadata['stats']['eyes']=eyes['summary'] if eyes else 'Legacy photographic projection; eye detail has not been quality checked.'
    metadata['stats']['eyeMaterialTexels']=eye_texels
    metadata['stats']['nativeVideoDetailViews']=len(detail_audit['frames'])
    metadata['stats']['fineDetail']='Native video pixels; a single visible camera owns fine eyebrow, beard and hair texture; only broad color is blended.'
    metadata['stats']['material']='photo'
    metadata['stats']['colour']='Captured browser-decoded video colour and lighting retained; no additional studio relighting at rest.'
    metadata['textureSha256']=hashlib.sha256((folder/'appearance.png').read_bytes()).hexdigest()
    metadata['positionsSha256']=hashlib.sha256(np.asarray(p,dtype='<f4').tobytes()).hexdigest()
    if eyes:metadata['roughnessTexture']=f'/api/face-asset?id={folder.name}&asset=appearance-roughness.png'
    metadata['stats']['skinDetail']='3072px photographic skin, eyebrows and hair; eyeballs receive separate quality-checked iris material.' if eyes else metadata['stats']['skinDetail']
    metadata['stats']['method']='Photographic skin and hair with separate eye materials and 3D glasses. Missing eye detail is explicitly labeled estimated.' if eyes else metadata['stats']['method']
    atomic(folder/'texture-atlas.json',metadata);return metadata['stats']
