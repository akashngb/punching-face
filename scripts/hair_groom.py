"""Photo-classified, editable strand groom on the measured hair envelope.

Only roots are inferred here. Browser fibers are deterministic geometry, with
triangle/barycentric binding to preserve edits, deformation and session saves.
"""
import numpy as np
import cv2
from PIL import Image


def scalp_weight(x, p, hair):
    height=p[10,1]-p[152,1]
    azimuth=np.abs(np.arctan2(x[:,0],x[:,2]+.09))
    threshold=np.interp(azimuth,[0,.6,1.45,2.15,np.pi],
        [p[10,1]+.002,p[10,1]-.012,p[10,1]-height*hair.get('sideHairlineFraction',.3),
         p[10,1]-height*hair.get('rearHairlineFraction',.85),p[10,1]-height*hair.get('rearHairlineFraction',.85)])
    w=np.clip((x[:,1]-threshold)/.008,0,1)
    return w*w*(3-2*w)


def build_hair_groom(p, faces, spec, rec=None, center=None, B=None, transform=None,folder=None):
    hair=(spec or {}).get('hair',{})
    if not hair.get('present') or hair.get('type')=='bald' or hair.get('density',.8)<=0:return None
    tri=p[faces];centroids=tri.mean(1)
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)*.5
    normals=cross/np.maximum(2*area[:,None],1e-12)
    weight=scalp_weight(centroids,p,hair)
    # Internal anatomy and the neck cut never grow hair.
    radial=centroids-np.array([0,.025,-.09])
    weight*=np.sum(normals*radial,axis=1)>0
    if rec is not None:
        world=(centroids/transform['scale'])@B+center
        for view in spec.get('views',[]):
            regions=view.get('hairRegions',[])
            if not regions:continue
            im=next((im for im in rec.images.values() if im.name==view['filename']),None)
            if im is None:continue
            cam=rec.cameras[im.camera_id];pose=im.cam_from_world()
            xy=cam.img_from_cam(world@pose.rotation.matrix().T+pose.translation)
            crop=np.asarray(spec['crops'][im.name]);mask=np.zeros((cam.height,cam.width),np.uint8)
            for polygon in regions:
                if len(polygon)>=3:
                    poly=np.rint(np.asarray(polygon)*(crop[2:]-crop[:2])+crop[:2]).astype(np.int32)
                    cv2.fillPoly(mask,[poly],255)
            ij=np.rint(np.nan_to_num(xy,nan=-1,posinf=-1,neginf=-1)).astype(int)
            inside=(ij[:,0]>=0)&(ij[:,0]<cam.width)&(ij[:,1]>=0)&(ij[:,1]<cam.height)
            toward=im.projection_center()-world;toward/=np.maximum(np.linalg.norm(toward,axis=1)[:,None],1e-9)
            facing=np.sum((normals@B)*toward,axis=1)
            visible=inside&(facing>.55)
            # Apply only facing evidence: a front polygon cannot erase rear hair.
            hit=mask[np.clip(ij[:,1],0,cam.height-1),np.clip(ij[:,0],0,cam.width-1)]>0
            weight[visible&~hit]=0
    probability=area*weight
    if probability.sum()<=1e-9:return None
    count=int(2500+hair.get('density',.8)*8500)
    rng=np.random.default_rng(20260919)
    chosen=rng.choice(len(faces),count,p=probability/probability.sum())
    a=np.sqrt(rng.random(count));b=rng.random(count)
    bary=np.c_[1-a,a*(1-b),a*b]
    guide=None
    if folder is not None and rec is not None:
        roots=np.einsum('ij,ijk->ik',bary,tri[chosen]);rn=normals[chosen]
        world=roots/transform['scale']@B+center
        best=np.zeros(count);direction=np.zeros((count,3));colors=np.tile(hair['colorSrgb'],(count,1));coherence=np.zeros(count)
        for view in spec.get('views',[]):
            im=next((im for im in rec.images.values() if im.name==view['filename']),None)
            if im is None:continue
            px=np.asarray(Image.open(folder/'images'/im.name).convert('RGBA'));h,w=px.shape[:2]
            gray=cv2.cvtColor(px[:,:,:3],cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
            gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
            xx=cv2.GaussianBlur(gx*gx,(0,0),2);yy=cv2.GaussianBlur(gy*gy,(0,0),2);cross=cv2.GaussianBlur(gx*gy,(0,0),2)
            angle=.5*np.arctan2(2*cross,xx-yy)+np.pi/2
            certainty=np.sqrt((xx-yy)**2+4*cross**2)/np.maximum(xx+yy,1e-8)
            pose=im.cam_from_world();cam=rec.cameras[im.camera_id];cp=world@pose.rotation.matrix().T+pose.translation
            xy=cam.img_from_cam(cp);ij=np.rint(np.nan_to_num(xy,nan=-1,posinf=-1,neginf=-1)).astype(int)
            inside=(ij[:,0]>=1)&(ij[:,0]<w-1)&(ij[:,1]>=1)&(ij[:,1]<h-1)&(cp[:,2]>0)
            x=np.clip(ij[:,0],0,w-1);y=np.clip(ij[:,1],0,h-1)
            toward=im.projection_center()-world;toward/=np.maximum(np.linalg.norm(toward,axis=1)[:,None],1e-9)
            facing=np.maximum(np.sum((rn@B)*toward,axis=1),0)
            score=facing**4*inside*(px[y,x,3]>220)*certainty[y,x]
            keep=(score>best)&(score>.08)
            if not keep.any():continue
            theta=angle[y[keep],x[keep]];uv=xy[keep];delta=np.c_[np.cos(theta),np.sin(theta)]
            rays=cam.cam_from_img(uv+delta)-cam.cam_from_img(uv)
            tangent=np.c_[rays,np.zeros(len(rays))]@pose.rotation.matrix()@B.T
            tangent-=rn[keep]*np.sum(tangent*rn[keep],axis=1)[:,None]
            tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-9)
            direction[keep]=tangent;colors[keep]=px[y[keep],x[keep],:3]/255.;best[keep]=score[keep];coherence[keep]=certainty[y[keep],x[keep]]
        # Unseen roots receive no invented groom. Photographic continuation
        # handles those regions until another viewpoint supplies evidence.
        guide={'directions':direction.astype(np.float32).ravel().tolist(),'colors':colors.astype(np.float32).ravel().tolist(),'confidence':best.astype(np.float32).tolist()}
    return {'version':2,'type':'strand-hair','mode':'photo-detail' if guide else 'editable-prior','photoGuides':guide,'source':'Captured local hair flow and color; Astra classifies style without replacing its silhouette',
        'estimated':True,'parameters':hair,'seed':20260919,'rootTriangles':faces[chosen].ravel().tolist(),
        'rootWeights':bary.astype(np.float32).ravel().tolist(),'sourceVertexCount':len(p),
        'hairlineY':float(p[10,1]),'rootCount':count,'binding':'Barycentric scalp roots; rigid skull motion and local surface edits',
        'limitation':'Strand detail is a hairstyle-conditioned model, not measured individual hairs. Hidden hair and skin remain estimates.'}
