"""Dense photographic depth: Poisson-disk surface samples -> multi-view
photo-consistency -> screened Poisson surface -> conformed rig topology.

The landmark mesh measures 468 points. Everything between them is subdivision
smoothing, not the subject. This stage measures the surface between landmarks
from the same photographs and recovered cameras. No Gaussian/radiance model.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from scipy.ndimage import binary_dilation,gaussian_filter
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve

# MediaPipe indices spanning brows, under-eye skin and temples: the region a
# pair of glasses can cover.
EYEWEAR=[70,63,105,66,107,336,296,334,293,300,117,118,119,120,121,346,347,348,349,350,127,356,234,454,162,389]

def bandpass(green,alpha,fine=.9,coarse=3.5):
    """Fine albedo texture only. The head turns beneath fixed lights, so broad
    shading is not view-consistent; matching it produced flat, ambiguous curves.
    Normalized convolution keeps the erased background from ringing."""
    w=alpha.astype(np.float32);lo=gaussian_filter(green*w,coarse)/np.maximum(gaussian_filter(w,coarse),1e-3);hi=gaussian_filter(green*w,fine)/np.maximum(gaussian_filter(w,fine),1e-3)
    return ((hi-lo)/np.maximum(lo,.05)).astype(np.float32)

def eyewear_mask(landmarks,luminance,alpha):
    """Dark pixels inside the eyewear region. Frames are a separate object up
    to 15 mm in front of the skin; brows and lashes are excluded with them."""
    h,w=alpha.shape;hull=np.array([[landmarks[i]['x']*w,landmarks[i]['y']*h] for i in EYEWEAR]);centre=hull.mean(0);hull=centre+(hull-centre)*1.12
    from scipy.spatial import ConvexHull
    region=Image.new('L',(w,h),0);ImageDraw.Draw(region).polygon([tuple(v) for v in hull[ConvexHull(hull).vertices]],fill=1);region=np.asarray(region,bool)
    skin=np.median(luminance[alpha&~region]) if np.any(alpha&~region) else .5
    return binary_dilation(region&alpha&(luminance<.55*skin),iterations=2)

def wears_glasses(frames,folder):
    """A dark bridge across the nose between the eyes in most frontal frames.
    Clear or thin metal frames are not detected; pass --glasses on for those."""
    votes=[]
    for frame in sorted(frames,key=lambda f:abs(f['yaw']))[:9]:
        px=np.asarray(Image.open(folder/'images'/frame['filename']).convert('RGB'),np.float32)/255.;lum=px@np.array([.299,.587,.114],np.float32);h,w=lum.shape;L=frame['landmarks']
        at=lambda i,r:lum[max(int(L[i]['y']*h)-r,0):int(L[i]['y']*h)+r+1,max(int(L[i]['x']*w)-r,0):int(L[i]['x']*w)+r+1]
        r=max(2,int(abs(L[133]['x']-L[362]['x'])*w*.12));bridge=min(at(i,r).min() for i in (168,6,197));cheek=np.median(np.r_[at(50,r*2).ravel(),at(280,r*2).ravel()]);votes.append(bridge<.45*cheek)
    return bool(np.mean(votes)>.5)


class Views:
    """Registered photographs as one luminance stack with their cameras."""
    def __init__(self,folder,cameras,names,frames=None,gray=None,alpha=None):
        # Camera arrays come from photo_dense_export.export_cameras. pycolmap and
        # Open3D each bundle an OpenMP runtime and cannot share a process.
        keep=[i for i,name in enumerate(cameras['names']) if str(name) in names];self.names=[str(cameras['names'][i]) for i in keep]
        params=cameras['params'];self.f,self.cx,self.cy=params[:3];self.k=params[3] if len(params)>3 else 0.;self.w,self.h=int(cameras['width']),int(cameras['height'])
        self.R=cameras['R'][keep];self.t=cameras['t'][keep];self.centers=cameras['centers'][keep]
        if gray is not None:self.gray,self.alpha=gray,alpha;return
        gray=[];alpha=[]
        for name in self.names:
            px=np.asarray(Image.open(folder/'images'/name).convert('RGBA'),np.float32)/255.;mask=px[...,3]>.78
            # frames carries 2D landmarks only when eyewear must be excluded.
            if frames is not None:mask&=~eyewear_mask(frames[name]['landmarks'],px[...,:3]@np.array([.299,.587,.114],np.float32),mask)
            gray.append(bandpass(px[...,1],mask));alpha.append(mask)
        self.gray=np.stack(gray);self.alpha=np.stack(alpha)
    def subset(self,ids):
        other=object.__new__(Views);other.__dict__.update(self.__dict__);other.names=[self.names[i] for i in ids]
        for key in ('R','t','centers','gray','alpha'):setattr(other,key,getattr(self,key)[ids])
        return other
    def __len__(self):return len(self.names)
    def project(self,view,X):
        """World points [...,3] with per-point view ids [...] -> pixel arrays."""
        Xc=np.einsum('...ij,...j->...i',self.R[view],X)+self.t[view];z=Xc[...,2];ok=z>1e-6;z=np.where(ok,z,1.)
        x=Xc[...,0]/z;y=Xc[...,1]/z;d=1+self.k*(x*x+y*y)
        # COLMAP places the first pixel centre at (.5,.5).
        return self.f*d*x+self.cx-.5,self.f*d*y+self.cy-.5,ok
    def sample(self,view,u,v,ok):
        """Bilinear luminance and mask validity at continuous array coordinates."""
        x0=np.floor(u).astype(np.int32);y0=np.floor(v).astype(np.int32);ok=ok&(x0>=0)&(y0>=0)&(x0<self.w-1)&(y0<self.h-1)
        x0=np.clip(x0,0,self.w-2);y0=np.clip(y0,0,self.h-2);a=(u-x0).astype(np.float32);b=(v-y0).astype(np.float32);g=self.gray
        value=(g[view,y0,x0]*(1-a)+g[view,y0,x0+1]*a)*(1-b)+(g[view,y0+1,x0]*(1-a)+g[view,y0+1,x0+1]*a)*b
        return value,ok&self.alpha[view,y0,x0]&self.alpha[view,y0+1,x0+1]


def head_frame(landmarks_world):
    """The builder's normalized head frame, recomputed from saved landmarks."""
    points=np.asarray(landmarks_world);center=(points[10]+points[152])/2;right=points[263]-points[33];right/=np.linalg.norm(right)
    up=points[10]-points[152];up-=right*np.dot(up,right);up/=np.linalg.norm(up);B=np.stack([right,up,np.cross(right,up)])
    scale=.20/(((points-center)@B.T)[10,1]-((points-center)@B.T)[152,1])
    return center,B,float(scale)

def to_world(p,center,B,scale):return (p/scale)@B+center

def raycaster(p,f):
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(p,np.float32)),o3d.core.Tensor(np.ascontiguousarray(f,np.uint32)));return scene

def poisson_disk(p,f,count,seed=0):
    """Blue-noise samples with interpolated normals on the given triangles."""
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p.astype(float)),o3d.utility.Vector3iVector(f.astype(np.int32)));mesh.compute_vertex_normals()
    o3d.utility.random.seed(seed);pc=mesh.sample_points_poisson_disk(int(count),init_factor=5)
    return np.asarray(pc.points),np.asarray(pc.normals)

def choose_views(x,n,views,scene_world,count,min_facing=.34,separation=4.):
    """Per sample: unoccluded, well-facing views spread across the baseline."""
    toward=views.centers[None]-x[:,None];distance=np.linalg.norm(toward,axis=2);toward/=distance[...,None];facing=np.einsum('nvj,nj->nv',toward,n)
    origin=x[:,None]+toward*(distance[...,None]*.004);rays=np.concatenate([origin,toward],axis=2).reshape(-1,6).astype(np.float32)
    hit=scene_world.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy().reshape(facing.shape);usable=(facing>min_facing)&~(hit<distance)
    # The mask must contain the sample itself.
    for v in range(len(views)):
        px,py,ok=views.project(np.full(len(x),v),x);xi=np.clip(np.rint(px).astype(int),0,views.w-1);yi=np.clip(np.rint(py).astype(int),0,views.h-1);usable[:,v]&=ok&views.alpha[v,yi,xi]
    score=np.where(usable,facing,-1.);chosen=np.full((len(x),count),-1);limit=np.cos(np.radians(separation))
    for slot in range(count):
        best=np.argmax(score,axis=1);good=score[np.arange(len(x)),best]>0;chosen[good,slot]=best[good]
        # Near-duplicate frontal frames add noise, not baseline.
        similar=np.einsum('nvj,nj->nv',toward,toward[np.arange(len(x)),best])>limit;score[similar&good[:,None]]=-1.
    return chosen

def sweep(x,n,chosen,views,offsets,half,spacing,chunk=1500,floor=.004):
    """Photo-consistency of a tangent-plane patch at each normal offset.

    Returns the robust mean ZNCC against the reference view [N,D] and the number
    of source views that contributed."""
    helper=np.where(np.abs(n[:,1:2])<.9,[[0.,1.,0.]],[[1.,0.,0.]]);a=np.cross(helper,n);a/=np.linalg.norm(a,axis=1,keepdims=True);b=np.cross(n,a)
    grid=np.arange(-half,half+1)*spacing;gu,gv=np.meshgrid(grid,grid);gu=gu.ravel();gv=gv.ravel();P=len(gu);D=len(offsets);K=chosen.shape[1]
    result=np.full((len(x),D),np.nan,np.float32);support=np.zeros((len(x),D),np.int16)
    for start in range(0,len(x),chunk):
        s=slice(start,start+chunk);X=(x[s,None,None]+n[s,None,None]*offsets[None,:,None,None]+a[s,None,None]*gu[None,None,:,None]+b[s,None,None]*gv[None,None,:,None])
        patches=[];valid=[]
        for slot in range(K):
            view=np.maximum(chosen[s,slot],0)[:,None,None]*np.ones((1,D,P),int);u,v,ok=views.project(view,X);value,ok=views.sample(view,u,v,ok)
            good=(ok.mean(axis=2)>.92)&(chosen[s,slot]>=0)[:,None];value=value-value.mean(axis=2,keepdims=True);norm=np.linalg.norm(value,axis=2)
            # A flat patch has no depth signal; treat it as missing, not as a match.
            good&=norm>floor*np.sqrt(P);patches.append(value/np.maximum(norm,1e-9)[...,None]);valid.append(good)
        ncc=np.stack([np.einsum('ndp,ndp->nd',patches[0],q) for q in patches[1:]],axis=2);ok=np.stack(valid[1:],axis=2)&valid[0][...,None]
        ncc=np.where(ok,ncc,-np.inf);order=-np.sort(-ncc,axis=2);used=ok.sum(axis=2);keep=np.maximum((used+1)//2,np.minimum(used,2))
        rank=np.arange(K-1)[None,None,:]<keep[...,None];total=np.where(rank&np.isfinite(order),order,0).sum(axis=2)
        result[s]=np.where(used>=2,total/np.maximum(keep,1),np.nan);support[s]=used
    return result,support

def pick(curves,offsets,minimum,margin=.04):
    """Sub-step peak, rejecting boundary maxima and indistinct curves."""
    filled=np.where(np.isfinite(curves),curves,-1.);best=np.argmax(filled,axis=1);rows=np.arange(len(curves));top=filled[rows,best];D=len(offsets)
    interior=(best>0)&(best<D-1);lo=np.clip(best-1,0,D-1);hi=np.clip(best+1,0,D-1);yl=filled[rows,lo];yh=filled[rows,hi];curve=yl-2*top+yh
    shift=np.where(curve<-1e-6,.5*(yl-yh)/np.minimum(curve,-1e-6),0.);step=offsets[1]-offsets[0];depth=offsets[best]+np.clip(shift,-.5,.5)*step
    # Distinctness: the peak must rise above the curve's typical level.
    typical=np.nanmedian(np.where(np.isfinite(curves),curves,np.nan),axis=1);accepted=interior&(top>minimum)&(top-np.nan_to_num(typical,nan=1.)>margin)
    return depth,top,accepted

def barycentric(p,f,x):
    """Closest face-region triangle and barycentric weights for each sample."""
    ans=raycaster(p,f).compute_closest_points(o3d.core.Tensor(x.astype(np.float32)));uv=ans['primitive_uvs'].numpy().astype(float)
    return ans['primitive_ids'].numpy().astype(int),np.column_stack([1-uv.sum(1),uv])

def vertex_normals(p,f):
    n=np.zeros_like(p);fn=np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])
    for k in range(3):np.add.at(n,f[:,k],fn)
    return n/np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)

def measure(x,n,scene_world,views,yaw,frame,reach=.010,step=.0005,groups=3,tolerance=.0012,half=4,spacing=.0008):
    """Normal offsets that independent view subsets agree on.

    One subset's best match is often wrong on denoised webcam skin. Each subset
    is interleaved across yaw so it spans the baseline; a sample is kept only
    when every subset finds a distinct interior peak at the same depth. Chance
    agreement is about (tolerance/reach)^(groups-1), near 1% by default."""
    center,B,scale=frame;unit=1/scale;xw=to_world(x,center,B,scale);nw=n@B;offsets=np.arange(-reach,reach+step/2,step)*unit
    order=np.argsort([yaw[name] for name in views.names]);depths=[];peaks=[];valid=[]
    for g in range(groups):
        part=views.subset(order[g::groups]);chosen=choose_views(xw,nw,part,scene_world,5,separation=6.);curves,_=sweep(xw,nw,chosen,part,offsets,half,spacing*unit)
        depth,top,ok=pick(curves,offsets,.5,margin=.05);depths.append(depth/unit);peaks.append(top);valid.append(ok)
    depths=np.stack(depths,1);valid=np.stack(valid,1);peaks=np.stack(peaks,1);spread=np.ptp(depths,axis=1)
    accepted=valid.all(1)&(spread<tolerance)
    return depths.mean(1),peaks.mean(1),accepted,{'samples':len(x),'anySubsetPeak':float(valid.any(1).mean()),'everySubsetPeak':float(valid.all(1).mean()),'consensus':float(accepted.mean()),'chanceConsensus':float((tolerance/reach)**(groups-1))}

def solve_field(p,f,tri,weights_bary,offset,confidence,length=.010,prior=.02,rounds=6):
    """Robust screened Poisson equation for the depth-correction field:
    (A'WA + lambda*L + eps*I) d = A'W o on the mesh. Measurements screen the
    solution, the Laplacian carries them across unmeasured skin, and a weak
    prior returns unsupported regions to the landmark surface. The region rim
    is pinned so the seam with the rest of the head does not move.

    Roughly one consensus sample in seven is a chance agreement. Reweighting by
    residual (Cauchy) lets neighbours outvote it instead of denting the face."""
    used=np.unique(f);local=np.full(len(p),-1);local[used]=np.arange(len(used));F=local[f];V=len(used)
    rows=np.repeat(np.arange(len(tri)),3);A=coo_matrix((weights_bary.ravel(),(rows,F[tri].ravel())),shape=(len(tri),V)).tocsr()
    every=np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1);edges,counts=np.unique(every,axis=0,return_counts=True);rim=np.zeros(V);rim[np.unique(edges[counts==1])]=1
    adjacency=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(V,V));adjacency=adjacency+adjacency.T;L=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    h=np.median(np.linalg.norm(p[used][edges[:,0]]-p[used][edges[:,1]],axis=1));per_vertex=max(confidence.sum()/V,1e-6);smooth=(length/h)**2*per_vertex*L+diags(prior*per_vertex+1e6*per_vertex*rim)
    weight=confidence.copy()
    for _ in range(rounds):
        solution=spsolve((A.T@diags(weight)@A+smooth).tocsc(),A.T@(weight*offset));residual=A@solution-offset
        sigma=max(1.4826*np.median(np.abs(residual)),.0004);weight=confidence/(1+(residual/(2*sigma))**2)
    d=np.zeros(len(p));d[used]=solution;return d,weight/np.maximum(confidence,1e-12)

def sparse_offsets(cameras,p,face,frame,landmarks,eyewear):
    """Bundle-adjusted COLMAP points as signed offsets from the face surface.
    They are the most reliable depth in a capture: multi-view verified, though
    only where SIFT finds texture (brows, nostrils, lips, stubble, moles)."""
    if 'points' not in cameras.files:return np.zeros(0,int),np.zeros((0,3)),np.zeros(0),np.zeros(0)
    center,B,scale=frame;good=(cameras['pointTrack']>=3)&(cameras['pointAngle']>=10)&(cameras['pointError']<1.5);q=((cameras['points'][good]-center)@B.T)*scale;track=cameras['pointTrack'][good]
    ans=raycaster(p,face).compute_closest_points(o3d.core.Tensor(q.astype(np.float32)));tri=ans['primitive_ids'].numpy().astype(int);uv=ans['primitive_uvs'].numpy().astype(float);bary=np.column_stack([1-uv.sum(1),uv]);closest=ans['points'].numpy()
    normal=np.cross(p[face[tri,1]]-p[face[tri,0]],p[face[tri,2]]-p[face[tri,0]]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    delta=q-closest;signed=np.einsum('ij,ij->i',delta,normal);lateral=np.linalg.norm(delta-signed[:,None]*normal,axis=1);keep=(np.abs(signed)<.012)&(lateral<.002)
    if eyewear:
        # Frames and lenses are always in front of the skin they cover.
        low=landmarks[[117,346],1].mean()-.004;high=landmarks[[105,334],1].mean()+.006;keep&=~((closest[:,1]>low)&(closest[:,1]<high)&(signed>.0025))
    return tri[keep],bary[keep],signed[keep],.35*np.minimum(track[keep],8)/8

def consistency(p,f_all,f_face,tri,bary,views,frame):
    """Mean photo-consistency of withheld views on a surface. No search: the
    score is evaluated at the surface itself, so a better surface scores higher."""
    center,B,scale=frame;normals=vertex_normals(p,f_all);x=(p[f_face[tri]]*bary[...,None]).sum(1);n=(normals[f_face[tri]]*bary[...,None]).sum(1);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
    xw=to_world(x,center,B,scale);nw=n@B;chosen=choose_views(xw,nw,views,raycaster(to_world(p,center,B,scale),f_all),4,separation=5.)
    score,_=sweep(xw,nw,chosen,views,np.zeros(1),4,.0008/scale);return score[:,0]

def refine(folder,cameras,out,samples=14000,glasses='auto',log=print):
    started=time.perf_counter();mesh=json.loads((folder/'mesh.json').read_text());validation=json.loads((folder/'surface-validation.json').read_text());capture=json.loads((folder/'capture.json').read_text())
    p=np.array(mesh['positions'],float).reshape(-1,3);f=np.array(mesh['indices']).reshape(-1,3);face=f[:mesh['stats']['observedFaceTriangles']];frame=head_frame(validation['landmarksWorld'])
    frames={fr['filename']:fr for fr in capture['frames']};yaw={k:v['yaw'] for k,v in frames.items()};landmarks=((np.asarray(validation['landmarksWorld'])-frame[0])@frame[1].T)*frame[2]
    eyewear=wears_glasses(capture['frames'],folder) if glasses=='auto' else glasses=='on';log(f'Eyewear exclusion: {eyewear}')
    registered={str(n) for n in cameras['names']};train=Views(folder,cameras,set(validation['trainingFrames'])&registered,frames if eyewear else None);held=Views(folder,cameras,set(validation['withheldFrames'])&registered,frames if eyewear else None)
    x,n=poisson_disk(p,face,samples);tri,bary=barycentric(p,face,x);scene=raycaster(to_world(p,*frame),f)
    offset,peak,accepted,evidence=measure(x,n,scene,train,yaw,frame);log(f"Photo-consistency consensus on {evidence['consensus']:.1%} of {samples} Poisson-disk samples.")
    stri,sbary,soffset,sconf=sparse_offsets(cameras,p,face,frame,landmarks,eyewear);fit=np.random.default_rng(7).random(len(stri))<.7;log(f'{len(stri)} verified sparse points on the face; {int((~fit).sum())} withheld for validation.')
    evidence.update(eyewearExcluded=eyewear,trainingViews=len(train),withheldViews=len(held),sparsePoints={'onFace':int(len(stri)),'fitted':int(fit.sum()),'withheld':int((~fit).sum())})
    T=np.r_[tri[accepted],stri[fit]];Bc=np.vstack([bary[accepted],sbary[fit]]);O=np.r_[offset[accepted],soffset[fit]];C=np.r_[np.clip(peak[accepted]-.5,.02,.5),sconf[fit]]
    if len(T)<80:
        evidence.update(applied=False,reason='Too few verified depth measurements to constrain a field.');return evidence,None
    d,trust=solve_field(p,face,T,Bc,O,C);refined=p+vertex_normals(p,f)*d[:,None];evidence['measurements']={'used':int(len(T)),'downweightedAsOutliers':int((trust<.25).sum())}
    # Two validators that took no part in the fit decide whether to apply it:
    # withheld photographs, and withheld bundle-adjusted points.
    before=consistency(p,f,face,tri,bary,held,frame);after=consistency(refined,f,face,tri,bary,held,frame);moved=np.abs((d[face[tri]]*bary).sum(1))>.0005;both=np.isfinite(before)&np.isfinite(after)&moved
    gain=float(np.mean(after[both]-before[both])) if both.any() else 0.;improved=float(np.mean(after[both]>before[both])) if both.any() else 0.
    miss_before=np.abs(soffset[~fit]);miss_after=np.abs(soffset[~fit]-(d[face[stri[~fit]]]*sbary[~fit]).sum(1));enough=len(miss_before)>=40
    evidence.update(fieldMm={'rms':float(np.sqrt(np.mean(d[np.unique(face)]**2))*1000),'max':float(np.abs(d).max()*1000)},withheldPhotographs={'samplesCompared':int(both.sum()),'meanZnccBefore':float(np.mean(before[both])) if both.any() else None,'meanZnccAfter':float(np.mean(after[both])) if both.any() else None,'meanGain':gain,'fractionImproved':improved},
        withheldSparsePoints={'count':int(len(miss_before)),'medianMmBefore':float(np.median(miss_before)*1000),'medianMmAfter':float(np.median(miss_after)*1000),'fractionCloser':float(np.mean(miss_after<miss_before))} if enough else None,seconds=round(time.perf_counter()-started,1))
    photographs_agree=both.sum()>=200 and gain>.01 and improved>.55;points_agree=(not enough) or np.median(miss_after)<np.median(miss_before)
    evidence['applied']=bool(photographs_agree and points_agree)
    if not evidence['applied']:evidence['reason']='Withheld data does not prefer the refined surface; the landmark surface is kept.'
    np.savez(out/'dense-depth.npz',displacement=d,samples=x,offset=offset,accepted=accepted,peak=peak)
    return evidence,refined if evidence['applied'] else None

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--cameras',type=Path,required=True);parser.add_argument('--out',type=Path);parser.add_argument('--samples',type=int,default=14000);parser.add_argument('--glasses',choices=['auto','on','off'],default='auto');args=parser.parse_args()
    out=args.out or args.folder;out.mkdir(parents=True,exist_ok=True);evidence,refined=refine(args.folder.resolve(),np.load(args.cameras),out,args.samples,args.glasses)
    (out/'dense-depth.json').write_text(json.dumps(evidence,indent=2,allow_nan=False));print(json.dumps(evidence,indent=2))
    if refined is not None:
        # Same schema and vertex order as mesh.json, written beside it rather
        # than over it: the builder decides when to adopt the refined surface.
        # Vertices no triangle references (a separate physics cage) are left
        # in place and must be re-projected by whoever adopts this mesh.
        mesh=json.loads((args.folder/'mesh.json').read_text());f=np.array(mesh['indices']).reshape(-1,3)
        mesh['positions']=refined.astype(np.float32).ravel().tolist();mesh['normals']=vertex_normals(refined,f).astype(np.float32).ravel().tolist();mesh['stats']['denseDepth']=evidence
        (out/'mesh-dense.json').write_text(json.dumps(mesh,allow_nan=False));print('Wrote',out/'mesh-dense.json')
