"""Decode the official LAM_WebRender p2-1 demo to a world-coordinate PLY.

LAM renderer adds offset.ply xyz to skin.glb POSITION in vertex order.
No mesh sampling, new Gaussian training, or generated identity is performed.
"""
import sys,json,struct,hashlib
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from plyfile import PlyData,PlyElement
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reconstruction import reconstruct

root=Path(__file__).resolve().parents[1];source=root/'.local/sources/lam/p2-1';out=root/'public/online-face';out.mkdir(exist_ok=True)
b=(source/'skin.glb').read_bytes();n=struct.unpack_from('<I',b,12)[0];g=json.loads(b[20:20+n]);binary=b[28+n:]
def accessor(index):
    a=g['accessors'][index];v=g['bufferViews'][a['bufferView']]
    dtype={5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']];width={'SCALAR':1,'VEC3':3,'VEC4':4}[a['type']]
    start=v.get('byteOffset',0)+a.get('byteOffset',0)
    return np.frombuffer(binary,dtype,count=a['count']*width,offset=start).reshape(-1,width).copy()
p=g['meshes'][0]['primitives'][0];base=accessor(p['attributes']['POSITION'])
ply=PlyData.read(source/'offset.ply');v=ply['vertex'].data.copy()
assert len(base)==len(v)==20018
for k,name in enumerate('xyz'):v[name]+=base[:,k]
world=out/'lam-head.ply';PlyData([PlyElement.describe(v,'vertex')],text=False).write(world)
meta={'source':'https://github.com/aigc3d/LAM_WebRender/tree/main/asset/arkit','archive':'p2-1.zip','archiveSha256':hashlib.sha256((source.parent/'p2-1.zip').read_bytes()).hexdigest(),'gaussians':len(v),'method':'Official LAM example; skin POSITION plus learned Gaussian offsets. Single-image inferred avatar, not a multiview scan of the user.','license':'LAM_WebRender repository MIT. LAM weights have separate terms; example identity provenance is not independently verified. Local research demonstration only.'}
(out/'source.json').write_text(json.dumps(meta,indent=2))
mesh=reconstruct(world.read_bytes(),depth=8)
mesh['stats']['source']=meta['source'];mesh['stats']['provenance']=meta['method']
(out/'poisson.json').write_text(json.dumps(mesh,separators=(',',':')))
print(json.dumps(mesh['stats'],indent=2))

# A radiance field does not guarantee a surface. Use the shipped registered
# topology as a strong prior, restricting learned offsets to normal motion.
faces=accessor(p['indices']).reshape(-1,3)
surface=trimesh.Trimesh(base,faces,process=False)
normals=np.array(surface.vertex_normals)
xyz=np.column_stack([v[k] for k in 'xyz'])
alpha=1/(1+np.exp(-np.clip(v['opacity'],-30,30)))
offset=xyz-base
normal_offset=np.einsum('ij,ij->i',offset,normals)
confidence=alpha*np.exp(-np.sum(offset**2,axis=1)/(.012**2))
displacement=np.clip(normal_offset,-.006,.006)*confidence
edges=surface.edges_unique;rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
degree=np.bincount(rows,minlength=len(base));adj=coo_matrix((1/np.maximum(degree[rows],1),(rows,cols)),shape=(len(base),len(base))).tocsr()
for _ in range(12):displacement=.55*displacement+.45*(adj@displacement)
fitted=base+normals*displacement[:,None]
valid=(alpha>.2)&(np.linalg.norm(offset,axis=1)<.04)
dist,ids=cKDTree(xyz[valid]).query(fitted,k=8)
colors=np.clip(.5+.28209479177387814*np.column_stack([v[f'f_dc_{k}'] for k in range(3)]),0,1)[valid]
weights=alpha[valid][ids]*np.exp(-(dist/.006)**2);weights+=1e-12;weights/=weights.sum(axis=1,keepdims=True)
colors=(colors[ids]*weights[:,:,None]).sum(axis=1)
lo,hi=np.quantile(fitted,[0,1],axis=0);center=(lo+hi)/2;scale=.28/(hi[1]-lo[1]);fitted=(fitted-center)*scale
result={'positions':fitted.ravel().tolist(),'indices':faces.ravel().tolist(),'colors':colors.ravel().tolist(),'transform':{'center':center.tolist(),'scale':float(scale)},'stats':{'sourceSplats':len(v),'vertices':len(base),'triangles':len(faces),'method':'Registered LAM topology, opacity-weighted Gaussian normal offsets, Laplacian smoothing, DC color transfer','source':meta['source'],'maxNormalFitMm':round(float(np.max(np.abs(displacement)))*1000,2),'limitation':'Strong template prior, not an implementation of SplatFace or the Fedkiw paper. Hair and unseen geometry remain approximate. Raw Poisson failed visual inspection and is retained separately.'}}
(out/'mesh.json').write_text(json.dumps(result,separators=(',',':')))
