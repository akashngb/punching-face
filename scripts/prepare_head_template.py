"""Extract the CC0 MakeHuman head and neck, retaining its facial edge loops."""
from pathlib import Path
import numpy as np,json,trimesh
ROOT=Path(__file__).resolve().parents[1]
def load():
 p=[];quads=[];group=''
 for line in (ROOT/'public/head-template/base.obj').read_text().splitlines():
  t=line.split()
  if not t:continue
  if t[0]=='v':p.append(list(map(float,t[1:4])))
  elif t[0]=='g':group=t[1]
  elif t[0]=='f' and group=='body':quads.append([int(x.split('/')[0])-1 for x in t[1:]])
 p=np.array(p);quads=np.array(quads);quads=quads[np.min(p[quads,1],axis=1)>5.8]
 ids=np.unique(quads);remap=np.full(len(p),-1);remap[ids]=np.arange(len(ids));q=remap[quads];v=p[ids]
 # MakeHuman's unit is a decimeter. Canonical working space is metres.
 v=(v-np.array([0,7.1,1.35]))*.1
 f=np.concatenate([q[:,[0,1,2]],q[:,[0,2,3]]]);m=trimesh.Trimesh(v,f,process=False);m.fix_normals()
 return np.array(m.vertices),np.array(m.faces),q,ids
if __name__=='__main__':
 p,f,q,ids=load();d={'positions':p.ravel().tolist(),'indices':f.ravel().tolist(),'sourceIds':ids.tolist(),'quads':q.tolist()};(ROOT/'public/head-template/head.json').write_text(json.dumps(d));print(len(p),len(f),p.min(0),p.max(0))
