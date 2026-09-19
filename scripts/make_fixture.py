"""Known-surface fixture, derived from Lee Perry-Smith's CC BY 3.0 head scan.

Mesh -> sampled surface Gaussians -> independent Poisson mesh. This tests the
converter, not image-to-Gaussian training or arbitrary raw radiance fields.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
from reconstruction import reconstruct

root = Path(__file__).resolve().parents[1]
mesh = trimesh.load(root / "public/reference/LeePerrySmith.glb", force="mesh", process=False)
np.random.seed(17)
points, ids = trimesh.sample.sample_surface(mesh, 100000)
# Remove the shoulder base so the target is a head at human scale.
floor = mesh.bounds[0,1] + (mesh.bounds[1,1]-mesh.bounds[0,1]) * .25
keep = points[:,1] > floor
points, ids = points[keep], ids[keep]
triangles = mesh.triangles[ids]
bary = trimesh.triangles.points_to_barycentric(triangles, points)
uv = (mesh.visual.uv[mesh.faces[ids]] * bary[:,:,None]).sum(axis=1)
texture = np.asarray(Image.open(root / "public/reference/Map-COL.jpg").convert("RGB"))
# glTF texture coordinate convention. The original map uses OBJ's V direction.
px = np.clip((uv[:,0] * (texture.shape[1]-1)).astype(int), 0, texture.shape[1]-1)
py = np.clip((uv[:,1] * (texture.shape[0]-1)).astype(int), 0, texture.shape[0]-1)
colors = texture[py, px] / 255.
normals = mesh.face_normals[ids]
axis = np.tile([0.,0.,1.], (len(points),1))
cross = np.cross(axis, normals)
quat = np.column_stack((cross, 1 + normals[:,2]))
opposite = np.linalg.norm(quat,axis=1) < 1e-6
quat[opposite] = [1,0,0,0]
quat /= np.linalg.norm(quat,axis=1)[:,None]
size = np.max(np.ptp(points,axis=0)) * .004
names = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
values = np.column_stack((points,normals,(colors-.5)/.28209479177387814,np.full(len(points),4.),
                          np.tile(np.log([size,size,size*.07]),(len(points),1)),quat[:,[3,0,1,2]]))
data = np.empty(len(points),dtype=[(n,"f4") for n in names])
for i,n in enumerate(names): data[n] = values[:,i]
out = root / "public/reference/face-surface-fixture.ply"
PlyData([PlyElement.describe(data,"vertex")],text=False).write(out)
result = reconstruct(out.read_bytes(), depth=8)
result["stats"]["fixture"] = "Synthetic surface Gaussians sampled from the CC BY 3.0 Lee Perry-Smith scan. Not a trained photo splat."
(root / "public/reference/face-poisson.json").write_text(json.dumps(result))
print(json.dumps(result["stats"],indent=2))
