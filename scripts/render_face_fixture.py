"""Generate deterministic perspective views of a public mesh for pipeline QA.
These are synthetic test views, never a personal scan or validation of likeness.
"""

from pathlib import Path
import json, sys
import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/generated/face-fixture'
OUT.mkdir(parents=True, exist_ok=True)
mesh = trimesh.load(ROOT / 'public/reference/LeePerrySmith.glb', force='mesh')
v = np.asarray(mesh.vertices)
faces = np.asarray(mesh.faces)
uv = np.asarray(mesh.visual.uv)
# The same normalized reference orientation used by the existing test fixture.
v = v - (v.min(axis=0) + v.max(axis=0)) / 2
v *= 0.28 / np.ptp(v[:, 1])
texture = np.array(Image.open(ROOT / 'public/reference/Map-COL.jpg').convert('RGB'))
H, W = texture.shape[:2]
size = 768
focal = 1250.0
frames = []
for num, yaw in enumerate(np.linspace(-50, 50, 41)):
    a = np.radians(yaw)
    toward = np.array([np.sin(a), 0, np.cos(a)])
    right = np.cross([0, 1, 0], toward)
    B = np.stack([right, [0, 1, 0], -toward])
    cam = toward * 0.65
    q = (v - cam) @ B.T
    z = q[:, 2]
    xy = q[:, :2] / z[:, None] * focal
    xy[:, 1] *= -1
    xy += size / 2
    rgb = np.zeros((size, size, 4), np.uint8)
    depth = np.full((size, size), np.inf)
    for ids in faces:
        t = xy[ids]
        zz = z[ids]
        if min(zz) <= 0:
            continue
        lo = np.maximum(np.floor(t.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(t.max(axis=0)).astype(int), size - 1)
        if np.any(lo > hi):
            continue
        xx, yy = np.meshgrid(
            np.arange(lo[0], hi[0] + 1) + 0.5, np.arange(lo[1], hi[1] + 1) + 0.5
        )
        den = (t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0]) + (t[2, 0] - t[1, 0]) * (
            t[0, 1] - t[2, 1]
        )
        if abs(den) < 1e-8:
            continue
        aa = (
            (t[1, 1] - t[2, 1]) * (xx - t[2, 0]) + (t[2, 0] - t[1, 0]) * (yy - t[2, 1])
        ) / den
        bb = (
            (t[2, 1] - t[0, 1]) * (xx - t[2, 0]) + (t[0, 0] - t[2, 0]) * (yy - t[2, 1])
        ) / den
        cc = 1 - aa - bb
        weights = np.stack([aa, bb, cc], axis=-1) / zz
        iz = weights.sum(axis=-1)
        d = 1 / np.maximum(iz, 1e-8)
        region = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        visible = (aa >= 0) & (bb >= 0) & (cc >= 0) & (d < depth[region])
        uvs = (weights @ uv[ids]) / np.maximum(iz[:, :, None], 1e-8)
        tx = np.clip((uvs[:, :, 0] * W).astype(int), 0, W - 1)
        ty = np.clip((uvs[:, :, 1] * H).astype(int), 0, H - 1)
        rgb[region][visible, :3] = texture[ty[visible], tx[visible]]
        rgb[region][visible, 3] = 255
        depth[region][visible] = d[visible]
    name = f'{num:03d}.png'
    Image.fromarray(rgb).save(OUT / name)
    frames.append({'filename': name, 'yaw': float(yaw)})
    if num % 10 == 0:
        print(f'Rendered {num+1}/41', flush=True)
(OUT / 'source.json').write_text(
    json.dumps(
        {
            'source': 'Lee Perry-Smith, CC BY 3.0',
            'synthetic': True,
            'frames': frames,
            'focal': focal,
        }
    )
)
