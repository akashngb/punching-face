"""Bake a static multi-view Gaussian appearance onto the registered face UVs.

CPU orthographic 3DGS projection, alpha compositing, xatlas parameterization,
normal/opacity/depth-weighted view selection. Captured radiance, NOT de-lit albedo.
"""

from pathlib import Path
import json, time, argparse
import numpy as np
import trimesh, xatlas
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/online-face'
CACHE = ROOT / '.local/texture-bake'
CACHE.mkdir(parents=True, exist_ok=True)
SIZE = 2048
VIEW = 768
SPAN = 0.34
PLY_NAME = 'lam-head.ply'
TEXTURE_URL = '/online-face/appearance.png'


def basis(yaw, pitch):
    y, p = np.radians([yaw, pitch])
    toward = np.array([np.sin(y) * np.cos(p), np.sin(p), np.cos(y) * np.cos(p)])
    right = np.cross([0, 1, 0], toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    return np.stack([right, up, toward])


def render(xyz, cov, color, opacity, B):
    points = xyz @ B.T
    pixels = points[:, :2] * (VIEW / SPAN) + VIEW / 2
    C = np.einsum('ai,nij,bj->nab', B[:2], cov, B[:2]) * (VIEW / SPAN) ** 2
    C[:, 0, 0] += 0.3
    C[:, 1, 1] += 0.3
    rgb = np.zeros((VIEW, VIEW, 3), np.float32)
    alpha = np.zeros((VIEW, VIEW), np.float32)
    depth = np.zeros((VIEW, VIEW), np.float32)
    for i in np.argsort(-points[:, 2]):
        if opacity[i] < 0.004:
            continue
        r = np.minimum(np.ceil(3 * np.sqrt(np.diag(C[i]))), 120).astype(int)
        x0, y0 = np.maximum(np.floor(pixels[i] - r).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(pixels[i] + r).astype(int) + 1, VIEW)
        if x1 <= x0 or y1 <= y0:
            continue
        xx, yy = np.meshgrid(
            np.arange(x0, x1) + 0.5 - pixels[i, 0],
            np.arange(y0, y1) + 0.5 - pixels[i, 1],
        )
        inv = np.linalg.inv(C[i])
        power = inv[0, 0] * xx * xx + 2 * inv[0, 1] * xx * yy + inv[1, 1] * yy * yy
        a = np.minimum(0.99, opacity[i] * np.exp(-0.5 * power))
        a[power > 9] = 0
        weight = a * (1 - alpha[y0:y1, x0:x1])
        rgb[y0:y1, x0:x1] += weight[:, :, None] * color[i]
        depth[y0:y1, x0:x1] += weight * points[i, 2]
        alpha[y0:y1, x0:x1] += weight
    rgb /= np.maximum(alpha[:, :, None], 1e-7)
    depth /= np.maximum(alpha, 1e-7)
    return rgb, alpha, depth


def texels(positions, normals, indices, mapping, uv):
    world = np.zeros((SIZE, SIZE, 3), np.float32)
    normal = np.zeros_like(world)
    covered = np.zeros((SIZE, SIZE), bool)
    for triangle in indices:
        t = uv[triangle] * SIZE
        lo = np.maximum(np.floor(t.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(t.max(axis=0)).astype(int), SIZE - 1)
        if np.any(lo > hi):
            continue
        xx, yy = np.meshgrid(
            np.arange(lo[0], hi[0] + 1) + 0.5, np.arange(lo[1], hi[1] + 1) + 0.5
        )
        d = (t[1, 1] - t[2, 1]) * (t[0, 0] - t[2, 0]) + (t[2, 0] - t[1, 0]) * (
            t[0, 1] - t[2, 1]
        )
        if abs(d) < 1e-9:
            continue
        a = (
            (t[1, 1] - t[2, 1]) * (xx - t[2, 0]) + (t[2, 0] - t[1, 0]) * (yy - t[2, 1])
        ) / d
        b = (
            (t[2, 1] - t[0, 1]) * (xx - t[2, 0]) + (t[0, 0] - t[2, 0]) * (yy - t[2, 1])
        ) / d
        c = 1 - a - b
        inside = (a >= -1e-5) & (b >= -1e-5) & (c >= -1e-5)
        weights = np.stack([a, b, c], axis=-1)
        ids = mapping[triangle]
        region = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        world[region][inside] = (weights @ positions[ids])[inside]
        normal[region][inside] = (weights @ normals[ids])[inside]
        covered[region] |= inside
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
    return world[covered], normal[covered], covered


def main():
    started = time.perf_counter()
    data = json.loads((OUT / 'mesh.json').read_text())
    positions = np.array(data['positions']).reshape(-1, 3)
    faces = np.array(data['indices']).reshape(-1, 3)
    surface = trimesh.Trimesh(positions, faces, process=False)
    normals = np.array(surface.vertex_normals)
    atlas = xatlas.Atlas()
    atlas.add_mesh(positions.astype(np.float32), faces.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = SIZE
    pack.padding = 4
    atlas.generate(pack_options=pack)
    mapping, indices, uv = atlas[0]
    world, normal, covered = texels(positions, normals, indices, mapping, uv)
    v = PlyData.read(OUT / PLY_NAME)['vertex'].data
    scale = data['transform']['scale']
    center = np.array(data['transform']['center'])
    xyz = (np.column_stack([v[k] for k in 'xyz']) - center) * scale
    sizes = (
        np.exp(np.clip(np.column_stack([v[f'scale_{i}'] for i in range(3)]), -25, 5))
        * scale
    )
    rotations = Rotation.from_quat(
        np.column_stack([v[f'rot_{i}'] for i in (1, 2, 3, 0)])
    ).as_matrix()
    cov = np.einsum('nij,nj,nkj->nik', rotations, sizes * sizes, rotations)
    color = np.clip(
        0.5 + 0.28209479177387814 * np.column_stack([v[f'f_dc_{i}'] for i in range(3)]),
        0,
        1,
    )
    opacity = 1 / (1 + np.exp(-np.clip(v['opacity'], -30, 30)))
    accumulated = np.zeros((len(world), 3))
    total = np.zeros(len(world))
    best = np.zeros(len(world))
    fallback = np.zeros_like(accumulated)
    front_values = None
    front_alpha = None
    views = [
        (0, 0),
        (-25, 0),
        (25, 0),
        (-55, 0),
        (55, 0),
        (-90, 0),
        (90, 0),
        (-135, 0),
        (135, 0),
        (180, 0),
        (0, 40),
        (0, -35),
        (180, 40),
        (180, -35),
    ]
    for yaw, pitch in views:
        B = basis(yaw, pitch)
        rgb, alpha, depth = render(xyz, cov, color, opacity, B)
        Image.fromarray(np.uint8(np.clip(rgb[::-1], 0, 1) * 255)).save(
            CACHE / f'view-{yaw}-{pitch}.png'
        )
        projected = world @ B.T
        xy = np.floor(projected[:, :2] * (VIEW / SPAN) + VIEW / 2).astype(int)
        inside = (xy >= 0).all(axis=1) & (xy < VIEW).all(axis=1)
        xy = np.clip(xy, 0, VIEW - 1)
        x, y = xy.T
        if yaw == 0 and pitch == 0:
            front_values = rgb[y, x].copy()
            front_alpha = alpha[y, x].copy()
        facing = np.maximum(normal @ B[2], 0)
        visibility = np.exp(-(((projected[:, 2] - depth[y, x]) / 0.018) ** 2))
        quality = facing**8 * alpha[y, x] ** 2 * inside
        # Preserve the neutral front view where it is visible; alternate views
        # primarily fill profiles and disocclusions, avoiding blended facial edges.
        quality *= 2 if yaw == 0 and pitch == 0 else 1
        w = quality * visibility
        replace = quality > best
        best[replace] = quality[replace]
        fallback[replace] = rgb[y[replace], x[replace]]
        accumulated += rgb[y, x] * w[:, None]
        total += w
        print(f'Baked source view yaw={yaw}, pitch={pitch}', flush=True)
    values = accumulated / np.maximum(total[:, None], 1e-10)
    values[total < 1e-6] = fallback[total < 1e-6]
    # The same neutral source image carries the highest-frequency facial detail.
    # Project it over the central face to avoid double pupils from blending
    # imperfectly registered side views. Profiles/back retain multiview colors.
    central = (
        np.clip((0.091 - np.abs(world[:, 0])) / 0.018, 0, 1)
        * np.clip((world[:, 2] - 0.013) / 0.026, 0, 1)
        * np.clip((world[:, 1] + 0.087) / 0.025, 0, 1)
    )
    central *= np.clip((front_alpha - 0.05) / 0.20, 0, 1)
    values = values * (1 - central[:, None]) + front_values * central[:, None]
    texture = np.zeros((SIZE, SIZE, 3), np.uint8)
    texture[covered] = np.uint8(np.clip(values, 0, 1) * 255)
    # Extend island colors into the gutter to prevent filtering seams.
    _, nearest = distance_transform_edt(~covered, return_indices=True)
    texture[~covered] = texture[nearest[0][~covered], nearest[1][~covered]]
    Image.fromarray(texture[::-1]).save(OUT / 'appearance.png')
    metadata = {
        'mapping': mapping.tolist(),
        'indices': indices.ravel().tolist(),
        'uv': uv.ravel().tolist(),
        'texture': TEXTURE_URL,
        'stats': {
            'textureSize': SIZE,
            'sourceViews': len(views),
            'uvVertices': len(mapping),
            'coveredTexels': int(covered.sum()),
            'lowConfidenceFraction': float(np.mean(total < 1e-6)),
            'seconds': round(time.perf_counter() - started, 2),
            'method': (
                'Orthographic Gaussian alpha-composite views projected into '
                'xatlas UVs. Captured radiance, not de-lit albedo.'
            ),
        },
    }
    (OUT / 'texture-atlas.json').write_text(json.dumps(metadata, separators=(',', ':')))
    print(json.dumps(metadata['stats'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    if args.folder:
        OUT = args.folder.resolve()
        CACHE = OUT / 'texture-bake'
        CACHE.mkdir(exist_ok=True)
        PLY_NAME = 'face.ply'
        SIZE = 1024
        VIEW = 512
        TEXTURE_URL = f'/api/face-asset?id={OUT.name}&asset=appearance.png'
    main()
