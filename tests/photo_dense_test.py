"""Ground-truth checks for the dense depth stage. Synthetic cameras photograph
a solid-textured surface, so the true prior error is known exactly. No capture
data and no pycolmap: run with .venv/bin/python tests/photo_dense_test.py
"""

from pathlib import Path
import sys
import numpy as np
import open3d as o3d
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.photo_dense import (
    Views,
    barycentric,
    measure,
    poisson_disk,
    raycaster,
    solve_field,
)


def surface():
    ball = trimesh.creation.icosphere(subdivisions=5, radius=0.09)
    p = np.array(ball.vertices)
    f = np.array(ball.faces)
    # Low, face-scale relief so the patch normal is not a perfect sphere's.
    p *= (1 + 0.06 * np.sin(p[:, 0] * 40) * np.cos(p[:, 1] * 35))[:, None]
    front = p[f].mean(1)[:, 2] > 0.035
    return p, np.vstack([f[front], f[~front]]), int(front.sum())


def solid_texture(x, seed=3):
    rng = np.random.default_rng(seed)
    frequency = rng.normal(size=(40, 3)) * rng.choice([250, 600, 1200], size=(40, 1))
    phase = rng.uniform(0, 6.28, 40)
    amplitude = rng.uniform(0.3, 1, 40)
    return 0.5 + 0.25 * np.tanh(
        (np.sin(x @ frequency.T + phase) * amplitude).sum(1) / 3
    )


def photograph(p, f, yaws, width=640, height=480, focal=900.0, distance=0.5):
    scene = raycaster(p, f)
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.stack(
        [(u - width / 2) / focal, (v - height / 2) / focal, np.ones_like(u)], -1
    ).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    R = []
    t = []
    centers = []
    gray = []
    alpha = []
    for yaw in np.radians(yaws):
        center = np.array([np.sin(yaw), 0.0, np.cos(yaw)]) * distance
        forward = -center / distance
        right = np.cross(forward, [0.0, -1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        world = rays @ rotation
        hit = scene.cast_rays(
            o3d.core.Tensor(
                np.concatenate([np.tile(center, (len(world), 1)), world], 1).astype(
                    np.float32
                )
            )
        )['t_hit'].numpy()
        seen = np.isfinite(hit)
        image = np.zeros(len(world), np.float32)
        image[seen] = solid_texture(center + world[seen] * hit[seen, None])
        R.append(rotation)
        t.append(-rotation @ center)
        centers.append(center)
        gray.append(image.reshape(height, width))
        alpha.append(seen.reshape(height, width))
    names = np.array([f'view_{i:02d}.png' for i in range(len(yaws))])
    cameras = {
        'names': names,
        'R': np.stack(R),
        't': np.stack(t),
        'centers': np.stack(centers),
        'params': np.array([focal, width / 2, height / 2, 0.0]),
        'width': width,
        'height': height,
    }
    return Views(
        None, cameras, set(names), gray=np.stack(gray), alpha=np.stack(alpha)
    ), dict(zip(names, yaws))


def test_measured_offsets_match_known_prior_error():
    p, f, face_count = surface()
    views, yaw = photograph(p, f, np.linspace(-42, 42, 15))
    x, n = poisson_disk(p, f[:face_count], 1500)
    known = 0.003 * np.sin(x[:, 0] * 55) * np.cos(x[:, 1] * 45)
    frame = (np.zeros(3), np.eye(3), 1.0)
    offset, peak, accepted, evidence = measure(
        x + n * known[:, None], n, raycaster(p, f), views, yaw, frame, reach=0.006
    )
    error = np.abs(offset + known)[accepted] * 1000
    assert evidence['consensus'] > 0.3, evidence
    assert np.median(error) < 0.25 and np.quantile(error, 0.9) < 0.8, (
        np.median(error),
        np.quantile(error, 0.9),
    )
    return evidence['consensus'], float(np.median(error))


def test_field_solve_outvotes_outliers():
    p, f, face_count = surface()
    face = f[:face_count]
    x, _ = poisson_disk(p, face, 900)
    tri, bary = barycentric(p, face, x)
    rng = np.random.default_rng(5)
    truth = lambda q: 0.002 * np.sin(q[:, 0] * 30) * np.cos(q[:, 1] * 25)
    offset = truth(x) + rng.normal(0, 0.0003, len(x))
    wrong = rng.random(len(x)) < 0.12
    offset[wrong] += rng.choice([-1, 1], wrong.sum()) * rng.uniform(
        0.004, 0.008, wrong.sum()
    )
    d, trust = solve_field(p, face, tri, bary, offset, np.full(len(x), 0.3))
    used = np.unique(face)
    rim = np.abs(d[used]) < 1e-9
    error = np.sqrt(np.mean((d[used] - truth(p[used]))[~rim] ** 2)) * 1000
    assert error < 0.5, error
    assert np.median(trust[wrong]) < 0.25 < np.median(trust[~wrong]), (
        np.median(trust[wrong]),
        np.median(trust[~wrong]),
    )
    return float(error)


if __name__ == '__main__':
    consensus, median = test_measured_offsets_match_known_prior_error()
    print(f'measure: consensus {consensus:.0%}, median depth error {median:.3f} mm')
    print(
        f'solve_field: RMS field error {test_field_solve_outvotes_outliers():.3f} mm with 12% gross outliers'
    )
    print('photo_dense_test passed')
