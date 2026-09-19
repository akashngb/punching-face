"""Newton 1.6 CPU soft-tissue simulation, independent of the Three.js renderer.

The material parameters and layers are engineering priors, not measured anatomy.
Newton solves contact, shear and volumetric constraints on the tetrahedral cage.
"""

from pathlib import Path
import json, time
import numpy as np
import warp as wp
import newton

ROOT = Path(__file__).resolve().parent
OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
wp.config.quiet = True


def surface_normals(p, f):
    tri = p[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(p)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    if np.mean(out[:, 2]) < 0:
        out = -out
    return out


class NewtonFace:
    def __init__(self, cage, softness=0.6):
        started = time.perf_counter()
        self.rest = np.array(cage['positions'], dtype=float).reshape(-1, 3)
        self.faces = np.array(cage['indices'], dtype=np.int32).reshape(-1, 3)
        self.n = len(self.rest)
        if (
            self.n != 468
            or not np.isfinite(self.rest).all()
            or np.max(np.abs(self.rest)) > 1
        ):
            raise ValueError('Expected a finite, local 468-point facial cage.')
        self.normals = surface_normals(self.rest, self.faces)
        self.time = 0.0
        self.contact = None
        self.impacts = 0
        self.last_peak = 0.0
        self.softness = float(np.clip(softness, 0, 1))
        x, y, z = self.rest.T
        cheek = np.exp(
            -(((np.abs(x) - 0.052) / 0.03) ** 2) - ((y + 0.003) / 0.044) ** 2
        )
        lip = np.exp(-((x / 0.036) ** 2) - ((y + 0.040) / 0.025) ** 2)
        nose = np.exp(-((x / 0.021) ** 2) - ((y - 0.008) / 0.04) ** 2)
        forehead = np.clip((y - 0.04) / 0.045, 0, 1)
        self.thickness = (
            0.006 + 0.007 * cheek + 0.003 * lip - 0.002 * forehead - 0.002 * nose
        )
        stiffness = (
            2200 - 1000 * cheek - 600 * lip + 6500 * nose + 5500 * forehead
        ) * (1.35 - 0.7 * self.softness)
        inward = np.tile([0.0, 0.0, 1.0], (self.n, 1))
        layers = [
            self.rest,
            self.rest - inward * self.thickness[:, None] * 0.45,
            self.rest - inward * self.thickness[:, None],
        ]
        all_p = np.vstack(layers)
        tets = []
        mu = []
        for layer in range(2):
            for face in self.faces:
                a, b, c = sorted(map(int, face))
                a += layer * self.n
                b += layer * self.n
                c += layer * self.n
                A = a + self.n
                B = b + self.n
                C = c + self.n
                for tet in [[a, b, c, C], [a, b, B, C], [a, A, B, C]]:
                    q = all_p[tet]
                    volume = (
                        np.linalg.det(
                            np.stack([q[1] - q[0], q[2] - q[0], q[3] - q[0]], axis=1)
                        )
                        / 6
                    )
                    if abs(volume) < 1e-15:
                        continue
                    if volume < 0:
                        tet[1], tet[2] = tet[2], tet[1]
                    tets.append(tet)
                    mu.append(float(stiffness[face].mean()) * (1.6 if layer else 1.0))
        self.tets = np.asarray(tets, np.int32)
        self.initial = all_p
        self.rest_volumes = self.volumes(all_p)
        mass = np.zeros(len(all_p))
        for j in range(4):
            np.add.at(mass, self.tets[:, j], self.rest_volumes * 1000 / 4)
        mass[mass < 1e-10] = 0
        mass[self.n * 2 :] = 0
        mass[OVAL] = 0
        mass[np.array(OVAL) + self.n] = 0
        builder = newton.ModelBuilder()
        builder.gravity = np.array([0.0, 0.0, 0.0])
        builder.default_particle_radius = 0.0007
        builder.particle_max_velocity = 3.0
        for i, point in enumerate(all_p):
            builder.add_particle(
                wp.vec3(*point),
                wp.vec3(0.0),
                float(mass[i]),
                radius=0.0007 if i < self.n else 0.0001,
            )
        for tet, k in zip(self.tets, mu):
            builder.add_tetrahedron(*map(int, tet), k_mu=k, k_lambda=k * 8, k_damp=0.03)
        # A kinematic fist transfers contact into the skin through Newton.
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
            is_kinematic=True,
            label='tracked_fist',
        )
        cfg = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=0.05, kf=100.0, mu=0.35)
        builder.add_shape_sphere(body, radius=0.032, cfg=cfg, label='fist_contact')
        self.model = builder.finalize(device='cpu')
        self.model.soft_contact_margin = 0.004
        self.model.soft_contact_mu = 0.35
        self.model.soft_contact_ke = 1e4
        self.model.soft_contact_kd = 0.05
        # Adjacent FEM nodes must not collide with each other. Their volume constraints
        # already couple the tissue; dense mouth landmarks can be <1 mm apart.
        self.model.particle_max_radius = 0.0
        self.model.particle_grid = None
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=8,
            soft_body_relaxation=0.035,
            soft_contact_relaxation=0.6,
        )
        self.state = self.model.state()
        self.next = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.ready_ms = (time.perf_counter() - started) * 1000

    def volumes(self, positions):
        q = positions[self.tets]
        return (
            np.linalg.det(
                np.stack(
                    [q[:, 1] - q[:, 0], q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]], axis=-1
                )
            )
            / 6
        )

    def impact(self, point, direction, speed):
        point = np.asarray(point, float)
        direction = np.asarray(direction, float)
        if (
            point.shape != (3,)
            or direction.shape != (3,)
            or not np.isfinite(point).all()
            or not np.isfinite(direction).all()
            or np.max(np.abs(point)) > 1
        ):
            raise ValueError('Invalid contact vectors.')
        active = np.unique(self.faces)
        closest = int(
            active[np.argmin(np.linalg.norm(self.rest[active] - point, axis=1))]
        )
        normal = self.normals[closest]
        direction /= max(np.linalg.norm(direction), 1e-9)
        tangent = direction - normal * np.dot(direction, normal)
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        depth = min(self.thickness[closest] * 0.5, 0.006) * float(
            np.clip(speed / 0.9, 0.5, 1.2)
        )
        self.contact = {
            'point': self.rest[closest].copy(),
            'normal': normal.copy(),
            'tangent': tangent,
            'depth': depth,
            'start': self.time,
        }
        self.impacts += 1

    def step(self, dt=1 / 30):
        started = time.perf_counter()
        dt = float(np.clip(dt, 1 / 240, 1 / 30))
        steps = 8
        h = dt / steps
        for _ in range(steps):
            center = np.array([0.0, 0.0, 2.0])
            velocity = np.zeros(3)
            if self.contact:
                age = self.time - self.contact['start']
                duration = 0.18
                t = age / duration
                if t >= 1:
                    self.contact = None
                else:
                    c = self.contact
                    press = np.sin(np.pi * t)
                    center = (
                        c['point']
                        + c['normal'] * (0.034 - c['depth'] * press)
                        + c['tangent'] * 0.0015 * press
                    )
            self.state.body_q.assign(np.array([[*center, 0, 0, 0, 1]], np.float32))
            self.state.body_qd.zero_()
            self.state.clear_forces()
            previous = self.state.particle_q.numpy().copy()
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next, self.control, self.contacts, h)
            self.state, self.next = self.next, self.state
            # Backtrack an update that would invert a thin facial element.
            # This geometric safeguard retains Newton's direction and prevents
            # inverted tetrahedra from reaching the rendered surface.
            candidate = self.state.particle_q.numpy()
            ratios = self.volumes(candidate) / self.rest_volumes
            if not np.isfinite(candidate).all():
                raise ValueError('Newton produced a non-finite state.')
            if ratios.min() < 0.12:
                lo, hi = 0.0, 1.0
                for _ in range(12):
                    mid = (lo + hi) / 2
                    if (
                        np.min(
                            self.volumes(previous + (candidate - previous) * mid)
                            / self.rest_volumes
                        )
                        >= 0.12
                    ):
                        lo = mid
                    else:
                        hi = mid
                self.state.particle_q.assign(
                    (previous + (candidate - previous) * lo).astype(np.float32)
                )
                self.state.particle_qd.assign(
                    ((candidate - previous) * lo / h).astype(np.float32)
                )
            # Viscous damping removes kinetic energy, independently of FEM stiffness.
            vel = self.state.particle_qd.numpy()
            vel *= np.exp(-12 * h)
            self.state.particle_qd.assign(vel)
            self.time += h
        positions = self.state.particle_q.numpy()
        offset = positions[: self.n] - self.rest
        peak = float(np.max(np.linalg.norm(offset, axis=1)))
        volume = self.volumes(positions) / self.rest_volumes
        if not np.isfinite(positions).all() or peak > 0.045:
            raise ValueError(
                'Newton face simulation exceeded its stable range. Reset the face.'
            )
        self.last_peak = peak
        return {
            'offsets': offset.astype(np.float32).ravel().tolist(),
            'peakMm': peak * 1000,
            'minimumVolumeRatio': float(volume.min()),
            'medianVolumeRatio': float(np.median(volume)),
            'stepMs': (time.perf_counter() - started) * 1000,
            'simulationTime': self.time,
            'contacts': self.impacts,
        }

    def info(self):
        return {
            'engine': 'Newton',
            'version': newton.__version__,
            'solver': 'XPBD tetrahedral FEM',
            'device': 'cpu',
            'particles': self.model.particle_count,
            'tetrahedra': self.model.tet_count,
            'layers': ['skin', 'soft tissue', 'fixed inner support'],
            'inversionSafeguard': 'Backtracking limits tetrahedral volume ratios to 0.12 or above.',
            'materialParameters': 'Estimated visual simulation parameters; not measured tissue properties.',
        }


def load_cage(folder):
    path = folder / 'physics-cage.json'
    if not path.exists():
        raise ValueError('Build the photo model before starting Newton.')
    return json.loads(path.read_text())
