"""Conservative surface smoothing of low-frequency camera ownership.

Captured RGB, high-frequency detail and physical-support/fallback scores stay
separate. A second projection visits only texels whose ownership changed.
"""

import numpy as np


def _smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def lower_lateral_region(points, face, parts):
    width = np.ptp(face[:468, 0])
    lower = _smooth((face[1, 1] + 0.015 - points[:, 1]) / 0.035)
    lateral = _smooth((np.abs(points[:, 0]) - width * 0.22) / (width * 0.16))
    jaw = _smooth(
        (face[17, 1] - points[:, 1]) / max((face[17, 1] - face[152, 1]) * 0.65, 0.005)
    )
    return lower * np.maximum(lateral, jaw) * (np.asarray(parts) == 0)


class CameraTexture:
    def __init__(
        self, vertices, faces, points, parts, labels, binding, views, completion=None
    ):
        self.vertices, self.faces, self.binding = vertices, faces, binding
        self.points, self.completion = points, completion
        self.region = lower_lateral_region(points, vertices, parts)
        self.ids = np.flatnonzero(self.region > 0)
        self.slots = np.full(len(points), -1, dtype=np.int32)
        self.slots[self.ids] = np.arange(len(self.ids))
        self.views = {view.name: index for index, view in enumerate(views)}
        self.weights = np.zeros((len(self.ids), len(views)))
        self.domain = lower_lateral_region(vertices, vertices, labels) > 0

    def observe(self, view, near, weight):
        slots = self.slots[near]
        valid = slots >= 0
        self.weights[slots[valid], self.views[view.name]] = weight[valid]

    def reblend(
        self, views, project, workers, confidence, cleanup, hair, hair_total, hair_best
    ):
        from concurrent.futures import ThreadPoolExecutor
        from scripts.camera_ownership import (
            fit_camera_ownership_delta,
            apply_camera_ownership_delta,
        )

        ids = self.ids
        total = self.weights.sum(axis=1)
        probabilities = np.divide(
            self.weights,
            total[:, None],
            out=np.zeros_like(self.weights),
            where=total[:, None] > 1e-7,
        )
        del self.weights
        hair_ratio = np.divide(
            hair[ids],
            hair_total[ids],
            out=np.zeros(len(ids)),
            where=hair_total[ids] > 0,
        )
        from scripts.head_material import photographed_scalp_region

        scalp = photographed_scalp_region(
            self.points[ids],
            self.vertices,
            self.completion,
            hair[ids],
            hair_total[ids],
            hair_best[ids],
        )
        blend = (
            self.region[ids]
            * _smooth((confidence[ids] - 0.12) / 0.13)
            * (cleanup[ids] == 0)
            * (hair_ratio < 0.05)
            * (1 - scalp)
        )
        eligible = (blend > 0) & ((probabilities > 1e-5).sum(axis=1) > 1)
        sampled = np.flatnonzero(eligible)
        sampled = sampled[:: max(1, int(np.ceil(len(sampled) / 60000)))]
        audit = {
            'method': 'Surface-edge diffusion of low-frequency camera ownership',
            'sampledTexels': int(len(sampled)),
            'changedTexels': 0,
            'sourceColorsUnchanged': True,
            'physicalSupportUnchanged': True,
            'fineDetailOwnershipUnchanged': True,
            'limitation': 'Local view blending, not recovered skin albedo or illumination.',
        }
        if len(sampled) < 32:
            return np.empty(0, dtype=int), np.empty((0, 3)), audit

        def subset(selected):
            return {
                'triangles': self.binding['triangles'],
                'triangleIds': self.binding['triangleIds'][selected],
                'weights': self.binding['weights'][selected],
            }

        used = np.unique(self.faces)
        cut = self.vertices[used, 1].min()
        cap = np.all(self.vertices[self.faces, 1] < cut + 0.0001, axis=1)
        delta, diffusion = fit_camera_ownership_delta(
            self.vertices,
            self.faces[~cap],
            subset(ids[sampled]),
            probabilities[sampled],
            vertex_domain=self.domain,
            steps=6,
            diffusion_scale=0.004,
        )
        revised, masking = apply_camera_ownership_delta(
            probabilities,
            subset(ids),
            delta,
            support_mask=probabilities > 0,
            blend=blend,
        )
        changed = np.max(np.abs(revised - probabilities), axis=1) > 1e-10
        audit.update(
            diffusion=diffusion, masking=masking, changedTexels=int(changed.sum())
        )
        selected = ids[changed]
        probabilities = revised[changed]
        if not len(selected):
            return selected, np.empty((0, 3)), audit
        # The projector's original alpha, depth, ownership and source masks
        # are evaluated again. No camera rejected in pass one receives weight.
        result = np.zeros((len(selected), 3))
        with ThreadPoolExecutor(workers, thread_name_prefix='photo-ownership') as pool:
            for start in range(0, len(views), workers):
                batch = views[start : start + workers]
                outputs = pool.map(lambda view: project(view, selected), batch)
                for view, projected in zip(batch, outputs):
                    near, contributions = projected[:2]
                    weight = contributions['total']
                    rgb = np.divide(
                        contributions['accum'],
                        weight[:, None],
                        out=np.zeros_like(contributions['accum']),
                        where=weight[:, None] > 0,
                    )
                    slot = np.searchsorted(selected, near)
                    revised_weight = probabilities[slot, self.views[view.name]]
                    if np.any((revised_weight > 0) & (weight <= 0)):
                        raise ValueError(
                            'Camera ownership revived a rejected projection.'
                        )
                    result[slot] += rgb * revised_weight[:, None]
        return selected, result, audit
