"""Estimated local eye occlusion from the fitted globe and surrounding lids.

This is a bounded appearance prior, not recovered lighting or scanned eye
reflectance. It returns scalar multipliers and never changes eye albedo/chroma.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def apply_estimated_eye_shading(
    positions, triangles, vertex_parts, texels, texel_parts, color, estimated_parts
):
    """Shade generated eye appearance in linear light; preserve all other pixels."""
    if not set(estimated_parts).issubset({1, 2}):
        raise ValueError('Only generated eyeball parts may receive socket shading.')
    mask = np.isin(texel_parts, list(estimated_parts))
    result = color.copy()
    if not mask.any():
        return result, {
            'estimated': True,
            'applied': False,
            'reason': 'No generated eyeball material.',
        }
    multiplier, audit = estimate_eye_shading(
        positions, triangles, vertex_parts, texels[mask], texel_parts[mask]
    )
    rgb = np.clip(color[mask], 0, 1)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    linear *= multiplier[:, None]
    shaded = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    # Exactly preserve samples receiving no shade, including hidden globe backs.
    result[mask] = np.where((multiplier < 1)[:, None], shaded, color[mask])
    audit.update(
        applied=True,
        generatedParts=sorted(estimated_parts),
        photographicIrisUnchanged=True,
    )
    return result, audit


def _spread_samples(points, count):
    """Deterministic farthest-point samples; no dependence on atlas density."""
    selected = [int(np.argmax(points[:, 2]))]
    distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, min(count, len(points))):
        index = int(np.argmax(distance))
        if distance[index] < 1e-18:
            break
        selected.append(index)
        distance = np.minimum(distance, np.sum((points - points[index]) ** 2, axis=1))
    return np.asarray(selected)


def _hemisphere(normals, count):
    """Fixed cosine-weighted directions around each globe's outward normal."""
    unit = (np.arange(count) + 0.5) / count
    angle = np.arange(count) * (np.pi * (3 - np.sqrt(5)))
    local = np.column_stack(
        (
            np.sqrt(unit) * np.cos(angle),
            np.sqrt(unit) * np.sin(angle),
            np.sqrt(1 - unit),
        )
    )
    axis = np.tile([0.0, 1.0, 0.0], (len(normals), 1))
    axis[np.abs(normals[:, 1]) > 0.9] = [1.0, 0.0, 0.0]
    tangent = np.cross(axis, normals)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    bitangent = np.cross(normals, tangent)
    return (
        local[None, :, :1] * tangent[:, None]
        + local[None, :, 1:2] * bitangent[:, None]
        + local[None, :, 2:] * normals[:, None]
    )


def estimate_eye_shading(
    positions,
    triangles,
    vertex_parts,
    texels,
    texel_parts,
    samples_per_eye=96,
    rays_per_sample=24,
    max_distance=0.025,
    minimum_shading=0.55,
):
    """Return ``(multipliers, audit)`` in the input texel order.

    Parts 1 and 2 identify disconnected eyeballs; other parts receive exactly
    one. Coordinates use the pipeline's normalized head metres, with +Z forward.
    Only front-globe samples cast short rays. Interpolation in 3D and a smooth
    silhouette fade keep unsampled rear eye surfaces from inheriting lid shade.
    Ray count is bounded by two eyes * samples_per_eye * rays_per_sample.
    """
    positions = np.asarray(positions, float)
    triangles = np.asarray(triangles, int).reshape(-1, 3)
    vertex_parts = np.asarray(vertex_parts)
    texels = np.asarray(texels, float)
    texel_parts = np.asarray(texel_parts)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or texels.ndim != 2
        or texels.shape[1] != 3
        or vertex_parts.shape != (len(positions),)
        or texel_parts.shape != (len(texels),)
        or not np.isfinite(positions).all()
        or not np.isfinite(texels).all()
        or (
            triangles.size
            and (triangles.min() < 0 or triangles.max() >= len(positions))
        )
    ):
        raise ValueError('Invalid eye shading geometry or part labels.')
    if (
        not isinstance(samples_per_eye, int)
        or not 4 <= samples_per_eye <= 512
        or not isinstance(rays_per_sample, int)
        or not 4 <= rays_per_sample <= 128
        or not np.isfinite(max_distance)
        or max_distance <= 0
        or not np.isfinite(minimum_shading)
        or not 0 < minimum_shading <= 1
    ):
        raise ValueError('Invalid bounded eye shading settings.')
    shading = np.ones(len(texels))
    audit = {
        'estimated': True,
        'method': 'Short-range globe-normal hemisphere visibility against fitted face and lids.',
        'limitation': (
            'Fitted geometry supplies estimated socket shading; illumination and '
            'eye reflectance are not measured.'
        ),
        'minimumAllowedMultiplier': float(minimum_shading),
        'maxRayDistance': float(max_distance),
        'eyes': {},
        'raysCast': 0,
    }
    # Eye geometry must never shadow itself or its partner. Mixed triangles are
    # excluded conservatively rather than treating an eye boundary as a lid.
    eye_vertices = np.isin(vertex_parts, [1, 2])
    occluder_faces = triangles[~eye_vertices[triangles].any(axis=1)]
    if not len(occluder_faces):
        audit['reason'] = (
            'No surrounding face or lid triangles; retained baseline shading.'
        )
        return shading, audit
    occluders = trimesh.Trimesh(positions, occluder_faces, process=False)
    for part in (1, 2):
        globe = positions[vertex_parts == part]
        owned = np.flatnonzero(texel_parts == part)
        if len(globe) < 4 or not len(owned):
            continue
        center = (globe.min(axis=0) + globe.max(axis=0)) * 0.5
        radii = np.ptp(globe, axis=0) * 0.5
        if np.any(radii < 1e-6):
            audit['eyes'][str(part)] = {
                'available': False,
                'reason': 'Degenerate fitted globe.',
            }
            continue

        def normals(points):
            normal = (points - center) / radii**2
            return normal / np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)

        globe_normals = normals(globe)
        front = globe[globe_normals[:, 2] > 0.05]
        if not len(front):
            continue
        samples = front[_spread_samples(front, samples_per_eye)]
        sample_normals = normals(samples)
        directions = _hemisphere(sample_normals, rays_per_sample).reshape(-1, 3)
        epsilon = min(0.0001, float(radii.min()) * 0.01, max_distance * 0.01)
        origins = np.repeat(samples + sample_normals * epsilon, rays_per_sample, axis=0)
        locations, ray_ids, _ = occluders.ray.intersects_location(
            origins, directions, multiple_hits=False
        )
        blocked = np.zeros(len(origins))
        if len(ray_ids):
            distance = np.linalg.norm(locations - origins[ray_ids], axis=1)
            # Remote head surfaces cannot act as a binary occluder. The local
            # effect falls continuously to zero at the finite ray horizon.
            proximity = np.clip(1 - distance / max_distance, 0, 1)
            blocked[ray_ids] = proximity * proximity * (3 - 2 * proximity)
        visibility = blocked.reshape(len(samples), rays_per_sample).mean(axis=1)
        distance, indices = cKDTree(samples).query(
            texels[owned], k=min(8, len(samples))
        )
        distance = np.asarray(distance).reshape(len(owned), -1)
        indices = np.asarray(indices).reshape(len(owned), -1)
        weights = 1 / np.maximum(distance, 0.00025) ** 2
        occlusion = np.sum(visibility[indices] * weights, axis=1) / weights.sum(axis=1)
        front_weight = np.clip(normals(texels[owned])[:, 2] / 0.2, 0, 1)
        front_weight = front_weight * front_weight * (3 - 2 * front_weight)
        shading[owned] = 1 - (1 - minimum_shading) * occlusion * front_weight
        audit['eyes'][str(part)] = {
            'available': True,
            'samples': len(samples),
            'rays': len(origins),
            'texels': len(owned),
            'minimumMultiplier': float(shading[owned].min()),
            'meanMultiplier': float(shading[owned].mean()),
        }
        audit['raysCast'] += len(origins)
    return np.clip(shading, minimum_shading, 1), audit
