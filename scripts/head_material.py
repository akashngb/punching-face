"""Continuous missing-head appearance without spherical UV poles.

Unseen hair is synthesized from a bounded photo swatch in Cartesian space;
Astra supplies hairstyle parameters, not photographic evidence.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, distance_transform_edt


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def rear_reference(path):
    rear = np.asarray(Image.open(path).convert('RGB')) / 255.0
    fg = ((np.ptp(rear, axis=2) > 0.10) | (rear.max(2) < 0.44)) & (
        (rear.max(2) < 0.85) | (rear.min(2) < 0.55)
    )
    rows = np.where(fg.sum(1) > rear.shape[1] * 0.03)[0]
    cols = np.where(fg.sum(0) > rear.shape[0] * 0.03)[0]
    if not len(rows) or not len(cols):
        return None
    rear = rear[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()
    fg = fg[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    _, near = distance_transform_edt(~fg, return_indices=True)
    rear[~fg] = rear[near[0][~fg], near[1][~fg]]
    return rear


def sample(image, u, v):
    return np.stack(
        [
            map_coordinates(
                image[:, :, i],
                [v * (image.shape[0] - 1), u * (image.shape[1] - 1)],
                order=1,
                mode='nearest',
            )
            for i in range(3)
        ],
        1,
    )


def head_scalp_region(x, p, spec, scalp_override=None):
    """Shared photographed/prior scalp ownership for skin and hair materials."""
    hair = (spec or {}).get('hair', {})
    height = p[10, 1] - p[152, 1]
    azimuth = np.abs(np.arctan2(x[:, 0], x[:, 2] + 0.09))
    threshold = np.interp(
        azimuth,
        [0, 0.6, 1.45, 2.15, np.pi],
        [
            p[10, 1] + 0.005,
            p[10, 1] - 0.015,
            p[10, 1] - height * hair.get('sideHairlineFraction', 0.3),
            p[10, 1] - height * hair.get('rearHairlineFraction', 0.85),
            p[10, 1] - height * hair.get('rearHairlineFraction', 0.85),
        ],
    )
    scalp = smooth((x[:, 1] - threshold) / 0.014)
    if scalp_override is not None:
        scalp = np.where(np.isfinite(scalp_override), scalp_override, scalp)
    if hair.get('present') is False:
        scalp *= 0
    return scalp


def photographed_scalp_region(x, p, spec, votes, visibility, strongest):
    """Let visible semantic observations refine the hairstyle prior smoothly.

    Texture selection uses facing**8 to choose sharp photographs. A low value
    is not evidence that visible temple skin is hair. Semantic support uses
    projected area with a separate grazing fade, retains visibility/cutout
    tests and excludes occluders.
    Require a useful individual view as well as aggregate support, so many
    grazing views cannot jointly erase an unseen region's prior. These scores
    are viewing heuristics, not calibrated classification probabilities.
    """
    prior = head_scalp_region(x, p, spec)
    visibility = np.asarray(visibility)
    ratio = np.divide(
        votes,
        visibility,
        out=np.zeros_like(visibility, dtype=float),
        where=visibility > 0,
    )
    semantic = np.clip((ratio - 0.25) / 0.5, 0, 1)
    trust = smooth((np.asarray(strongest) - 0.15) / 0.15) * smooth(
        (visibility - 0.15) / 0.15
    )
    scalp = prior * (1 - trust) + semantic * trust
    if (spec or {}).get('hair', {}).get('present') is False:
        scalp *= 0
    return scalp


def missing_head_material(
    x,
    n,
    p,
    spec,
    skin,
    rear=None,
    swatch=None,
    scalp_override=None,
    short_swatches=None,
):
    hair = (spec or {}).get('hair', {})
    lo = p.min(0)
    hi = p.max(0)
    base = np.array(hair.get('colorSrgb', [0.075, 0.064, 0.049]))
    # The swatch stays strictly inside the rear hair mass. Cartesian mirrored
    # repeat has finite derivatives at the crown, unlike atan2/spherical UVs.
    if swatch is None and rear is not None:
        h, w = rear.shape[:2]
        patch = rear[
            int(h * 0.12) : int(h * 0.42), int(w * 0.28) : int(w * 0.72)
        ].copy()
        dark = (patch.mean(2) < 0.38) & (patch.max(2) < 0.62)
        if dark.any():
            _, nearest = distance_transform_edt(~dark, return_indices=True)
            patch[~dark] = patch[nearest[0][~dark], nearest[1][~dark]]
            swatch = patch
    scale = 0.070
    angle = np.radians(hair.get('flowDegrees', 15))
    c, s = np.cos(angle), np.sin(angle)

    def tile(a, b, patch, texture_scale=scale):
        a, b = (c * a - s * b) / texture_scale, (s * a + c * b) / texture_scale
        a = 1 - np.abs((a + 0.37) % 2 - 1)
        b = 1 - np.abs((b + 0.21) % 2 - 1)
        if patch is not None:
            return sample(patch, a, b)
        wavelength = hair.get('waveLengthMm', 12) * 0.001
        strand = 0.8 + 0.2 * np.sin(
            (a * scale + 0.004 * np.sin(b * 12)) * 2 * np.pi / wavelength
        )
        fine = 0.92 + 0.08 * np.sin(a * 780 + b * 13)
        return base[None] * strand[:, None] * fine[:, None]

    weights = np.maximum(np.abs(n), 0.001) ** 4
    weights /= weights.sum(1)[:, None]
    hair_rgb = (
        tile(x[:, 2], x[:, 1], swatch) * weights[:, 0, None]
        + tile(x[:, 0], x[:, 2], swatch) * weights[:, 1, None]
        + tile(x[:, 0], x[:, 1], swatch) * weights[:, 2, None]
    )
    if short_swatches and hair.get('sideLengthMm', 0) < hair.get('lengthMm', 1) * 0.7:

        def short_texture(patch):
            return (
                tile(x[:, 2], x[:, 1], patch, 0.024) * weights[:, 0, None]
                + tile(x[:, 0], x[:, 2], patch, 0.024) * weights[:, 1, None]
                + tile(x[:, 0], x[:, 1], patch, 0.024) * weights[:, 2, None]
            )

        fallback = short_swatches.get('rear', next(iter(short_swatches.values())))
        left = short_texture(short_swatches.get('left', fallback))
        right = short_texture(short_swatches.get('right', fallback))
        side = smooth((x[:, 0] + 0.015) / 0.030)[:, None]
        short = left * (1 - side) + right * side
        posterior = smooth((np.abs(np.arctan2(x[:, 0], x[:, 2] + 0.09)) - 1.6) / 0.7)[
            :, None
        ]
        short = short * (1 - posterior) + short_texture(fallback) * posterior
        taper = (
            (1 - smooth((x[:, 1] - p[10, 1] + 0.010) / 0.060))
            * smooth((-x[:, 2] - 0.045) / 0.060)
        )[:, None]
        hair_rgb = hair_rgb * (1 - taper) + short * taper
    # Smooth semantic hairline around temples and nape, never a skin wedge at
    # a texture pole. At the crown the weight is uniformly one.
    scalp = head_scalp_region(x, p, spec, scalp_override)
    inferred = (
        np.tile(skin, (len(x), 1)) * (1 - scalp[:, None]) + hair_rgb * scalp[:, None]
    )
    if rear is not None:
        # Orthographic posterior projection, blended out before it becomes
        # tangential. No cylindrical wrap, no pole, no pinched center texel.
        u = np.clip(0.5 - x[:, 0] / max(hi[0] - lo[0], 0.1), 0.035, 0.965)
        v = np.clip((hi[1] - x[:, 1]) / max(hi[1] - p[152, 1], 0.2), 0.03, 0.96)
        posterior = sample(rear, u, v)
        # Keep any reference-skin pixels off the model's inferred hair area.
        is_skin = smooth((posterior.mean(1) - 0.24) / 0.16) * scalp
        posterior = posterior * (1 - is_skin[:, None]) + hair_rgb * is_skin[:, None]
        blend = smooth((-n[:, 2] - 0.25) / 0.55) * (1 - smooth((n[:, 1] - 0.25) / 0.45))
        inferred = inferred * (1 - blend[:, None]) + posterior * blend[:, None]
    return inferred, scalp
