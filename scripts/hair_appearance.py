"""Photographic support for hair, separate from camera blend preference."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def semantic_angular_support(facing):
    """Use projected area for visible classes, with a grazing-angle rejection.

    A skin/hair region is still discernible in an oblique photograph whose
    fine-detail RGB preference is tiny. Do not cube that preference again.
    Fade evidence between dot products .25 and .40 to retain uncertainty at
    grazing incidence. Occlusion, source alpha and accessory tests must still
    multiply this score; it is a viewing heuristic, not class probability.
    """
    facing = np.clip(np.asarray(facing), 0, 1)
    angular = np.clip((facing - 0.25) / 0.15, 0, 1)
    return facing * angular * angular * (3 - 2 * angular)


def skin_semantic_support(hair_mask, foreground, boundary_width):
    """Qualify extra skin support well outside a coarse hair annotation.

    A polygon's exterior is not automatically observed skin: missed curls,
    sparse taper and source-cutout fringes are unknown. Keep a deadband of
    one boundary width and smoothly gain support over the next width. The
    caller scales that width with the annotated head crop. Near either edge,
    retain the previous conservative angular evidence rather than treating
    the newly relaxed view weight as a confident skin observation.
    """
    hair_mask = np.asarray(hair_mask, bool)
    foreground = np.asarray(foreground, bool)
    if hair_mask.ndim != 2 or hair_mask.shape != foreground.shape:
        raise ValueError('Hair and foreground masks must share a 2D shape.')
    if not np.isfinite(boundary_width) or boundary_width <= 0:
        raise ValueError('Semantic boundary width must be positive and finite.')
    outside = (
        distance_transform_edt(~hair_mask)
        if hair_mask.any()
        else np.full(hair_mask.shape, boundary_width * 2)
    )
    # Padding also treats the image border as unobserved, including crops
    # whose alpha happens to be fully opaque.
    interior = distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
    margin = np.minimum(outside, interior)
    weight = np.clip((margin - boundary_width) / boundary_width, 0, 1)
    return weight * weight * (3 - 2 * weight)


def semantic_view_support(facing, class_support):
    """Relax oblique evidence only where class boundaries support that change.

    Deleting existing boundary evidence can replace already observed skin
    with the hair prior. Retain the conservative cubic score there, and use
    the projected-area score in supported regions. Other rejection masks
    remain authoritative and must multiply the returned score.
    """
    facing = np.clip(np.asarray(facing), 0, 1)
    conservative = facing**3
    return conservative + np.clip(class_support, 0, 1) * (
        semantic_angular_support(facing) - conservative
    )


def annotated_hair_support(rgb, mask):
    """Feather coarse polygons and reject colors unlike their photographed core.

    Vision polygons can include a few rows of neck skin. Estimate a robust
    color range from this view's own hair, so extending oblique support cannot
    promote those mislabeled pixels. There is no fixed dark-hair threshold.
    Original high-confidence texture is unaffected by this conservative gate.
    """
    distance = distance_transform_edt(np.asarray(mask, bool))
    core = np.asarray(rgb)[distance > 6]
    if len(core) < 32:
        return np.zeros(distance.shape)
    median = np.median(core, axis=0)
    spread = np.maximum(4 * np.median(np.abs(core - median), axis=0), 0.06)
    deviation = np.max(np.abs(rgb - median) / spread, axis=2)
    color = np.clip(2 - deviation, 0, 1)
    edge = np.clip(distance / 6, 0, 1)
    return (edge * edge * (3 - 2 * edge)) * (color * color * (3 - 2 * color))


def hair_photo_support(blend_quality, facing, hair_interior):
    """Undo the view-selection exponent without reviving rejected samples.

    The bake uses facing**8 to choose between overlapping photographs. That
    is not a visibility test: a visible surface at 60 degrees still has half
    its projected resolution, but only 1/256 of the blend weight. Using that
    weight for completion erased observed rear hair. Keep the actual alpha,
    silhouette, depth and accessory rejections, with facing**2 as support.
    Extend support only inside a feathered hair annotation: skin at the nape
    and unclassified views retain the original conservative support.
    """
    facing = np.asarray(facing)
    quality = np.asarray(blend_quality)
    support = np.divide(
        quality,
        facing**6,
        out=np.zeros_like(quality, dtype=float),
        where=(facing > 0) & (quality > 1e-7),
    )
    return np.maximum(quality, support * np.clip(hair_interior, 0, 1))


def blend_hair_completion(photo, inferred, hair_accum, hair_total, old_weight, weight):
    """Replace only discarded fallback with color from its supporting hair views."""
    usable = hair_total > 1e-7
    hair = np.divide(
        hair_accum, hair_total[:, None], out=np.zeros_like(photo), where=usable[:, None]
    )
    recovered = np.maximum(old_weight - weight, 0) * usable
    return (
        photo * (1 - old_weight[:, None])
        + hair * recovered[:, None]
        + inferred * (old_weight - recovered)[:, None]
    )
