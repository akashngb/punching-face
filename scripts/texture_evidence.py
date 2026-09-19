"""Conservative photographic color evidence; no skin-tone classifier."""

from collections.abc import Mapping
import numpy as np


def photographed_cheek_color(rgba, landmarks):
    """Return an RGB median in [0, 1], or None without enough valid evidence.

    Preserve the existing 11x11 patches at landmarks 50, 280, 205 and 425.
    Landmarks may be a sequence of x/y dictionaries or an index-keyed mapping.
    Only pixels with alpha at least 220 contribute. Opaque bright and dark
    pixels retain their measured values: intensity filtering can remove one
    illuminated cheek and bias the whole-view exposure toward the other cheek.

    Coverage judgment: require at least half of the nominal 484 sample pixels
    and at least two patches with a majority (61/121) of usable pixels. Missing
    image-edge pixels count against coverage. This permits partial occlusion
    while preventing a tiny surviving patch from setting whole-view exposure.
    Invalid required landmarks return None instead of sampling another location.
    """
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError('Photographic cheek evidence requires a uint8 RGBA image.')
    height, width = rgba.shape[:2]
    if not height or not width:
        return None
    samples = []
    supported_patches = 0
    for index in (50, 280, 205, 425):
        try:
            if isinstance(landmarks, Mapping):
                landmark = (
                    landmarks[index] if index in landmarks else landmarks[str(index)]
                )
            else:
                landmark = landmarks[index]
            x, y = float(landmark['x']), float(landmark['y'])
        except (KeyError, IndexError, TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite([x, y]).all() or not (0 <= x < 1 and 0 <= y < 1):
            return None
        x, y = int(x * width), int(y * height)
        patch = rgba[
            max(0, y - 5) : min(height, y + 6), max(0, x - 5) : min(width, x + 6)
        ].reshape(-1, 4)
        rgb = patch[:, :3]
        valid = patch[:, 3] >= 220
        supported_patches += int(np.count_nonzero(valid) >= 61)
        samples.append(rgb[valid])
    if supported_patches < 2 or sum(len(patch) for patch in samples) < 242:
        return None
    return np.median(np.concatenate(samples), axis=0) / 255.0


class SupportedColorRegression(ValueError):
    """A completion changed supported color; ``diagnostic`` is compact JSON data."""

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            f"Completion changed {diagnostic['changedTexelCount']} supported texels "
            f"(maximum channel difference {diagnostic['maximumChannelDifference']}/255)."
        )


class SupportedColorSnapshot:
    """Compact protected texel IDs and their independent, read-only RGB bytes."""

    __slots__ = ('texel_count', 'ids', 'rgb')

    def __init__(self, texel_count, ids, rgb):
        self.texel_count = texel_count
        self.ids = np.array(ids, copy=True)
        self.rgb = np.array(rgb, copy=True)
        self.ids.flags.writeable = False
        self.rgb.flags.writeable = False


def _supported_color_array(color):
    color = np.asarray(color)
    if (
        color.ndim != 2
        or color.shape[1] != 3
        or color.dtype.kind not in 'iuf'
        or not np.isfinite(color).all()
    ):
        raise ValueError('Supported-color evidence requires finite Nx3 RGB values.')
    return color


def _supported_mask(mask, count, name):
    mask = np.asarray(mask)
    if mask.shape != (count,) or mask.dtype.kind != 'b':
        raise ValueError(f'{name} must be a Boolean mask matching the texel count.')
    return mask


def _texture_rgb_bytes(color):
    # Match bake_photographs serialization exactly: clip then truncate, not
    # round-to-nearest. Out-of-range finite exposure values are clipped normally.
    return np.uint8(np.clip(color, 0, 1) * 255)


def snapshot_supported_colors(color, parts, best, cleaned_coverage, bottom):
    """Snapshot supported prepared color immediately before generic completion.

    Protect part 0 with strongest unmasked physical support >= .12, exactly
    zero cleanup coverage, and no bottom-cap ownership. This is preservation of
    an accepted prepared-image blend, not proof of photographed skin/albedo.
    The final validator can exclude mouth/interior pixels once those masks are
    available. Only compact IDs and uint8 RGB are retained, never a full float
    atlas or references into the caller's mutable color array.
    """
    color = _supported_color_array(color)
    count = len(color)
    parts, support, cleaned = map(np.asarray, (parts, best, cleaned_coverage))
    if (
        parts.shape != (count,)
        or parts.dtype.kind not in 'iu'
        or np.any(parts < 0)
        or any(
            a.shape != (count,) or a.dtype.kind not in 'iuf' for a in (support, cleaned)
        )
    ):
        raise ValueError('Invalid part/support arrays for supported-color evidence.')
    if (
        not np.isfinite(support).all()
        or not np.isfinite(cleaned).all()
        or np.any(support < 0)
        or np.any(cleaned < 0)
        or np.any(cleaned > 1)
    ):
        raise ValueError('Support/cleanup evidence must be finite and in valid ranges.')
    bottom = _supported_mask(bottom, count, 'Bottom')
    ids = np.flatnonzero((parts == 0) & (support >= 0.12) & (cleaned == 0) & ~bottom)
    # Avoid a protected_texels x RGB float copy over a large atlas.
    rgb = np.empty((len(ids), 3), np.uint8)
    for start in range(0, len(ids), 65536):
        rgb[start : start + 65536] = _texture_rgb_bytes(
            color[ids[start : start + 65536]]
        )
    return SupportedColorSnapshot(count, ids, rgb)


def validate_supported_colors(
    snapshot, final_color, exclude=None, *, raise_on_change=True, sample_limit=16
):
    """Return a JSON-safe audit, or raise SupportedColorRegression with that audit.

    ``exclude`` marks final mouth/interior ownership. Compare the bytes that
    will actually reach the appearance PNG, allowing sub-quantization changes.
    At most ``sample_limit`` (0..64) changed IDs and before/after RGB samples are
    retained. This guard does not evaluate unsupported fallback seams, source
    registration, or whether the prepared photographic blend was correct.
    """
    if (
        not isinstance(snapshot, SupportedColorSnapshot)
        or not isinstance(snapshot.texel_count, (int, np.integer))
        or isinstance(snapshot.texel_count, (bool, np.bool_))
        or snapshot.texel_count < 0
    ):
        raise ValueError('Invalid supported-color snapshot.')
    ids, old = snapshot.ids, snapshot.rgb
    if (
        ids.ndim != 1
        or ids.dtype.kind not in 'iu'
        or old.shape != (len(ids), 3)
        or old.dtype != np.uint8
        or (
            len(ids)
            and (
                ids.min() < 0
                or ids.max() >= snapshot.texel_count
                or np.any(ids[1:] <= ids[:-1])
            )
        )
    ):
        raise ValueError('Invalid supported-color snapshot IDs or bytes.')
    if (
        not isinstance(sample_limit, (int, np.integer))
        or isinstance(sample_limit, (bool, np.bool_))
        or not 0 <= sample_limit <= 64
        or not isinstance(raise_on_change, (bool, np.bool_))
    ):
        raise ValueError('Invalid supported-color validation options.')
    color = _supported_color_array(final_color)
    if len(color) != snapshot.texel_count:
        raise ValueError('Final colors must match the snapshot texel count.')
    excluded = (
        np.zeros(len(color), bool)
        if exclude is None
        else _supported_mask(exclude, len(color), 'Exclusion')
    )
    audit = dict(
        version=1,
        protectedTexelCount=int(len(ids)),
        checkedTexelCount=0,
        excludedTexelCount=0,
        changedTexelCount=0,
        maximumChannelDifference=0,
        changedIds=[],
        samples=[],
        passed=True,
        scope='Preservation of supported part-0 prepared colors through fallback completion.',
        quantization='clip RGB to [0,1], multiply by 255, truncate to uint8.',
        limitation='Does not validate unobserved fallback seams or source registration.',
    )
    for start in range(0, len(ids), 65536):
        selected = ids[start : start + 65536]
        keep = ~excluded[selected]
        selected = selected[keep]
        before = old[start : start + 65536][keep]
        after = _texture_rgb_bytes(color[selected])
        delta = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=1)
        changed = np.flatnonzero(delta)
        audit['checkedTexelCount'] += int(len(selected))
        audit['changedTexelCount'] += int(len(changed))
        if len(delta):
            audit['maximumChannelDifference'] = max(
                audit['maximumChannelDifference'], int(delta.max())
            )
        for row in changed[: max(0, sample_limit - len(audit['samples']))]:
            texel_id = int(selected[row])
            audit['changedIds'].append(texel_id)
            audit['samples'].append(
                dict(
                    texelId=texel_id,
                    before=before[row].tolist(),
                    after=after[row].tolist(),
                )
            )
    audit['excludedTexelCount'] = (
        audit['protectedTexelCount'] - audit['checkedTexelCount']
    )
    audit['passed'] = audit['changedTexelCount'] == 0
    if not audit['passed'] and raise_on_change:
        raise SupportedColorRegression(audit)
    return audit
