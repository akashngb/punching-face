"""Short-hair fallback from observed side/rear patches, never the front quiff."""

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def short_hair_patch(rgb, mask):
    """Choose a square fully inside the lower annotated hair, excluding cutouts.

    No inpainting across an ear/skin boundary and no fixed dark-hair assumption.
    Coordinates and patch size remain in the registered source image pixels.
    """
    rgb, mask = np.asarray(rgb), np.asarray(mask, bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != mask.shape:
        raise ValueError('Hair patch requires aligned RGB and mask arrays.')
    yy, xx = np.where(mask)
    if len(yy) < 100:
        return None
    span = yy.max() - yy.min()
    distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    rows = np.indices(mask.shape)[0]
    # Below the long crown but above the nape boundary where polygons can
    # include bare skin. Prefer the deepest interior in that horizontal band.
    eligible = (rows > yy.min() + span * 0.50) & (rows < yy.min() + span * 0.83)
    distance = np.minimum(
        distance, distance_transform_edt(np.pad(eligible, 1))[1:-1, 1:-1]
    )
    score = np.where(eligible, distance, 0)
    y, x = map(int, np.unravel_index(np.argmax(score), mask.shape))
    half = min(18, int(distance[y, x] / np.sqrt(2)) - 2)
    if half < 5:
        return None
    box = [x - half, y - half, x + half + 1, y + half + 1]
    patch = rgb[box[1] : box[3], box[0] : box[2]].copy()
    return patch, box


def rear_hair_swatches(folder, completion, frames, semantics=None):
    """Return side-specific fallback textures and source provenance."""
    from scripts.hair_appearance import annotated_hair_support

    chosen = {}
    audit = {'estimatedAppearance': True, 'sources': {}}
    for view in (completion or {}).get('views', []):
        name = view['filename']
        frame = frames.get(name, {})
        yaw = frame.get('cameraYaw', frame.get('yaw') or 0)
        if abs(yaw) < 45:
            continue
        path = folder / 'images' / name
        if not path.exists():
            continue
        pixels = np.asarray(Image.open(path).convert('RGBA'))
        h, w = pixels.shape[:2]
        crop = np.asarray(completion['crops'][name])
        mask = np.zeros((h, w), np.uint8)
        for poly in view.get('hairRegions', []):
            if len(poly) >= 3:
                cv2.fillPoly(
                    mask,
                    [
                        np.rint(
                            np.asarray(poly) * (crop[2:] - crop[:2]) + crop[:2]
                        ).astype(np.int32)
                    ],
                    1,
                )
        mask &= pixels[:, :, 3] > 200
        # Explicit ear/eyewear outlines exclude them even when a coarse hair
        # polygon crosses the auricle or a glasses arm.
        semantic = next(
            (v for v in (semantics or {}).get('views', []) if v['filename'] == name),
            None,
        )
        if semantic:
            scrop = np.asarray(semantics['crops'][name], float)
            # Semantics use native detail images; hair uses registered images.
            detail_path = folder / 'detail-images' / name
            factor = Image.open(detail_path).width / w if detail_path.exists() else 1
            scrop /= factor
            excluded = list(semantic.get('opaqueGlassesRegions', []))
            excluded += [
                semantic[k]['outline']
                for k in ('imageLeftEar', 'imageRightEar')
                if semantic.get(k, {}).get('visible')
            ]
            for poly in excluded:
                if len(poly) >= 3:
                    cv2.fillPoly(
                        mask,
                        [
                            np.rint(
                                np.asarray(poly) * (scrop[2:] - scrop[:2]) + scrop[:2]
                            ).astype(np.int32)
                        ],
                        0,
                    )
        rgb = pixels[:, :, :3] / 255.0
        mask &= annotated_hair_support(rgb, mask) > 0.4
        result = short_hair_patch(rgb, mask)
        if result is None:
            continue
        patch, box = result
        key = 'rear' if abs(yaw) > 115 else ('left' if yaw < 0 else 'right')
        # Prefer larger clean samples. Frequency stays tied to a ~24 mm
        # short-hair patch, rather than stretching front curls around an ear.
        if key not in chosen or patch.shape[0] > chosen[key].shape[0]:
            chosen[key] = patch
            audit['sources'][key] = {
                'filename': name,
                'box': box,
                'cameraYaw': float(yaw),
            }
    return chosen, audit
