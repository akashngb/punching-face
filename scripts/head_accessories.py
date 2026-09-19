"""Fit independent eyeglass geometry and remove its projected ink from skin."""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from PIL import Image
import cv2
from scripts.photo_detail import brow_mask

PATH_KEYS = (
    'imageLeftLens',
    'imageRightLens',
    'bridge',
    'imageLeftTemple',
    'imageRightTemple',
)


def paths_pixels(view, crop):
    left, top, right, bottom = crop
    return {
        k: np.asarray(view[k], dtype=float) * [right - left, bottom - top] + [left, top]
        for k in PATH_KEYS
        if len(view.get(k, [])) >= 2
    }


def semantic_mask(shape, filename, semantics, key='opaqueGlassesRegions'):
    """Rasterize only validated polygons from this exact source image."""
    mask = np.zeros(shape[:2], np.uint8)
    if not semantics:
        return mask
    view = next((v for v in semantics['views'] if v['filename'] == filename), None)
    if not view or view.get('accessoryConfidence', 0) < 0.65:
        return mask
    crop = np.array(semantics['crops'][filename])
    extent = crop[2:] - crop[:2]
    for polygon in view.get(key, []):
        uv = np.asarray(polygon, float)
        if len(uv) < 3 or not np.isfinite(uv).all() or np.any((uv < 0) | (uv > 1)):
            continue
        points = np.rint(uv * extent + crop[:2]).astype(np.int32)
        # A ribbon must not consume a lens or a substantial part of the face.
        if cv2.contourArea(points) > np.prod(extent) * 0.08:
            continue
        cv2.fillPoly(mask, [points], 255)
    # Annotation contours are approximate. Include their native-pixel margin
    # and the frame's antialias/shadow edge, otherwise inpainting samples the
    # surviving black rim immediately outside the mask and paints it back in.
    margin = max(3, int(round(float(extent[0]) * 0.028)) | 1)
    return cv2.dilate(mask, np.ones((margin, margin), np.uint8))


def clean_view(
    pixels,
    filename,
    spec,
    frame,
    frames=None,
    return_details=False,
    projected_mask=None,
    semantics=None,
):
    def result(value, mask=None, method="not-detected"):
        if mask is None:
            mask = np.zeros(pixels.shape[:2], np.uint8)
        audit = {
            "method": method,
            "excludedPixels": int(np.count_nonzero(mask)),
            "hiddenSurfaceEstimated": bool(mask.any()),
            "preservedEyePhotographs": True,
        }
        return (value, mask, audit) if return_details else value

    if not spec or not spec['glasses']['present']:
        return result(pixels)
    measured = semantic_mask(pixels.shape, filename, semantics)
    view = next((v for v in spec['views'] if v['filename'] == filename), None)
    if (
        (not view or not any(len(view.get(k, [])) >= 2 for k in PATH_KEYS))
        and frames
        and frame.get('landmarks')
    ):
        candidates = [
            v
            for v in spec['views']
            if frames.get(v['filename'], {}).get('landmarks')
            and any(len(v.get(k, [])) >= 2 for k in PATH_KEYS)
        ]
        if candidates:
            view = min(
                candidates,
                key=lambda v: abs(
                    (frames[v['filename']].get('yaw') or 0) - (frame.get('yaw') or 0)
                ),
            )
    if not view and not measured.any():
        return result(pixels, method='no-visible-contour')
    rgb = pixels[:, :, :3].copy()
    mask = np.zeros(rgb.shape[:2], np.uint8)
    paths = paths_pixels(view, spec['crops'][view['filename']])
    if view['filename'] != filename:
        ids = [
            33,
            133,
            159,
            145,
            263,
            362,
            386,
            374,
            168,
            6,
            70,
            300,
            105,
            334,
            107,
            336,
        ]
        h, w = rgb.shape[:2]

        def xy(f):
            return np.array(
                [[f['landmarks'][i]['x'] * w, f['landmarks'][i]['y'] * h] for i in ids]
            )

        matrix, _ = cv2.findHomography(
            xy(frames[view['filename']]), xy(frame), cv2.RANSAC, 3.0
        )
        if matrix is None:
            return result(pixels)
        paths = {
            k: cv2.perspectiveTransform(v[None].astype(np.float64), matrix)[0]
            for k, v in paths.items()
        }
    # Fill outside-alpha RGB first, so the cutout background cannot bleed into
    # an inpainted temple at the head silhouette.
    valid = pixels[:, :, 3] > 128
    if valid.any():
        _, nearest = distance_transform_edt(~valid, return_indices=True)
        rgb[~valid] = rgb[nearest[0][~valid], nearest[1][~valid]]
    # Narrow geometric masks remove only the actual opaque frame; the eyes,
    # eyebrows and all other measured identity features remain in the photo.
    eye_width = (
        abs(frame['landmarks'][263]['x'] - frame['landmarks'][33]['x']) * rgb.shape[1]
        if frame.get('landmarks')
        else (spec['crops'][view['filename']][2] - spec['crops'][view['filename']][0])
        * 0.45
    )
    thickness = max(2, int(eye_width * 0.035))
    for name, path in paths.items():
        points = np.rint(path).astype(np.int32)
        cv2.polylines(
            mask,
            [points],
            name.endswith('Lens') and len(points) >= 8,
            255,
            thickness,
            lineType=cv2.LINE_AA,
        )
    if projected_mask is not None:
        # Approximate 3D rims must not expand a measured contour over the brow.
        nearby = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        mask = np.maximum(mask, np.minimum(projected_mask, nearby))
    protected = brow_mask(rgb.shape, frame)
    mask[protected > 0] = 0
    # Precise ribbons replace coarse centerlines rather than unioning their
    # registration errors. Genuine obscured brow pixels are marked estimated.
    if measured.any():
        mask = measured
    if not mask.any():
        return result(pixels, method='no-visible-contour')
    rgb = cv2.inpaint(rgb, mask, max(3, thickness // 2), cv2.INPAINT_TELEA)
    output = pixels.copy()
    output[:, :, :3] = rgb
    return result(
        output,
        mask,
        'astra-opaque-ribbons' if measured.any() else 'opaque-frame-inpaint',
    )


def sample_frame_colors(folder, spec):
    """Sample opaque frame centerlines into the eyewear material ONLY."""
    samples = []
    for view in spec.get('views', []):
        image = np.asarray(
            Image.open(folder / 'images' / view['filename']).convert('RGBA')
        )
        pixels = image[:, :, :3] / 255.0
        h, w = pixels.shape[:2]
        for path in paths_pixels(view, spec['crops'][view['filename']]).values():
            for a, b in zip(path[:-1], path[1:]):
                xy = (
                    a[None]
                    + np.linspace(0, 1, max(2, int(np.linalg.norm(b - a))))[:, None]
                    * (b - a)[None]
                )
                ij = np.rint(xy).astype(int)
                inside = (
                    (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
                )
                ij = ij[inside]
                ij = ij[image[ij[:, 1], ij[:, 0], 3] > 200]
                samples.extend(pixels[ij[:, 1], ij[:, 0]])
    if not samples:
        return None
    samples = np.asarray(samples)
    base = np.asarray(spec['glasses']['frameColorSrgb'])
    # Reject skin/lens pixels where an approximate path is a few pixels off.
    distance = np.linalg.norm(samples - base, axis=1)
    good = samples[distance < 0.20]
    if len(good) < 16:
        return None
    luminance = good @ np.array([0.2126, 0.7152, 0.0722])
    opaque = good[luminance <= np.quantile(luminance, 0.35)]
    return np.median(opaque, axis=0).tolist()


def fit_temple(hinge, sign, spec, rec, center, B, scale, p):
    """Lift a visible profile contour onto a bounded side plane.

    The photo supplies the vertical path and ear bend. Lateral depth remains
    estimated, with a local clearance constraint against the fitted surface.
    """
    candidates = []
    for view in spec['views']:
        im = next(
            (im for im in rec.images.values() if im.name == view['filename']), None
        )
        if im is None:
            continue
        origin = (im.projection_center() - center) @ B.T * scale
        angle = np.degrees(np.arctan2(origin[0] * sign, origin[2]))
        if not 35 < angle < 110:
            continue
        cam = rec.cameras[im.camera_id]
        pose = im.cam_from_world()
        for key, xy in paths_pixels(view, spec['crops'][im.name]).items():
            if not key.endswith('Temple') or len(xy) < 3:
                continue
            rays = (
                np.column_stack([cam.cam_from_img(xy), np.ones(len(xy))])
                @ pose.rotation.matrix()
                @ B.T
            )
            if np.any(abs(rays[:, 0]) < 0.15):
                continue
            sides = np.full(len(xy), hinge[0])
            for _ in range(6):
                q = origin + rays * ((sides - origin[0]) / rays[:, 0])[:, None]
                for j, point in enumerate(q):
                    section = p[
                        (np.abs(p[:, 1] - point[1]) < 0.006)
                        & (np.abs(p[:, 2] - point[2]) < 0.008)
                        & (p[:, 0] * sign > 0)
                    ]
                    surface = (
                        float(np.quantile(section[:, 0] * sign, 0.95))
                        if len(section)
                        else abs(hinge[0])
                    )
                    sides[j] = sign * max(abs(hinge[0]), surface + 0.002)
            q = origin + rays * ((sides - origin[0]) / rays[:, 0])[:, None]
            if np.linalg.norm(q[-1] - hinge) < np.linalg.norm(q[0] - hinge):
                q = q[::-1]
            error = np.linalg.norm(q[0] - hinge)
            if error > 0.035 or q[0, 2] - q[-1, 2] < 0.045 or q[-1, 2] < -0.2:
                continue
            # A short hinge transition meets the observed arm without a kink.
            ordered = [hinge]
            for point in q[1:]:
                if point[2] < ordered[-1][2] - 0.003:
                    ordered.append(point)
            if len(ordered) < 3:
                continue
            candidates.append(
                (abs(angle - 80) + error * 1000, np.asarray(ordered), im.name)
            )
    if candidates:
        _, q, name = min(candidates, key=lambda v: v[0])
        if q[-1, 1] > hinge[1] - 0.012:
            q = np.vstack([q, q[-1] + [sign * -0.002, -0.014, -0.012]])
        return q, {
            'method': 'profile-contour',
            'view': name,
            'lateralDepthEstimated': True,
        }
    q = np.array(
        [
            hinge,
            hinge + [sign * 0.002, 0, -0.015],
            hinge + [sign * 0.005, -0.001, -0.070],
            hinge + [sign * 0.006, -0.003, -0.105],
            hinge + [sign * 0.004, -0.018, -0.120],
        ]
    )
    return q, {'method': 'estimated-ear-hook', 'lateralDepthEstimated': True}


def build_glasses(p, spec, rec, frames, center, B, transform):
    if (
        not spec
        or not spec['glasses']['present']
        or spec['glasses']['confidence'] < 0.6
    ):
        return None
    name = spec['frontFilename']
    view = next(v for v in spec['views'] if v['filename'] == name)
    paths = paths_pixels(view, spec['crops'][name])
    im = next((im for im in rec.images.values() if im.name == name), None)
    if im is None:
        return None
    cam = rec.cameras[im.camera_id]
    pose = im.cam_from_world()
    origin = (im.projection_center() - center) @ B.T * transform['scale']
    info = spec['glasses']
    half_width = max(abs(p[234, 0]), abs(p[454, 0]))
    z0 = p[168, 2] + info['bridgeClearanceMm'] * 0.001

    def lift(xy):
        rays = (
            np.column_stack([cam.cam_from_img(xy), np.ones(len(xy))])
            @ pose.rotation.matrix()
            @ B.T
        )
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        depth = np.full(len(xy), z0)
        for _ in range(5):
            q = origin + rays * ((depth - origin[2]) / rays[:, 2])[:, None]
            depth = z0 - 0.016 * (q[:, 0] / half_width) ** 2
        q = origin + rays * ((depth - origin[2]) / rays[:, 2])[:, None]
        return q

    rims = [
        lift(paths[k])
        for k in ['imageLeftLens', 'imageRightLens']
        if k in paths and len(paths[k]) >= 8
    ]
    if len(rims) != 2:
        return None
    bridge = (
        lift(paths['bridge'])
        if 'bridge' in paths
        else np.array(
            [rims[0][np.argmax(rims[0][:, 0])], rims[1][np.argmin(rims[1][:, 0])]]
        )
    )
    temples = []
    temple_fit = []
    for rim in rims:
        sign = np.sign(rim[:, 0].mean())
        outer = np.where(rim[:, 0] * sign > (rim[:, 0] * sign).max() - 0.005)[0]
        i = outer[np.argmax(rim[outer, 1])]
        hinge = rim[i].copy()
        arm, audit = fit_temple(
            hinge, sign, spec, rec, center, B, transform['scale'], p
        )
        temples.append(arm)
        temple_fit.append(audit)
    return {
        'type': 'eyeglasses',
        'version': 2,
        'source': 'Photo-fitted front and profile contours; lateral depth estimated',
        'model': spec['model'],
        'rims': [r.tolist() for r in rims],
        'bridge': bridge.tolist(),
        'temples': [t.tolist() for t in temples],
        'templeFit': temple_fit,
        'frameColor': info['frameColorSrgb'],
        'radius': info['frameRadiusMm'] * 0.001,
        'templeWidth': info['templeWidthMm'] * 0.001,
        'lensTint': info['lensTint'],
        'estimated': True,
        'description': info['description'],
    }
