"""Choose clear overlapping views using measured image quality and camera angle."""

import json, hashlib
import numpy as np, cv2
from PIL import Image
from face_pipeline import atomic


def assess_frames(folder, output_folder=None):
    frames = json.loads((folder / 'capture.json').read_text())['frames']
    digest = hashlib.sha256(b'frame-evidence-v1')
    images = {}
    for frame in frames:
        path = folder / 'images' / frame['filename']
        raw = path.read_bytes()
        digest.update(raw)
    signature = digest.hexdigest()
    cache = (folder if output_folder is None else output_folder) / 'frame-evidence.json'
    previous = folder / 'frame-evidence.json'
    if previous.exists():
        result = json.loads(previous.read_text())
        if result.get('inputHash') == signature:
            if cache != previous:
                atomic(cache, result)
            return result['frames']
    for frame in frames:
        with Image.open(folder / 'images' / frame['filename']) as image:
            rgba = np.array(image.convert('RGBA'))
            box = (
                image.getchannel('A').getbbox()
                if image.mode == 'RGBA'
                else (0, 0, image.width, image.height)
            )
        if box is None:
            images[frame['filename']] = {
                'quality': 0,
                'sharpness': 0,
                'foregroundPixels': 0,
            }
            continue
        x0, y0, x1, y1 = box
        cut = cv2.resize(rgba[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA)
        mask = cv2.erode(np.uint8(cut[:, :, 3] > 220), np.ones((5, 5), np.uint8)) > 0
        gray = cv2.cvtColor(cut[:, :, :3], cv2.COLOR_RGB2GRAY)
        edge = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness = float(np.mean(edge[mask] ** 2)) if mask.any() else 0
        clipped = (
            float(np.mean((gray[mask] < 8) | (gray[mask] > 248))) if mask.any() else 1
        )
        quality = float(
            np.clip(np.log1p(sharpness) / np.log(501), 0, 1) * (1 - clipped * 0.7)
        )
        images[frame['filename']] = {
            'quality': quality,
            'sharpness': sharpness,
            'clippedFraction': clipped,
            'foregroundPixels': int(np.sum(rgba[:, :, 3] > 220)),
        }
    atomic(
        cache,
        {
            'version': 1,
            'inputHash': signature,
            'frames': images,
            'method': (
                'Scale-normalized foreground sharpness, clipped exposure and '
                'registered camera angular coverage. Does not certify identity '
                'or anatomy.'
            ),
        },
    )
    return images


def choose_views(views, frames, targets, quality):
    if not views:
        raise ValueError('No registered views are available.')
    selected = set()
    for angle in targets:
        distances = {
            im.name: abs(
                (
                    frames[im.name].get('cameraYaw', frames[im.name].get('yaw') or 0)
                    - angle
                    + 180
                )
                % 360
                - 180
            )
            for im in views
        }
        nearest = min(distances.values())
        candidates = [im for im in views if distances[im.name] <= nearest + 8]
        best = max(
            candidates,
            key=lambda im: quality.get(im.name, {}).get('quality', 0)
            - 0.012 * (distances[im.name] - nearest),
        )
        selected.add(best.name)
    return selected
