"""Decode an uploaded orbit once, instead of seeking through HEVC for every view."""

import base64
import math
import time
import cv2


def decode(folder):
    started = time.perf_counter()
    decoder = cv2.VideoCapture(str(folder / 'source-video'))
    try:
        fps = decoder.get(cv2.CAP_PROP_FPS)
        count = decoder.get(cv2.CAP_PROP_FRAME_COUNT)
        if not decoder.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Native video decoding unavailable; use browser decoding.')
        duration = count / fps
        if not math.isfinite(duration) or not 3 <= duration <= 300:
            raise ValueError('Use a head video between 3 seconds and 5 minutes.')
        steps = min(220, max(48, math.ceil(duration / 0.3)))
        wanted = {round(min(duration - 0.05, duration * i / steps) * fps) for i in range(steps)}
        frames = []
        index = 0
        while index <= max(wanted) and decoder.grab():
            if index in wanted:
                ok, pixels = decoder.retrieve()
                if not ok:
                    raise ValueError('A sampled video frame could not be decoded.')
                height, width = pixels.shape[:2]
                scale = min(1, 1280 / width, 960 / height)
                pixels = cv2.resize(pixels, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.png', pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok:
                    raise ValueError('A sampled frame could not be encoded.')
                frames.append({'timeSeconds': index / fps, 'image': 'data:image/png;base64,' + base64.b64encode(encoded).decode()})
            index += 1
        if len(frames) != len(wanted):
            raise ValueError('Video ended before all requested frames were decoded.')
        return {'frames': frames, 'decodeSeconds': round(time.perf_counter() - started, 3), 'decoder': 'local sequential native frames'}
    finally:
        decoder.release()
