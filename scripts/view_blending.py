"""Photographic preference is separate from usable surface evidence."""

import numpy as np


def frontal_preference(central_region, front_facing, is_front, release_region=1.):
    """Retain frontal feature ownership only on front-facing surface regions.

    Full preference requires a normal within about 37 degrees of the front
    camera. It fades out by about 57 degrees, where a grazing frontal image
    cannot override clear side photographs merely because of head coordinates.
    The result is a blending preference, never a physical confidence score.
    """
    facing_weight = np.clip((np.asarray(front_facing) - 0.55) / 0.25, 0, 1)
    facing_weight *= facing_weight * (3 - 2 * facing_weight)
    # Eyes, nose, brows and lips may retain a single registered exposure even
    # on their curved sides. The caller releases the jaw/neck region separately.
    angle_gate = 1 - np.clip(release_region, 0, 1) * (1 - facing_weight)
    ownership = np.clip(central_region, 0, 1) * angle_gate
    return 1 + 30 * ownership if is_front else 1 - 0.97 * ownership
