"""Appearance priorities around an occluding photographed ear."""

import numpy as np


def hidden_scalp_completion(ear_hidden, total, estimated_total, scalp, parts, support):
    """A masked ear cannot override a useful unmasked surface observation.

    Multiple camera views of an ear describe an occluder, not evidence that
    the head behind it has the prior's hair color. Retain the old completion
    where no clean observation exists, and fade it with the same strongest
    physical-support threshold used by general missing-material completion.
    Never sum weak observations into a claim of reliable visible detail.
    """

    def smooth(value):
        value = np.clip(value, 0, 1)
        return value * value * (3 - 2 * value)

    hidden_ratio = np.asarray(ear_hidden) / np.maximum(
        np.asarray(total) + np.asarray(estimated_total), 1e-7
    )
    return (
        smooth((hidden_ratio - 0.2) / 0.6)
        * scalp
        * (np.asarray(parts) < 3)
        * (1 - smooth(np.asarray(support) / 0.12))
    )
