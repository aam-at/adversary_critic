import numpy as np
from .l2_attack import CarliniL2


def to_carlini_images(x):
    x = x.transpose(0, 2, 3, 1)
    x_ca = (2 * x - 1) / 2
    return np.asarray(x_ca, dtype=np.float32)


def from_carlini_images(x_ca):
    x = (x_ca * 2 + 1) / 2
    x = x.transpose(0, 3, 1, 2)
    return np.asarray(x, dtype=np.float32)
