import numpy as np

EPS = 1e-12


def max_flooring(x: np.ndarray, threshold=EPS):
    x[x < threshold] = threshold
    return x


def add_flooring(x: np.ndarray, threshold=EPS):
    x = x + threshold
    return x
