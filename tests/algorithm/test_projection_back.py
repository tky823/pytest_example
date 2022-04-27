import numpy as np

from pybss.algorithm import projection_back
from tests.bss.create_dataset import set_seed


def _create_dummpy_spectrogram(*args):
    return np.random.randn(*args) + 1j * np.random.randn(*args)


def test_projection_back():
    set_seed()

    n_sources, n_bins, n_frames = 3, 1025, 100

    estimated = _create_dummpy_spectrogram(n_sources, n_bins, n_frames)
    reference_single = _create_dummpy_spectrogram(n_bins, n_frames)
    reference_multi = _create_dummpy_spectrogram(n_sources, n_bins, n_frames)

    scale_single = projection_back(estimated, reference_single)
    target_shape = estimated.shape[:1] + reference_single.shape[:-1]

    assert scale_single.shape == target_shape, "{} != {}".format(
        scale_single.shape, target_shape
    )

    scale_multi = projection_back(estimated, reference_multi)
    target_shape = estimated.shape[:1] + reference_multi.shape[:-1]

    assert scale_multi.shape == target_shape, "{} != {}".format(
        scale_multi.shape, target_shape
    )
