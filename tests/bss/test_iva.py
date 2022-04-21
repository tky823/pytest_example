import os

import numpy as np
from scipy.io import loadmat, wavfile


def _convolve_rir(source_path, rir, n_channels=2):
    _, source = wavfile.read(source_path)
    n_samples = len(source)

    source_image = []

    for mic_idx in range(n_channels):
        _source_image = np.convolve(source, rir[mic_idx])
        source_image.append(_source_image[:n_samples])

    source_image = np.stack(source_image, axis=0)  # (n_channels, n_samples)

    return source_image


def _convolve_rirs(source_paths, rir_path, n_channels=2):
    rir_mat = loadmat(rir_path)
    rir = rir_mat["A"]

    source_images = {}

    for src_idx, source_path in enumerate(source_paths):
        source_images["src_{}".format(src_idx + 1)] = _convolve_rir(
            source_path, rir=rir[:, src_idx, :], n_channels=n_channels
        )

    return source_images


def test_iva(root="./tests/.data", tag="dev1_female3"):
    source_paths = [
        os.path.join(root, "{}_src_1.wav".format(tag)),
        os.path.join(root, "{}_src_2.wav".format(tag)),
    ]
    rir_path = os.path.join(root, "{}_synthconv_130ms_5cm_filt.mat".format(tag))
    n_channels = len(source_paths)

    source_images = _convolve_rirs(source_paths, rir_path, n_channels=n_channels)

    np.savez(os.path.join(root, "./source-images.npz"), **source_images)


if __name__ == "__main__":
    test_iva()
