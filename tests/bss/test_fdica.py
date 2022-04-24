import os

import numpy as np
from scipy.io import loadmat, wavfile
import scipy.signal as ss

from pybss.bss import GradLaplaceFDICA


def _set_seed(seed=42):
    np.random.seed(seed)


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


def _create_dataset(root="./tests/.data/SiSEC2011", tag="dev1_female3"):
    source_paths = [
        os.path.join(root, "{}_src_1.wav".format(tag)),
        os.path.join(root, "{}_src_2.wav".format(tag)),
    ]
    rir_path = os.path.join(root, "{}_synthconv_130ms_5cm_filt.mat".format(tag))
    npz_path = os.path.join(root, "./source-images.npz")
    n_channels = n_sources = len(source_paths)

    if not os.path.exists(npz_path):
        source_images = _convolve_rirs(source_paths, rir_path, n_channels=n_channels)
        np.savez(npz_path, n_sources=n_sources, n_channels=n_channels, **source_images)

    return npz_path


def test_fdica(root="./tests/.data/SiSEC2011", tag="dev1_female3"):
    _set_seed()

    ref_id = 0
    n_fft, hop_length = 2048, 1024
    npz_path = _create_dataset(root, tag=tag)

    npz = np.load(npz_path)
    n_sources = npz["n_sources"]
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=0)  # (n_sources, n_channels, n_samples)
    waveform_mix = np.sum(waveform_src, axis=0)  # (n_channels, n_samples)
    waveform_src = waveform_src[:, ref_id, :]  # (n_sources, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, nperseg=n_fft, noverlap=n_fft - hop_length, axis=-1
    )

    iva = GradLaplaceFDICA(lr=1e-2)
    spectrogram_est = iva(spectrogram_mix)

    assert spectrogram_mix.shape == spectrogram_est.shape, "Invalid shape."
