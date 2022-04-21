import os

import numpy as np
from scipy.io import loadmat, wavfile
import scipy.signal as ss

from pybss.bss import GradLaplaceIVA


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


def _create_dataset(root="./tests/.data", tag="dev1_female3"):
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


def test_iva(root="./tests/.data", tag="dev1_female3"):
    npz_path = _create_dataset(root, tag=tag)

    npz = np.load(npz_path)
    waveform_src = np.stack([npz["src_1"][0], npz["src_2"][0]], axis=0)
    waveform_mix = npz["src_1"] + npz["src_2"]
    _, _, spectrogram_src = ss.stft(waveform_src, nperseg=2048, noverlap=1024, axis=-1)
    _, _, spectrogram_mix = ss.stft(waveform_mix, nperseg=2048, noverlap=1024, axis=-1)

    print(spectrogram_src.shape, spectrogram_mix.shape)

    iva = GradLaplaceIVA(lr=0.01)
    spectrogram_est = iva(spectrogram_mix)
    print(spectrogram_est.shape)

    print(iva.loss)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(iva.loss) + 1), iva.loss)
    plt.savefig("./tests/.data/loss.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    test_iva()
