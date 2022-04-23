import os

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile, loadmat


def set_seed(seed=42):
    np.random.seed(seed)


def _resample_mird_rir(rir_path, sample_rate_out):
    sample_rate_in = 48000
    rir_mat = loadmat(rir_path)
    rir = rir_mat["impulse_response"]
    rir_resampled = ss.resample_poly(rir, sample_rate_out, sample_rate_in, axis=0)

    return rir_resampled.T


def _convolve_rir(source_path, rir, n_channels=2):
    _, source = wavfile.read(source_path)
    source = source / 2 ** 15
    n_samples = len(source)

    source_image = []

    for mic_idx in range(n_channels):
        _source_image = np.convolve(source, rir[mic_idx])
        source_image.append(_source_image[:n_samples])

    source_image = np.stack(source_image, axis=0)  # (n_channels, n_samples)

    return source_image


def _convolve_sisec2011_rirs(source_paths, rir_path, n_channels=2):
    rir_mat = loadmat(rir_path)
    rir = rir_mat["A"]

    source_images = {}

    for src_idx, source_path in enumerate(source_paths):
        source_images["src_{}".format(src_idx + 1)] = _convolve_rir(
            source_path, rir=rir[:, src_idx, :], n_channels=n_channels
        )

    return source_images


def _convolve_mird_rirs(source_paths, rir_paths, channels=[3, 4], n_samples=None):
    assert n_samples is not None, "Specify `n_samples`."

    source_images = {}

    for src_idx, (source_path, rir_path) in enumerate(zip(source_paths, rir_paths)):
        rir = _resample_mird_rir(rir_path, sample_rate_out=16000)

        source_images["src_{}".format(src_idx + 1)] = _convolve_rir(
            source_path, rir=rir[channels, :n_samples], n_channels=len(channels)
        )

    return source_images


def create_sisec2011_dataset(root="./tests/.data/SiSEC2011", tag="dev1_female3"):
    source_paths = [
        os.path.join(root, "{}_src_1.wav".format(tag)),
        os.path.join(root, "{}_src_2.wav".format(tag)),
    ]
    rir_path = os.path.join(root, "{}_synthconv_130ms_5cm_filt.mat".format(tag))
    npz_path = os.path.join(root, "./source-images.npz")
    n_channels = n_sources = len(source_paths)

    if not os.path.exists(npz_path):
        source_images = _convolve_sisec2011_rirs(
            source_paths, rir_path, n_channels=n_channels
        )
        np.savez(npz_path, n_sources=n_sources, n_channels=n_channels, **source_images)

    return npz_path


def create_sisec2011_mird_dataset(
    root="./tests/.data/SiSEC2011+MIRD",
    sisec2011_root="./tests/.data/SiSEC2011",
    mird_root="./tests/.data/MIRD",
    tag="dev1_female3",
    n_channels=3,
):
    sample_rate = 16000
    duration = 0.160
    n_samples = int(sample_rate * duration)

    source_paths = [
        os.path.join(sisec2011_root, "{}_src_1.wav".format(tag)),
        os.path.join(sisec2011_root, "{}_src_2.wav".format(tag)),
        os.path.join(sisec2011_root, "{}_src_3.wav".format(tag)),
    ]
    rir_paths = [
        os.path.join(
            mird_root,
            "Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3_1m_030.mat",  # noqa: E501
        ),
        os.path.join(
            mird_root,
            "Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3_1m_345.mat",  # noqa: E501
        ),
        os.path.join(
            mird_root,
            "Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3_1m_000.mat",  # noqa: E501
        ),
    ]
    channels = [3, 4, 2, 5]

    source_paths = source_paths[:n_channels]
    rir_paths = rir_paths[:n_channels]
    channels = channels[:n_channels]

    n_sources = len(source_paths)
    npz_path = os.path.join(root, "./source-images-{}ch.npz".format(n_channels))

    assert n_channels == n_sources, "Mixing system should be determined."

    if not os.path.exists(npz_path):
        source_images = _convolve_mird_rirs(
            source_paths, rir_paths, channels=channels, n_samples=n_samples
        )

        os.makedirs(root, exist_ok=True)
        np.savez(npz_path, n_sources=n_sources, n_channels=n_channels, **source_images)

    return npz_path


def create_sisec2011_mird_spectrograms(
    root="./tests/.data/SiSEC2011+MIRD",
    sisec2011_root="./tests/.data/SiSEC2011",
    mird_root="./tests/.data/MIRD",
    tag="dev1_female3",
    n_fft=4096,
    hop_length=2048,
    window="hann",
    ref_id=0,
):
    npz_path = create_sisec2011_mird_dataset(
        root, sisec2011_root=sisec2011_root, mird_root=mird_root, tag=tag
    )
    npz = np.load(npz_path)
    n_sources = npz["n_sources"]
    waveform_src = []

    for src_idx in range(n_sources):
        waveform_src.append(npz["src_{}".format(src_idx + 1)])

    waveform_src = np.stack(waveform_src, axis=1)  # (n_channels, n_sources, n_samples)
    waveform_mix = np.sum(waveform_src, axis=1)  # (n_channels, n_samples)
    waveform_src = waveform_src[ref_id, :, :]  # (n_sources, n_samples)

    _, _, spectrogram_mix = ss.stft(
        waveform_mix, nperseg=n_fft, noverlap=n_fft - hop_length, window=window,
    )

    return spectrogram_mix


if __name__ == "__main__":
    create_sisec2011_mird_dataset()
