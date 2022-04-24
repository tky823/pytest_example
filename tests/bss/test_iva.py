import scipy.signal as ss

from pybss.bss import GradLaplaceIVA, NaturalGradLaplaceIVA
from tests.bss.create_dataset import set_seed, create_sisec2011_mird_spectrograms


def test_grad_iva(
    root="./tests/.data/SiSEC2011+MIRD",
    sisec2011_root="./tests/.data/SiSEC2011",
    mird_root="./tests/.data/MIRD",
    tag="dev1_female3",
):
    set_seed()

    n_fft, hop_length = 4096, 2048
    window = "hann"
    ref_id = 0

    spectrogram_mix = create_sisec2011_mird_spectrograms(
        root,
        sisec2011_root=sisec2011_root,
        mird_root=mird_root,
        tag=tag,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        ref_id=ref_id,
    )

    n_sources = spectrogram_mix.shape[0]

    iva = GradLaplaceIVA(lr=1e-1)
    spectrogram_est = iva(spectrogram_mix)

    assert spectrogram_mix.shape == spectrogram_est.shape, "Invalid shape."

    _, waveform_est = ss.istft(
        spectrogram_est, nperseg=n_fft, noverlap=n_fft - hop_length, window=window
    )

    import os
    from scipy.io import wavfile
    import matplotlib.pyplot as plt

    save_dir = os.path.join(root, "IVA/GradLaplaceIVA")
    os.makedirs(save_dir, exist_ok=True)

    for src_idx in range(n_sources):
        wav_path = os.path.join(save_dir, "src_{}.wav".format(src_idx + 1))
        png_path = os.path.join(save_dir, "src_{}.png".format(src_idx + 1))
        wavfile.write(
            wav_path, 16000, waveform_est[src_idx],
        )

        plt.figure()
        plt.plot(waveform_est[src_idx])
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

    png_path = os.path.join(save_dir, "loss.png")

    plt.figure()
    plt.plot(iva.loss)
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()


def test_natural_grad_iva(
    root="./tests/.data/SiSEC2011+MIRD",
    sisec2011_root="./tests/.data/SiSEC2011",
    mird_root="./tests/.data/MIRD",
    tag="dev1_female3",
):
    set_seed()

    n_fft, hop_length = 4096, 2048
    window = "hann"
    ref_id = 0

    spectrogram_mix = create_sisec2011_mird_spectrograms(
        root,
        sisec2011_root=sisec2011_root,
        mird_root=mird_root,
        tag=tag,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        ref_id=ref_id,
    )

    n_sources = spectrogram_mix.shape[0]

    iva = NaturalGradLaplaceIVA(lr=1e-1)
    spectrogram_est = iva(spectrogram_mix)

    assert spectrogram_mix.shape == spectrogram_est.shape, "Invalid shape."

    _, waveform_est = ss.istft(
        spectrogram_est, nperseg=n_fft, noverlap=n_fft - hop_length, window=window
    )

    import os
    from scipy.io import wavfile
    import matplotlib.pyplot as plt

    save_dir = os.path.join(root, "IVA/NaturalGradLaplaceIVA")
    os.makedirs(save_dir, exist_ok=True)

    for src_idx in range(n_sources):
        wav_path = os.path.join(save_dir, "src_{}.wav".format(src_idx + 1))
        png_path = os.path.join(save_dir, "src_{}.png".format(src_idx + 1))
        wavfile.write(
            wav_path, 16000, waveform_est[src_idx],
        )

        plt.figure()
        plt.plot(waveform_est[src_idx])
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

    png_path = os.path.join(save_dir, "loss.png")

    plt.figure()
    plt.plot(iva.loss)
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
