from typing import List, Callable, Union

import pytest

from pybss.bss.iva import GradLaplaceIVA, NaturalGradLaplaceIVA
from pybss.transforms import PCA
from tests.bss.create_dataset import set_seed, create_sisec2011_mird_spectrograms

root = "./tests/.data/SiSEC2011+MIRD"
sisec2011_root = "./tests/.data/SiSEC2011"
mird_root = "./tests/.data/MIRD"
tag = "dev1_female3"

n_fft, hop_length = 4096, 2048
window = "hann"
ref_id = 0
n_iter = 5


def dummy_function(_) -> None:
    pass


class DummyCallback:
    def __init__(self) -> None:
        pass

    def __call__(self, _) -> None:
        pass


parameters = [
    (None, False),
    (dummy_function, False),
    ([DummyCallback(), dummy_function], True),
]


@pytest.mark.parametrize("callbacks, should_initialize_demix_filter", parameters)
def test_grad_iva(
    callbacks: Union[Callable, List[Callable]], should_initialize_demix_filter: bool,
) -> None:
    set_seed()

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

    iva = GradLaplaceIVA(step_size=1e-1, callbacks=callbacks)

    if should_initialize_demix_filter:
        pca = PCA()
        _ = pca(spectrogram_mix)

        spectrogram_est = iva(spectrogram_mix, n_iter=n_iter, demix_filter=pca.proj_matrix)
    else:
        spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape, "Invalid shape."

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(iva.loss)
    plt.savefig("loss.png", bbox_inches="tight")
    plt.close()


@pytest.mark.parametrize("callbacks, should_initialize_demix_filter", parameters)
def test_natural_grad_iva(
    callbacks: Union[Callable, List[Callable]], should_initialize_demix_filter: bool,
) -> None:
    set_seed()

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

    iva = NaturalGradLaplaceIVA(step_size=1e-1, callbacks=callbacks)

    if should_initialize_demix_filter:
        pca = PCA()
        _ = pca(spectrogram_mix)

        spectrogram_est = iva(spectrogram_mix, n_iter=n_iter, demix_filter=pca.proj_matrix)
    else:
        spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape, "Invalid shape."
