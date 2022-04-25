from typing import List, Callable, Union

import pytest

from pybss.bss.iva import GradLaplaceIVA, NaturalGradLaplaceIVA
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


callbacks = [None, dummy_function, [DummyCallback(), dummy_function]]


@pytest.mark.parametrize("callbacks", callbacks)
def test_grad_iva(callbacks: Union[Callable, List[Callable]]) -> None:
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
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape, "Invalid shape."


@pytest.mark.parametrize("callbacks", callbacks)
def test_natural_grad_iva(callbacks: Union[Callable, List[Callable]]) -> None:
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
    spectrogram_est = iva(spectrogram_mix, n_iter=n_iter)

    assert spectrogram_mix.shape == spectrogram_est.shape, "Invalid shape."
