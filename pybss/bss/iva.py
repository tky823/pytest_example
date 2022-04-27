from typing import List, Callable, Optional, Union
from functools import partial

import numpy as np

from ._flooring import max_flooring
from ..algorithm.projection_back import projection_back

STEP_SIZE = 1e-1
EPS = 1e-12
THRESHOLD = 1e12

__all__ = ["GradLaplaceIVA", "NaturalGradLaplaceIVA"]
__algorithms_spatial__ = [
    "IP",
    "IVA",
    "ISS",
    "IPA",
    "IP1",
    "IP2",
    "ISS1",
    "ISS2",
]


class IVAbase:
    def __init__(
        self,
        flooring_fn: Optional[Callable] = partial(max_flooring, threshold=EPS),
        callbacks: Optional[
            Union[Callable[["IVAbase"], None], List[Callable[["IVAbase"], None]]]
        ] = None,
        should_record_loss: bool = True,
    ) -> None:
        if flooring_fn is None:
            self.flooring_fn = lambda x: x  # identity
        else:
            self.flooring_fn = flooring_fn

        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]

            self.callbacks = callbacks
        else:
            self.callbacks = None

        self.input = None
        self.should_record_loss = should_record_loss

        if self.should_record_loss:
            self.loss = []
        else:
            self.loss = None

    def _reset(self, **kwargs) -> None:
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels  # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        if not hasattr(self, "demix_filter"):
            W = np.eye(n_sources, n_channels, dtype=np.complex128)
            self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        else:
            # To avoid overwriting `demix_filter` given by keyword arguments.
            W = self.demix_filter.copy()
            self.demix_filter = W

        self.output = self.separate(X, demix_filter=W)

    def __call__(
        self, input: np.ndarray, n_iter: Optional[int] = 100, **kwargs
    ) -> np.ndarray:
        """Separate multichannel signal by IVA.

        Args:
            input (``numpy.ndarray``):
                Mixture spectrogram with shape of (n_channels, n_bins, n_frames).
            n_iter (``Optional[int]``):
                Number of iterations. Default: ``100``.

        Returns:
            ``numpy.ndarray``:
                Separated spectrogram with shape of (n_channels, n_bins, n_frames).
        """
        self.input = input.copy()

        self._reset(**kwargs)

        if self.should_record_loss:
            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for _ in range(n_iter):
            self.update_once()

            if self.should_record_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

        X, W = input, self.demix_filter
        self.output = self.separate(X, demix_filter=W)

        return self.output

    def __repr__(self) -> str:
        s = "IVA("
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        raise NotImplementedError("Implement 'update_once' method.")

    def separate(self, input: np.ndarray, demix_filter: np.ndarray) -> np.ndarray:
        """Separate ``input`` using ``demixing_filter``.

        Args:
            input (``numpy.ndarray``):
                Mixture spectrogram.
            demix_filter (``numpy.ndarray``):
                Demixing filters to separate signal.

        Returns:
            ``numpy.ndarray``:
                Separated spectrogram.
        """
        input = input.transpose(1, 0, 2)
        output = demix_filter @ input
        output = output.transpose(1, 0, 2)

        return output

    def compute_negative_loglikelihood(self):
        """Compute negative log-likelihood.
        """
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")


class GradIVAbase(IVAbase):
    def __init__(
        self,
        step_size: Optional[float] = STEP_SIZE,
        flooring_fn: Optional[Callable] = partial(max_flooring, threshold=EPS),
        reference_id: Optional[int] = 0,
        callbacks: Optional[
            Union[
                Callable[["GradIVAbase"], None], List[Callable[["GradIVAbase"], None]],
            ]
        ] = None,
        should_apply_projection_back: bool = True,
        should_record_loss: bool = True,
        eps: float = EPS,
    ) -> None:
        super().__init__(
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
            eps=eps,
        )

        self.step_size = step_size
        self.reference_id = reference_id
        self.should_apply_projection_back = should_apply_projection_back

    def __call__(
        self, input: np.ndarray, n_iter: Optional[int] = 100, **kwargs
    ) -> np.ndarray:
        """Separate multichannel signal by IVA using gradient descent.

        Args:
            input (``numpy.ndarray``):
                Mixture spectrogram with shape of (n_channels, n_bins, n_frames).
            n_iter (``Optional[int]``):
                Number of iterations. Default: ``100``.

        Returns:
            ``numpy.ndarray``:
                Separated spectrogram with shape of (n_channels, n_bins, n_frames).
        """
        self.output = super().__call__(input, n_iter=n_iter, **kwargs)

        if self.should_apply_projection_back:
            self.apply_projection_back()

        return self.output

    def __repr__(self) -> str:
        s = "GradIVA("
        s += "step_size={step_size}"
        s += ")"

        return s.format(**self.__dict__)

    def apply_projection_back(self) -> None:
        """Apply projection back to separated signal.
        """
        assert (
            self.should_apply_projection_back
        ), "Set should_apply_projection_back True."

        reference_id = self.reference_id
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])

        self.output = Y * scale[..., np.newaxis]  # (n_sources, n_bins, n_frames)

    def update_once(self) -> None:
        raise NotImplementedError("Implement 'update_once' method.")


class GradLaplaceIVA(GradIVAbase):
    """Independent vector analysis on Laplacian distribution using gradient descent [1]_.

    Args:
        step_size (``Optional[float]``):
            Step size of gradient descent. Default: ``{STEP_SIZE}``.
        flooring_fn (``Optional[Callable[[numpy.ndarray], numpy.ndarray]]``):
            Flooring function for numerical stability. Default: None.
        reference_id (``Optional[int]``):
            Reference microphone index for projection back. Default: ``0``.
        callbacks (``Optional[List[Callable[[GradLaplaceIVA], None]]]``):
            Callback function(s). Default: None.
        should_solve_permutation (``Optional[bool]``):
            Solve permutation. Default: True,
        should_record_loss (``Optional[bool]``)
            Record loss. Default: True.
        eps (``Optional[float]``):
            Epsilon value for numerical stability. Default: ``{EPS}``.

    Examples:
        >>> import soundfile as sf
        >>> import scipy.signal as ss
        >>> from pybss.bss.iva import GradLaplaceIVA
        >>> waveform_mix, _ = sf.read("sample-2ch.wav")
        >>> iva = GradLaplaceIVA()
        >>> _, _, spectrogram_mix = ss.stft(waveform_mix.T, nperseg=4096, noverlap=2048)
        >>> spectrogram_est = iva(spectrogram_mix)
        >>> print(spectrogram_mix.shape, spectrogram_est.shape)
        (2, 2049, 80), (2, 2049, 80)

    .. [1] T. Kim et al., *Independent Vector Analysis: An Extension of ICA
        to Multivariate Components*, 2006.
    """

    def __init__(
        self,
        step_size: Optional[float] = STEP_SIZE,
        flooring_fn: Optional[Callable] = partial(max_flooring, threshold=EPS),
        reference_id: Optional[int] = 0,
        callbacks: Optional[
            Union[
                Callable[["GradLaplaceIVA"], None],
                List[Callable[["GradLaplaceIVA"], None]],
            ]
        ] = None,
        should_apply_projection_back: Optional[bool] = True,
        should_record_loss: Optional[bool] = True,
        eps: Optional[float] = EPS,
    ) -> None:
        super().__init__(
            step_size=step_size,
            flooring_fn=flooring_fn,
            reference_id=reference_id,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            eps=eps,
        )

    def __repr__(self) -> str:
        s = "GradLaplaceIVA("
        s += "step_size={step_size}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        """Update demixing filters once using gradient descent.
        """
        n_frames = self.n_frames
        step_size = self.step_size

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        X_Hermite = X.transpose(1, 2, 0).conj()  # (n_bins, n_frames, n_sources)
        W_inverse = np.linalg.inv(W)
        W_inverseHermite = W_inverse.transpose(
            0, 2, 1
        ).conj()  # (n_bins, n_channels, n_sources)

        Y = Y.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        P = np.abs(Y) ** 2
        denominator = np.sqrt(P.sum(axis=0))
        denominator = self.flooring_fn(denominator)
        Phi = Y / denominator  # (n_bins, n_sources, n_frames)

        delta = (Phi @ X_Hermite) / n_frames - W_inverseHermite
        W = W - step_size * delta  # (n_bins, n_sources, n_channels)

        X = self.input
        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y

    def compute_negative_loglikelihood(self) -> float:
        """Compute negative log-likelihood.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y) ** 2, axis=1)
        logdetW = np.log(np.abs(np.linalg.det(W))).sum()
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * logdetW

        return loss


class NaturalGradLaplaceIVA(GradIVAbase):
    """Independent vector analysis on Laplacian distribution
    using natural gradient descent.

    Args:
        step_size (``Optional[float]``):
            Step size of gradient descent. Default: ``{STEP_SIZE}``.
        flooring_fn (``Optional[Callable[[numpy.ndarray], numpy.ndarray]]``):
            Flooring function for numerical stability. Default: None.
        reference_id (``Optional[int]``):
            Reference microphone index for projection back. Default: ``0``.
        callbacks (``Optional[List[Callable[[NaturalGradLaplaceIVA], None]]]``):
            Callback function(s). Default: None.
        should_solve_permutation (``Optional[bool]``):
            Solve permutation. Default: True,
        should_record_loss (``Optional[bool]``)
            Record loss. Default: True.
        eps (``Optional[float]``):
            Epsilon value for numerical stability. Default: ``{EPS}``.

    Examples:
        >>> import soundfile as sf
        >>> import scipy.signal as ss
        >>> from pybss.bss.iva import NaturalGradLaplaceIVA
        >>> waveform_mix, _ = sf.read("sample-2ch.wav")
        >>> iva = NaturalGradLaplaceIVA()
        >>> _, _, spectrogram_mix = ss.stft(waveform_mix.T, nperseg=4096, noverlap=2048)
        >>> spectrogram_est = iva(spectrogram_mix)
        >>> print(spectrogram_mix.shape, spectrogram_est.shape)
        (2, 2049, 80), (2, 2049, 80)
    """

    def __init__(
        self,
        step_size: Optional[float] = STEP_SIZE,
        flooring_fn: Optional[Callable] = partial(max_flooring, threshold=EPS),
        reference_id: Optional[int] = 0,
        callbacks: Optional[
            Union[
                Callable[["NaturalGradLaplaceIVA"], None],
                List[Callable[["NaturalGradLaplaceIVA"], None]],
            ]
        ] = None,
        should_apply_projection_back: Optional[bool] = True,
        should_record_loss: Optional[bool] = True,
        eps: Optional[float] = EPS,
    ):
        super().__init__(
            step_size=step_size,
            flooring_fn=flooring_fn,
            reference_id=reference_id,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            eps=eps,
        )

    def __repr__(self) -> str:
        s = "NaturalGradLaplaceIVA("
        s += "step_size={step_size}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        """Update demixing filters once using gradient descent.
        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_frames = self.n_frames
        step_size = self.step_size

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        eye = np.eye(n_sources, n_channels, dtype=np.complex128)

        Y = Y.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        Y_Hermite = Y.transpose(0, 2, 1).conj()  # (n_bins, n_frames, n_sources)
        P = np.abs(Y) ** 2
        denominator = np.sqrt(P.sum(axis=0))
        denominator = self.flooring_fn(denominator)
        Phi = Y / denominator  # (n_bins, n_sources, n_frames)

        delta = ((Phi @ Y_Hermite) / n_frames - eye) @ W
        W = W - step_size * delta  # (n_bins, n_sources, n_channels)

        X = self.input
        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y

    def compute_negative_loglikelihood(self) -> float:
        """Compute negative log-likelihood.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y) ** 2, axis=1)
        logdetW = np.log(np.abs(np.linalg.det(W))).sum()
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * logdetW

        return loss


GradLaplaceIVA.__doc__ = GradLaplaceIVA.__doc__.format(STEP_SIZE=STEP_SIZE, EPS=EPS)
NaturalGradLaplaceIVA.__doc__ = NaturalGradLaplaceIVA.__doc__.format(
    STEP_SIZE=STEP_SIZE, EPS=EPS
)
