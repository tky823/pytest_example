from functools import partial
import itertools
from typing import List, Callable, Optional, Union

import numpy as np

from ._flooring import max_flooring
from ..algorithm.projection_back import projection_back

STEP_SIZE = 1e-1
EPS = 1e-12

__all__ = ["GradLaplaceFDICA", "NaturalGradLaplaceFDICA"]


class FDICAbase:
    def __init__(
        self,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = partial(
            max_flooring, threshold=EPS
        ),
        callbacks: Optional[List[Callable[["FDICAbase"], None]]] = None,
        should_record_loss: Optional[bool] = True,
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
        """Separate multichannel signal by FDICA.

        Args:
            input (``numpy.ndarray``):
                Mixture spectrogram with shape of (n_channels, n_bins, n_frames).
            n_iter (``Optional[int]``):
                Number of iterations. Default: ``100``.

        Returns:
            ``numpy.ndarray``:
                Separated spectrogram with shape of (n_channels, n_bins, n_frames).
        """
        self.input = input

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
        s = "FDICA("
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        raise NotImplementedError("Implement 'update' method.")

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

    def solve_permutation(self) -> None:
        n_sources, n_bins = self.n_sources, self.n_bins

        permutations = list(itertools.permutations(range(n_sources)))

        W = self.demix_filter  # (n_bins, n_sources, n_chennels)
        Y = self.output  # (n_sources, n_bins, n_frames)

        P = np.abs(Y).transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        norm = np.sqrt(np.sum(P ** 2, axis=1, keepdims=True))
        norm = self.flooring_fn(norm)
        P = P / norm  # (n_bins, n_sources, n_frames)
        correlation = np.sum(P @ P.transpose(0, 2, 1), axis=(1, 2))  # (n_sources,)
        indices = np.argsort(correlation)

        min_idx = indices[0]
        P_criteria = P[min_idx]  # (n_sources, n_frames)

        for idx in range(1, n_bins):
            min_idx = indices[idx]
            P_max = None
            perm_max = None

            for perm in permutations:
                P_perm = np.sum(P_criteria * P[min_idx, perm, :])
                if P_max is None or P_perm > P_max:
                    P_max = P_perm
                    perm_max = perm

            P_criteria = P_criteria + P[min_idx, perm_max, :]
            W[min_idx, :, :] = W[min_idx, perm_max, :]

        self.demix_filter = W

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")


class GradFDICAbase(FDICAbase):
    def __init__(
        self,
        step_size: Optional[float] = STEP_SIZE,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = partial(
            max_flooring, threshold=EPS
        ),
        reference_id: Optional[int] = 0,
        callbacks: Optional[
            Union[
                Callable[["GradFDICAbase"], None],
                List[Callable[["GradFDICAbase"], None]],
            ]
        ] = None,
        should_apply_projection_back: Optional[bool] = True,
        should_solve_permutation: Optional[bool] = True,
        should_record_loss: Optional[bool] = True,
    ) -> None:
        super().__init__(
            flooring_fn=flooring_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
        )

        self.step_size = step_size
        self.reference_id = reference_id
        self.should_apply_projection_back = should_apply_projection_back
        self.should_solve_permutation = should_solve_permutation

    def __call__(
        self, input: np.ndarray, n_iter: Optional[int] = 100, **kwargs
    ) -> np.ndarray:
        """Separate multichannel signal by FDICA using gradient descent.

        Args:
            input (``numpy.ndarray``):
                Mixture spectrogram with shape of (n_channels, n_bins, n_frames).
            n_iter (``Optional[int]``):
                Number of iterations. Default: ``100``.

        Returns:
            ``numpy.ndarray``:
                Separated spectrogram with shape of (n_channels, n_bins, n_frames).
        """
        self.input = input

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

        if self.should_solve_permutation:
            self.solve_permutation()

        if self.should_apply_projection_back:
            self.apply_projection_back()

        return self.output

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

    def __repr__(self) -> str:
        s = "GradFDICA("
        s += "step_size={step_size}"
        s += ")"

        return s.format(**self.__dict__)


class GradLaplaceFDICA(GradFDICAbase):
    """Frequency-domain independent component analysis
    on Laplacian distribution using gradient descent [1]_.

    Args:
        step_size (``Optional[float]``):
            Step size of gradient descent. Default: ``{STEP_SIZE}``.
        flooring_fn (``Optional[Callable[[numpy.ndarray], numpy.ndarray]]``):
            Flooring function for numerical stability. Default: None.
        reference_id (``Optional[int]``):
            Reference microphone index for projection back. Default: ``0``.
        callbacks (``Optional[List[Callable[[GradLaplaceFDICA], None]]]``):
            Callback function(s). Default: None.
        should_solve_permutation (``Optional[bool]``):
            Solve permutation after updates of demixing filters. Default: ``True``.
        should_record_loss (``Optional[bool]``)
            Record loss. Default: ``True``.

    Examples:
        >>> import soundfile as sf
        >>> import scipy.signal as ss
        >>> from pybss.bss.fdica import GradLaplaceFDICA
        >>> waveform_mix, _ = sf.read("sample-2ch.wav")
        >>> fdica = GradLaplaceFDICA()
        >>> _, _, spectrogram_mix = ss.stft(waveform_mix.T, nperseg=4096, noverlap=2048)
        >>> spectrogram_est = fdica(spectrogram_mix)
        >>> print(spectrogram_mix.shape, spectrogram_est.shape)
        (2, 2049, 80), (2, 2049, 80)

    .. rubric:: References

    .. [1] H. Sawada et al., *Underdetermined Convolutive Blind Source Separation
        via Frequency Bin-Wise Clustering and Permutation Alignment*, 2011.
    """

    def __init__(
        self,
        step_size: Optional[float] = STEP_SIZE,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = partial(
            max_flooring, threshold=EPS
        ),
        reference_id: Optional[int] = 0,
        callbacks: Optional[
            Union[
                Callable[["GradLaplaceFDICA"], None],
                List[Callable[["GradLaplaceFDICA"], None]],
            ]
        ] = None,
        should_solve_permutation: Optional[bool] = True,
        should_record_loss: Optional[bool] = True,
    ) -> None:
        super().__init__(
            step_size=step_size,
            flooring_fn=flooring_fn,
            reference_id=reference_id,
            callbacks=callbacks,
            should_solve_permutation=should_solve_permutation,
            should_record_loss=should_record_loss,
        )

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
        denominator = np.abs(Y)
        denominator = self.flooring_fn(denominator)
        Phi = Y / denominator  # (n_bins, n_sources, n_frames)

        delta = (Phi @ X_Hermite) / n_frames - W_inverseHermite
        W = W - step_size * delta  # (n_bins, n_sources, n_channels)

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y

    def __repr__(self) -> str:
        s = "GradLaplaceFDICA("
        s += "step_size={step_size}"
        s += ")"

        return s.format(**self.__dict__)

    def compute_negative_loglikelihood(self) -> float:
        """Compute negative log-likelihood.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        logdetW = np.log(np.abs(np.linalg.det(W)))
        loss = 2 * np.abs(Y).sum(axis=0).mean(axis=1) - 2 * logdetW

        return loss


class NaturalGradLaplaceFDICA(GradFDICAbase):
    """Frequency-domain independent component analysis
    on Laplacian distribution using natural gradient descent.

    Args:
        step_size (``Optional[float]``):
            Step size of gradient descent. Default: ``{STEP_SIZE}``.
        flooring_fn (``Optional[Callable[[numpy.ndarray], numpy.ndarray]]``):
            Flooring function for numerical stability. Default: ``None``.
        reference_id (``Optional[int]``):
            Reference microphone index for projection back. Default: ``0``.
        is_holonomic (``Optional[bool]``):
            Holonomic update. Default: ``True``.
        callbacks (``Optional[Union[Callable[[NaturalGradLaplaceFDICA], None], \
            List[Callable[[NaturalGradLaplaceFDICA], None]]]]``):
            Callback function(s). Default: ``None``.
        should_solve_permutation (``Optional[bool]``):
            Solve permutation after updates of demixing filters. Default: ``True``.
        should_record_loss (``Optional[bool]``):
            Record loss. Default: ``True``.

    Examples:
        >>> import soundfile as sf
        >>> import scipy.signal as ss
        >>> from pybss.bss.fdica import NaturalGradLaplaceFDICA
        >>> waveform_mix, _ = sf.read("sample-2ch.wav")
        >>> fdica = NaturalGradLaplaceFDICA()
        >>> _, _, spectrogram_mix = ss.stft(waveform_mix.T, nperseg=4096, noverlap=2048)
        >>> spectrogram_est = fdica(spectrogram_mix)
        >>> print(spectrogram_mix.shape, spectrogram_est.shape)
        (2, 2049, 80), (2, 2049, 80)
    """

    def __init__(
        self,
        step_size: Optional[float] = STEP_SIZE,
        flooring_fn: Optional[Callable[[np.ndarray], np.ndarray]] = partial(
            max_flooring, threshold=EPS
        ),
        reference_id: Optional[int] = 0,
        is_holonomic: Optional[bool] = True,
        callbacks: Optional[
            Union[
                Callable[["NaturalGradLaplaceFDICA"], None],
                List[Callable[["NaturalGradLaplaceFDICA"], None]],
            ]
        ] = None,
        should_solve_permutation: Optional[bool] = True,
        should_record_loss: Optional[bool] = True,
    ) -> None:
        super().__init__(
            step_size=step_size,
            flooring_fn=flooring_fn,
            reference_id=reference_id,
            callbacks=callbacks,
            should_solve_permutation=should_solve_permutation,
            should_record_loss=should_record_loss,
        )

        self.is_holonomic = is_holonomic

    def __repr__(self) -> str:
        s = "GradLaplaceFDICA("
        s += "step_size={step_size}"
        s += ", is_holonomic={is_holonomic}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self) -> None:
        """Update demixing filters once using natural gradient descent.
        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_frames = self.n_frames
        step_size = self.step_size

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        eye = np.eye(n_sources, n_channels, dtype=np.complex128)

        Y = Y.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        Y_Hermite = Y.transpose(0, 2, 1).conj()  # (n_bins, n_frames, n_sources)
        denominator = np.abs(Y)
        denominator = self.flooring_fn(denominator)
        Phi = Y / denominator  # (n_bins, n_sources, n_frames)

        if self.is_holonomic:
            delta = ((Phi @ Y_Hermite) / n_frames - eye) @ W
        else:
            raise NotImplementedError("only suports for is_holonomic = True")

        W = W - step_size * delta  # (n_bins, n_sources, n_channels)

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y

    def compute_negative_loglikelihood(self) -> float:
        """Compute negative log-likelihood.
        """
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        logdetW = np.abs(np.linalg.det(W))
        loss = 2 * np.abs(Y).sum(axis=0).mean(axis=1) - 2 * logdetW
        loss = loss.sum()

        return loss


GradLaplaceFDICA.__doc__ = GradLaplaceFDICA.__doc__.format(STEP_SIZE=STEP_SIZE)
NaturalGradLaplaceFDICA.__doc__ = NaturalGradLaplaceFDICA.__doc__.format(
    STEP_SIZE=STEP_SIZE
)
