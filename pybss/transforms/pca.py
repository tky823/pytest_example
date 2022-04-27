import numpy as np


class PCA:
    """Principal component analysis

    Attributes:
        proj_matrix (``numpy.ndarray``):
            (n_bins, n_sources, n_channels)
    """

    def __init__(self) -> None:
        self.proj_matrix = None

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Args:
            input (``numpy.ndarray``):
                (n_channels, n_bins, n_frames)

        Returns:
            ``numpy.ndarray``:
                (n_sources, n_bins, n_frames)
        """
        X = input.transpose(1, 2, 0)  # (n_bins, n_frames, n_channels)

        if np.iscomplexobj(input):
            XX = np.mean(
                X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj(), axis=1
            )  # (n_bins, n_channels, n_channels)
        else:
            raise NotImplementedError

        _, W = np.linalg.eigh(XX)  # (n_bins, n_channels, n_sources)
        Y = X @ W  # (n_bins, n_frames, n_sources)
        self.proj_matrix = W.transpose(0, 2, 1)  # (n_bins, n_sources, n_channels)

        output = Y.transpose(2, 0, 1)  # (n_sources, n_bins, n_frames)

        return output
