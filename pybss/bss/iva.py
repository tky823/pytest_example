from functools import partial

import numpy as np

from ..algorithm.projection_back import projection_back

EPS = 1e-12
THRESHOLD = 1e12

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


def add_flooring(x, eps=EPS):
    x[x < eps] = eps
    return x


class IVAbase:
    def __init__(self, floor_fn=None, callbacks=None, should_record_loss=True, eps=EPS):
        if floor_fn is None:
            self.floor_fn = partial(add_flooring, eps=eps)
        else:
            self.floor_fn = floor_fn

        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None

        self.eps = eps

        self.input = None
        self.should_record_loss = should_record_loss

        if self.should_record_loss:
            self.loss = []
        else:
            self.loss = None

    def _reset(self, **kwargs):
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
            W = self.demix_filter.copy()
            self.demix_filter = W
        self.estimation = self.separate(X, demix_filter=W)

    def __call__(self, input, n_iter=100, **kwargs):
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
        output = self.separate(X, demix_filter=W)

        return output

    def __repr__(self):
        s = "IVA("
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        raise NotImplementedError("Implement 'update_once' method.")

    def separate(self, input, demix_filter):
        input = input.transpose(1, 0, 2)
        estimation = demix_filter @ input
        output = estimation.transpose(1, 0, 2)

        return output

    def compute_demix_filter(self, estimation, input):
        X, Y = input, estimation
        X_Hermite = X.transpose(1, 2, 0).conj()
        XX_Hermite = X.transpose(1, 0, 2) @ X_Hermite
        demix_filter = Y.transpose(1, 0, 2) @ X_Hermite @ np.linalg.inv(XX_Hermite)

        return demix_filter

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")


class GradIVAbase(IVAbase):
    def __init__(
        self,
        lr=1e-1,
        floor_fn=None,
        reference_id=0,
        callbacks=None,
        should_apply_projection_back=True,
        should_record_loss=True,
        eps=EPS,
    ):
        super().__init__(
            floor_fn=floor_fn,
            callbacks=callbacks,
            should_record_loss=should_record_loss,
            eps=eps,
        )

        self.lr = lr
        self.reference_id = reference_id
        self.should_apply_projection_back = should_apply_projection_back

    def __call__(self, input, n_iter=100, **kwargs):
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

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        if self.should_apply_projection_back:
            scale = projection_back(output, reference=X[reference_id])
            output = output * scale[..., np.newaxis]  # (n_sources, n_bins, n_frames)

        self.estimation = output

        return output

    def __repr__(self):
        s = "GradIVA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        raise NotImplementedError("Implement 'update_once' method.")

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")


class GradLaplaceIVA(GradIVAbase):
    def __init__(
        self,
        lr=1e-1,
        floor_fn=None,
        reference_id=0,
        callbacks=None,
        should_apply_projection_back=True,
        should_record_loss=True,
        eps=EPS,
    ):
        super().__init__(
            lr=lr,
            floor_fn=floor_fn,
            reference_id=reference_id,
            callbacks=callbacks,
            should_apply_projection_back=should_apply_projection_back,
            should_record_loss=should_record_loss,
            eps=eps,
        )

    def __repr__(self):
        s = "GradLaplaceIVA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        n_frames = self.n_frames
        lr = self.lr
        eps = self.eps

        X = self.input
        W = self.demix_filter
        Y = self.separate(X, demix_filter=W)

        X_Hermite = X.transpose(1, 2, 0).conj()  # (n_bins, n_frames, n_sources)
        W_inverse = np.linalg.inv(W)
        W_inverseHermite = W_inverse.transpose(
            0, 2, 1
        ).conj()  # (n_bins, n_channels, n_sources)

        Y = Y.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        P = np.abs(Y) ** 2
        denominator = np.sqrt(P.sum(axis=0))
        denominator[denominator < eps] = eps
        Phi = Y / denominator  # (n_bins, n_sources, n_frames)

        delta = (Phi @ X_Hermite) / n_frames - W_inverseHermite
        W = W - lr * delta  # (n_bins, n_sources, n_channels)

        X = self.input
        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.estimation = Y

    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y) ** 2, axis=1)
        logdetW = np.log(np.abs(np.linalg.det(W))).sum()
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * logdetW

        return loss
