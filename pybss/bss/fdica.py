from functools import partial
import itertools

import numpy as np

from ._flooring import max_flooring
from ..algorithm.projection_back import projection_back

EPS = 1e-12


class FDICAbase:
    def __init__(self, floor_fn=None, callbacks=None, should_record_loss=True, eps=EPS):
        if floor_fn is None:
            self.floor_fn = partial(max_flooring, threshold=eps)
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

        self.output = self.separate(X, demix_filter=W)

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

        X, W = input, self.demix_filter
        self.output = self.separate(X, demix_filter=W)

        return self.output

    def __repr__(self):
        s = "FDICA("
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        raise NotImplementedError("Implement 'update' method.")

    def separate(self, input, demix_filter):
        input = input.transpose(1, 0, 2)
        output = demix_filter @ input
        output = output.transpose(1, 0, 2)

        return output

    def solve_permutation(self):
        n_sources, n_bins = self.n_sources, self.n_bins

        permutations = list(itertools.permutations(range(n_sources)))

        W = self.demix_filter  # (n_bins, n_sources, n_chennels)
        Y = self.output  # (n_sources, n_bins, n_frames)

        P = np.abs(Y).transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        norm = np.sqrt(np.sum(P ** 2, axis=1, keepdims=True))
        norm = self.floor_fn(norm)
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
        lr=1e-1,
        reference_id=0,
        callbacks=None,
        should_apply_projection_back=True,
        should_solve_permutation=True,
        should_record_loss=True,
        eps=EPS,
    ):
        super().__init__(
            callbacks=callbacks, should_record_loss=should_record_loss, eps=eps
        )

        self.lr = lr
        self.reference_id = reference_id
        self.should_apply_projection_back = should_apply_projection_back
        self.should_solve_permutation = should_solve_permutation

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

        if self.should_solve_permutation:
            self.solve_permutation()

        if self.should_apply_projection_back:
            self.apply_projection_back()

        return self.output

    def apply_projection_back(self):
        assert (
            self.should_apply_projection_back
        ), "Set should_apply_projection_back True."

        reference_id = self.reference_id
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        self.output = Y * scale[..., np.newaxis]  # (n_sources, n_bins, n_frames)

    def __repr__(self):
        s = "GradFDICA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")


class GradLaplaceFDICA(GradFDICAbase):
    def __init__(
        self,
        lr=1e-1,
        reference_id=0,
        callbacks=None,
        should_solve_permutation=True,
        should_record_loss=True,
        eps=EPS,
    ):
        super().__init__(
            lr=lr,
            reference_id=reference_id,
            callbacks=callbacks,
            should_solve_permutation=should_solve_permutation,
            should_record_loss=should_record_loss,
            eps=eps,
        )

    def update_once(self):
        n_frames = self.n_frames
        lr = self.lr

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        X_Hermite = X.transpose(1, 2, 0).conj()  # (n_bins, n_frames, n_sources)
        W_inverse = np.linalg.inv(W)
        W_inverseHermite = W_inverse.transpose(
            0, 2, 1
        ).conj()  # (n_bins, n_channels, n_sources)

        Y = Y.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        denominator = np.abs(Y)
        denominator = self.floor_fn(denominator)
        Phi = Y / denominator  # (n_bins, n_sources, n_frames)

        delta = (Phi @ X_Hermite) / n_frames - W_inverseHermite
        W = W - lr * delta  # (n_bins, n_sources, n_channels)

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y

    def __repr__(self):
        s = "GradLaplaceFDICA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)

    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        logdetW = np.log(np.abs(np.linalg.det(W)))
        loss = 2 * np.abs(Y).sum(axis=0).mean(axis=1) - 2 * logdetW

        return loss


class NaturalGradLaplaceFDICA(GradFDICAbase):
    def __init__(
        self,
        lr=1e-1,
        reference_id=0,
        is_holonomic=True,
        callbacks=None,
        should_solve_permutation=True,
        should_record_loss=True,
        eps=EPS,
    ):
        super().__init__(
            lr=lr,
            reference_id=reference_id,
            callbacks=callbacks,
            should_solve_permutation=should_solve_permutation,
            should_record_loss=should_record_loss,
            eps=eps,
        )

        self.is_holonomic = is_holonomic

    def __repr__(self):
        s = "GradLaplaceFDICA("
        s += "lr={lr}"
        s += ", is_holonomic={is_holonomic}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_frames = self.n_frames
        lr = self.lr

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        eye = np.eye(n_sources, n_channels, dtype=np.complex128)

        Y = Y.transpose(1, 0, 2)  # (n_bins, n_sources, n_frames)
        Y_Hermite = Y.transpose(0, 2, 1).conj()  # (n_bins, n_frames, n_sources)
        denominator = np.abs(Y)
        denominator = self.floor_fn(denominator)
        Phi = Y / denominator  # (n_bins, n_sources, n_frames)

        if self.is_holonomic:
            delta = ((Phi @ Y_Hermite) / n_frames - eye) @ W
        else:
            raise NotImplementedError("only suports for is_holonomic = True")

        W = W - lr * delta  # (n_bins, n_sources, n_channels)

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.output = Y

    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        logdetW = np.abs(np.linalg.det(W))
        loss = 2 * np.abs(Y).sum(axis=0).mean(axis=1) - 2 * logdetW
        loss = loss.sum()

        return loss
