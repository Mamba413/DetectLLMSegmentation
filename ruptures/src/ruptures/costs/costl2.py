r"""CostL2 (least squared deviation)"""

from ruptures.costs import NotEnoughPoints

from ruptures.base import BaseCost

import numpy as np

class CostL2(BaseCost):
    r"""Least squared deviation."""

    model = "l2"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 1

    def fit(self, signal, weight=None) -> "CostL2":
        """Set parameters of the instance.

        Args:
            signal (array): array of shape (n_samples,) or (n_samples, n_features)

        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        
        if weight is None:
            self.weight = np.ones(self.signal.shape[0], dtype=float)
        else:
            self.weight = np.array(weight)

        return self

    def error(self, start, end) -> float:
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost

        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """
        if end - start < self.min_size:
            raise NotEnoughPoints

        X = self.signal[start:end]          # shape (n, d)
        w = self.weight[start:end]          # shape (n,)

        w_sum = w.sum()
        if w_sum == 0:
            return 0.0

        # weighted mean: shape (d,)
        mean = (X * w[:, None]).sum(axis=0) / w_sum

        # weighted SSE
        err = ((X - mean) ** 2 * w[:, None]).sum()

        return err
