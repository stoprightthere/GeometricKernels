"""Shared eigenfunction computations for hypercube and Hamming graphs."""

from functools import cached_property

import lab as B
import numpy as np
from scipy.special import gammaln

from geometric_kernels.lab_extras import from_numpy
from geometric_kernels.utils.special_functions import generalized_kravchuk_normalized
from geometric_kernels.utils.utils import hamming_distance


class HammingGraphEigenfunctions:
    """Shared eigenfunction computations for HypercubeGraph and HammingGraph.

    This mixin serves both :class:`~.hypercube_graph.WalshFunctions` on the
    binary hypercube and :class:`~.hamming_graph.VilenkinFunctions` on q-ary
    Hamming graphs. It evaluates their normalized Kravchuk polynomials and
    combines spectral weights with level multiplicities for kernel evaluation.
    Feature maps can use its log addition-theorem values without forming large
    multiplicities.

    Subclasses provide ``dim`` and ``num_levels``. The alphabet size is given
    by ``n_cat`` when present and defaults to two for the hypercube.
    """

    supports_log_domain = True

    @cached_property
    def log_num_eigenfunctions_per_level(self):
        """Log multiplicities, shape [L], without forming large integers."""
        levels = np.arange(self.num_levels, dtype=float)
        return (
            gammaln(self.dim + 1)
            - gammaln(levels + 1)
            - gammaln(self.dim - levels + 1)
            + levels * np.log(getattr(self, "n_cat", 2) - 1)
        )

    def _log_multiplicities(self, reference):
        return B.cast(
            B.dtype(reference),
            from_numpy(reference, self.log_num_eigenfunctions_per_level),
        )[:, None]

    def _normalized_kravchuk(self, X, X2, dtype):
        distances = B.cast(dtype, hamming_distance(X, X2))
        previous, previous_previous = None, None
        for level in range(self.num_levels):
            value = generalized_kravchuk_normalized(
                self.dim,
                level,
                distances,
                getattr(self, "n_cat", 2),
                previous,
                previous_previous,
            )
            previous_previous, previous = previous, value
            yield value

    def _weighted_outerproduct_from_level_weights(self, weights, X, X2=None):
        if X2 is None:
            X2 = X
        result = B.zeros(B.dtype(weights), X.shape[0], X2.shape[0])
        for level, value in enumerate(
            self._normalized_kravchuk(X, X2, B.dtype(weights))
        ):
            result = result + weights[level] * value
        return result

    def weighted_outerproduct(self, weights, X, X2=None, **kwargs):
        level_weights = B.exp(B.log(weights) + self._log_multiplicities(weights))
        return self._weighted_outerproduct_from_level_weights(level_weights, X, X2)

    def weighted_outerproduct_log(self, log_weights, X, X2=None, **kwargs):
        log_levels = log_weights + self._log_multiplicities(log_weights)
        level_weights = B.exp(log_levels)
        return self._weighted_outerproduct_from_level_weights(level_weights, X, X2)

    def weighted_outerproduct_diag(self, weights, X, **kwargs):
        diagonal = B.sum(B.exp(B.log(weights) + self._log_multiplicities(weights)))
        return diagonal * B.ones(B.dtype(weights), X.shape[0])

    def weighted_outerproduct_diag_log(self, log_weights, X, **kwargs):
        diagonal = B.sum(B.exp(log_weights + self._log_multiplicities(log_weights)))
        return diagonal * B.ones(B.dtype(log_weights), X.shape[0])

    def phi_product_log(self, X, X2=None, *, dtype, **kwargs):
        """Return log magnitudes and signs without forming multiplicities."""
        if X2 is None:
            X2 = X
        values = B.stack(*self._normalized_kravchuk(X, X2, dtype), axis=-1)
        nonzero = values != 0
        magnitudes = B.abs(values)
        safe_magnitudes = B.where(nonzero, magnitudes, B.ones(magnitudes))
        log_magnitudes = B.log(safe_magnitudes) + B.transpose(
            self._log_multiplicities(values)
        )
        log_magnitudes = B.where(nonzero, log_magnitudes, float("-inf"))
        return log_magnitudes, values / safe_magnitudes
