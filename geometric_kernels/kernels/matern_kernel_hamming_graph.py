r"""
This module provides the :class:`MaternKernelHammingGraph` kernel, a subclass of
:class:`MaternKarhunenLoeveLogDomain` for :class:`HammingGraph` and
:class:`HypercubeGraph` spaces with log-domain spectral weighting and a
closed-form heat kernel when $\nu = \infty$.
"""

import lab as B
import numpy as np
from beartype.typing import Dict, Optional, Union

from geometric_kernels.kernels.karhunen_loeve_log_domain import (
    MaternKarhunenLoeveLogDomain,
)
from geometric_kernels.spaces.eigenfunctions import Eigenfunctions
from geometric_kernels.spaces.hamming_graph import HammingGraph
from geometric_kernels.spaces.hamming_graph_eigenfunctions import (
    HammingGraphEigenfunctions,
)
from geometric_kernels.spaces.hypercube_graph import HypercubeGraph
from geometric_kernels.utils.kernel_formulas.hamming_graph import (
    _log_hamming_graph_heat_kernel,
)


class MaternKernelHammingGraph(MaternKarhunenLoeveLogDomain):
    r"""
    For $\nu = \infty$, there exists a closed-form formula for the heat kernel
    on hamming graphs :class:`HammingGraph` (including the binary hypercube case
    :class:`HypercubeGraph`). This class extends :class:`MaternKarhunenLoeveLogDomain`
    to implement this formula in the case of $\nu = \infty$ for efficiency.

    .. note::
        We only use the closed form expression if `num_levels` is `d + 1` which
        corresponds to exact computation. When truncated to fewer levels, we
        must use the parent class implementation to ensure consistency with
        feature map approximations.
    """

    def __init__(
        self,
        space: Union[HammingGraph, HypercubeGraph],
        num_levels: int,
        normalize: bool = True,
        eigenvalues_laplacian: Optional[B.Numeric] = None,
        eigenfunctions: Optional[Eigenfunctions] = None,
    ):
        if not isinstance(space, (HammingGraph, HypercubeGraph)):
            raise ValueError(
                f"`space` must be an instance of HammingGraph or HypercubeGraph, but got {type(space)}"
            )

        super().__init__(
            space,
            num_levels,
            normalize,
            eigenvalues_laplacian,
            eigenfunctions,
        )

    def K(
        self,
        params: Dict[str, B.Numeric],
        X: B.Numeric,
        X2: Optional[B.Numeric] = None,
        **kwargs,
    ) -> B.Numeric:
        if not isinstance(self.eigenfunctions, HammingGraphEigenfunctions):
            return super().K(params, X, X2, **kwargs)
        _, log_levels, log_normalizer = self._log_weights(params)
        if (
            B.all(params["nu"] == np.inf)
            and self.num_levels == self.space.dimension + 1
        ):
            log_kernel = _log_hamming_graph_heat_kernel(
                params["lengthscale"], X, X2, q=getattr(self.space, "n_cat", 2)
            )
            return B.exp(log_kernel if self.normalize else log_kernel + log_normalizer)

        if self.normalize:
            log_levels = log_levels - B.max(log_levels)
            log_levels = log_levels - B.logsumexp(log_levels)
        weights = B.exp(log_levels)
        return self.eigenfunctions._weighted_outerproduct_from_level_weights(
            weights, X, X2
        )

    def K_diag(self, params: Dict[str, B.Numeric], X: B.Numeric, **kwargs) -> B.Numeric:
        if not isinstance(self.eigenfunctions, HammingGraphEigenfunctions):
            return super().K_diag(params, X, **kwargs)
        _, log_levels, log_normalizer = self._log_weights(params)
        diagonal = B.ones(B.dtype(log_levels), X.shape[0])
        # Retain a differentiation graph for the constant normalized diagonal.
        return (
            diagonal + 0.0 * log_normalizer
            if self.normalize
            else B.exp(log_normalizer) * diagonal
        )
