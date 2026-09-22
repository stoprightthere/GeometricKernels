"""Matérn Karhunen-Loève kernels with log-domain spectral normalization."""

import lab as B
import numpy as np
from beartype.typing import Dict

from geometric_kernels.kernels.karhunen_loeve import MaternKarhunenLoeveKernel
from geometric_kernels.lab_extras import from_numpy, is_complex
from geometric_kernels.utils.utils import _check_1_vector, _check_field_in_params


class MaternKarhunenLoeveLogDomain(MaternKarhunenLoeveKernel):
    """A discrete-spectrum Matérn kernel whose weights are computed in log space.

    Eigenfunctions provide independent log multiplicities, allowing the
    normalizer to be evaluated even when linear multiplicities overflow.
    Eigenfunctions with ``supports_log_domain`` can also evaluate kernel
    matrices without forming linear per-eigenfunction weights.
    """

    @staticmethod
    def log_spectrum(s, nu, lengthscale, dimension):
        """Evaluate the log Matérn spectrum without forming linear weights."""
        _check_1_vector(lengthscale, "lengthscale")
        _check_1_vector(nu, "nu")
        s = B.cast(B.dtype(lengthscale), s)
        safe_nu = B.where(nu == np.inf, B.ones(lengthscale), nu)
        safe_lengthscale = B.where(nu == np.inf, B.ones(lengthscale), lengthscale)
        finite = -(safe_nu + dimension / 2.0) * B.log(
            2.0 * safe_nu / safe_lengthscale**2 + s
        )
        infinite = -(lengthscale**2) * s / 2.0
        return B.where(nu == np.inf, infinite, finite)

    @staticmethod
    def spectrum(s, nu, lengthscale, dimension):
        return B.exp(
            MaternKarhunenLoeveLogDomain.log_spectrum(s, nu, lengthscale, dimension)
        )

    def _log_weights(self, params):
        _check_field_in_params(params, "lengthscale")
        _check_field_in_params(params, "nu")
        log_spectrum = self.log_spectrum(
            self.eigenvalues_laplacian,
            params["nu"],
            params["lengthscale"],
            self.space.dimension,
        )
        log_multiplicities = B.cast(
            B.dtype(log_spectrum),
            from_numpy(
                log_spectrum, self.eigenfunctions.log_num_eigenfunctions_per_level
            ),
        )[:, None]
        log_levels = log_spectrum + log_multiplicities
        return log_spectrum, log_levels, B.logsumexp(log_levels)

    def log_eigenvalues(self, params: Dict[str, B.Numeric]) -> B.Numeric:
        """Return per-eigenfunction log weights, shape [L, 1]."""
        log_spectrum, _, log_normalizer = self._log_weights(params)
        return log_spectrum - log_normalizer if self.normalize else log_spectrum

    def eigenvalues(self, params: Dict[str, B.Numeric]) -> B.Numeric:
        return B.exp(self.log_eigenvalues(params))

    def K(self, params, X, X2=None, **kwargs):
        if not self.eigenfunctions.supports_log_domain:
            return super().K(params, X, X2, **kwargs)
        result = self.eigenfunctions.weighted_outerproduct_log(
            self.log_eigenvalues(params), X, X2, **kwargs
        )
        return B.real(result) if is_complex(result) else result

    def K_diag(self, params, X, **kwargs):
        if not self.eigenfunctions.supports_log_domain:
            return super().K_diag(params, X, **kwargs)
        result = self.eigenfunctions.weighted_outerproduct_diag_log(
            self.log_eigenvalues(params), X, **kwargs
        )
        return B.real(result) if is_complex(result) else result
