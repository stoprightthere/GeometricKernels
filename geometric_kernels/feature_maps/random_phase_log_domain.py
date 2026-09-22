"""Log-domain random-phase features for discrete-spectrum spaces."""

import lab as B
from beartype.typing import Dict, Tuple

from geometric_kernels.feature_maps.random_phase import RandomPhaseFeatureMapCompact
from geometric_kernels.lab_extras import from_numpy, is_complex
from geometric_kernels.spaces import DiscreteSpectrumSpace


class RandomPhaseFeatureMapLogDomain(RandomPhaseFeatureMapCompact):
    """Random-phase features with log-domain spectral weighting.

    Sampling, feature ordering, and row normalization follow
    :class:`RandomPhaseFeatureMapCompact`. When the eigenfunctions support log
    products, the map avoids premature spectral underflow and large
    multiplicities. Otherwise it uses the standard compact feature map.

    :param space:
        A discrete-spectrum space with random sampling and addition-theorem
        eigenfunctions.
    :param num_levels:
        Number of spectral levels to include.
    :param num_random_phases:
        Number of sampled phases. The map returns this many features per level.
    """

    def __init__(
        self,
        space: DiscreteSpectrumSpace,
        num_levels: int,
        num_random_phases: int = 3000,
    ):
        super().__init__(space, num_levels, num_random_phases)

    def __call__(
        self,
        X: B.Numeric,
        params: Dict[str, B.Numeric],
        *,
        key: B.RandomState,
        normalize: bool = True,
        **kwargs,
    ) -> Tuple[B.RandomState, B.Numeric]:
        """Return the updated random key and log-domain random-phase features.

        Arguments and return shapes follow ``RandomPhaseFeatureMapCompact``.
        Normalization produces unit-norm feature rows; unnormalized features
        can still exceed the floating-point range.
        """
        if not self.eigenfunctions.supports_log_domain:
            return super().__call__(X, params, key=key, normalize=normalize, **kwargs)

        from geometric_kernels.kernels.karhunen_loeve_log_domain import (
            MaternKarhunenLoeveLogDomain,
        )

        key, phases = self.space.random(key, self.num_random_phases)
        log_spectrum = MaternKarhunenLoeveLogDomain.log_spectrum(
            self.space.get_eigenvalues(self.num_levels),
            params["nu"],
            params["lengthscale"],
            self.space.dimension,
        )
        phases = B.cast(B.dtype(X), from_numpy(X, phases))
        return key, self._features_from_log_spectrum(log_spectrum, X, phases, normalize)

    def _features_from_log_spectrum(self, log_spectrum, X, phases, normalize):
        log_phi_magnitude, signs = self.eigenfunctions.phi_product_log(
            X, phases, dtype=B.dtype(log_spectrum)
        )
        log_magnitude = log_phi_magnitude + B.transpose(0.5 * log_spectrum)
        log_magnitude = B.reshape(log_magnitude, X.shape[0], -1)
        signs = B.reshape(signs, X.shape[0], -1)
        if normalize:
            log_magnitude = log_magnitude - B.max(log_magnitude, axis=1, squeeze=False)
            log_magnitude = log_magnitude - 0.5 * B.logsumexp(
                2 * log_magnitude, axis=1, squeeze=False
            )
        features = signs * B.exp(log_magnitude)
        if is_complex(features):
            features = B.concat(B.real(features), B.imag(features), axis=1)
        return features
