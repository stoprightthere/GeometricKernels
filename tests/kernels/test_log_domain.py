"""Tests for the space-independent log-domain kernel and feature map."""

import numpy as np
import pytest

from geometric_kernels.feature_maps import RandomPhaseFeatureMapLogDomain
from geometric_kernels.kernels import (
    MaternGeometricKernel,
    MaternKarhunenLoeveKernel,
    MaternKarhunenLoeveLogDomain,
    default_feature_map,
)
from geometric_kernels.spaces import Circle, HammingGraph, HypercubeGraph


def params(nu=1.5, lengthscale=0.7):
    return {"nu": np.array([nu]), "lengthscale": np.array([lengthscale])}


def test_log_domain_support_flag():
    assert not Circle().get_eigenfunctions(4).supports_log_domain
    assert HypercubeGraph(6).get_eigenfunctions(4).supports_log_domain
    assert HammingGraph(6, 4).get_eigenfunctions(4).supports_log_domain


@pytest.mark.parametrize("space", [Circle(), HypercubeGraph(6), HammingGraph(6, 4)])
def test_log_kernel_matches_linear_kernel_on_small_spaces(space):
    levels = 4
    kernel = MaternKarhunenLoeveLogDomain(space, levels)
    linear = MaternKarhunenLoeveKernel(space, levels)
    key = np.random.RandomState(4)
    _, x = space.random(key, 5)
    for nu in [0.5, np.inf]:
        p = params(nu)
        np.testing.assert_allclose(
            kernel.eigenvalues(p), linear.eigenvalues(p), rtol=1e-12
        )
        np.testing.assert_allclose(
            kernel.K(p, x), linear.K(p, x), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(kernel.K_diag(p, x), linear.K_diag(p, x), rtol=1e-12)


def test_large_log_kernel_avoids_underflow_and_multiplicity_overflow():
    space = HammingGraph(1024, 20)
    kernel = MaternKarhunenLoeveLogDomain(space, 24)
    p = params(lengthscale=0.1)
    x = np.zeros((2, 1024), dtype=int)
    x[1, 0] = 1
    assert np.all(np.isfinite(kernel.log_eigenvalues(p)))
    matrix = kernel.K(p, x)
    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(np.diag(matrix), 1, atol=1e-11)


def test_log_kernel_evaluates_when_all_linear_eigenvalues_underflow():
    space = HammingGraph(1024, 20)
    kernel = MaternKarhunenLoeveLogDomain(space, 1025)
    p = params(lengthscale=0.1)
    x = np.zeros((1, 1024), dtype=int)
    assert np.all(kernel.eigenvalues(p) == 0)
    np.testing.assert_allclose(kernel.K(p, x), [[1]], atol=1e-11)
    np.testing.assert_allclose(kernel.K_diag(p, x), [1], atol=1e-11)


def test_phi_product_log_exceeds_linear_range():
    phi = HammingGraph(1024, 20).get_eigenfunctions(600)
    x = np.zeros((1, 1024), dtype=int)
    log_magnitude, sign = phi.phi_product_log(x, dtype=np.float64)
    assert np.isfinite(log_magnitude[0, 0, 599])
    assert log_magnitude[0, 0, 599] > np.log(np.finfo(float).max)
    assert sign[0, 0, 599] == 1


def test_random_phase_map_works_on_non_hamming_space():
    space = Circle()
    x = np.array([[0.0], [0.3], [1.4]])
    p = params()
    fmap = RandomPhaseFeatureMapLogDomain(space, 4, 8)
    assert (
        type(default_feature_map(kernel=MaternKarhunenLoeveLogDomain(space, 4)))
        is RandomPhaseFeatureMapLogDomain
    )
    _, actual = fmap(x, p, key=np.random.RandomState(8), normalize=False)
    _, phases = space.random(np.random.RandomState(8), 8)
    spectrum = MaternKarhunenLoeveKernel.spectrum(
        space.get_eigenvalues(4), p["nu"], p["lengthscale"], space.dimension
    )
    expected = (
        fmap.eigenfunctions.phi_product(x, phases) * np.sqrt(spectrum.T)
    ).reshape(3, -1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_default_hamming_kernel_and_map_use_generic_log_classes():
    space = HammingGraph(8, 4)
    kernel, fmap = MaternGeometricKernel(space, num=5, return_feature_map=True)
    assert type(kernel) is MaternKarhunenLoeveLogDomain
    assert type(fmap) is RandomPhaseFeatureMapLogDomain
