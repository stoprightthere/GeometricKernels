import numpy as np
import torch
import lab as B

import geometric_kernels.torch  # noqa
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.utils.manifold_utils import manifold_laplacian


_TRUNCATION_LEVEL = 10
_NU = 2.5


def test_sphere_heat_kernel():
    # Parameters
    grid_size = 4
    nb_samples = 10
    dimension = 3

    # Create manifold
    hypersphere = Hypersphere(dim=dimension)

    # Generate samples
    ts = torch.linspace(0.1, 1, grid_size, requires_grad=True)
    xs = torch.tensor(np.array(hypersphere.random_point(nb_samples)), requires_grad=True)
    ys = xs

    # Define kernel
    kernel = MaternKarhunenLoeveKernel(hypersphere, _TRUNCATION_LEVEL)
    params, state = kernel.init_params_and_state()
    params["nu"] = torch.tensor(torch.inf)

    # Define heat kernel function
    def heat_kernel(t, x, y):
        params["lengthscale"] = B.sqrt(2*t)
        return kernel.K(params, state, x, y)

    for t in ts:
        for x in xs:
            for y in ys:
                # Compute the derivative of the kernel function wrt t
                dfdt, _, _ = torch.autograd.grad(heat_kernel(t, x[None], y[None]), (t, x, y))
                # Compute the Laplacian of the kernel on the manifold
                egrad = lambda u: torch.autograd.grad(heat_kernel(t, u[None], y[None]), (t, u, y))[1]  # noqa
                fx = lambda u: heat_kernel(t, u[None], y[None])  # noqa
                ehess = lambda u, h: torch.autograd.functional.hvp(fx, u, h)[1]  # noqa
                lapf = manifold_laplacian(hypersphere, x, egrad, ehess)

                # Check that they match
                assert np.isclose(dfdt.detach().numpy(), lapf, atol=1.e-6)
