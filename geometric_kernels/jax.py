"""
Loads JAX backend in lab, spherical_harmonics and geometric_kernels.

..note::
    A tutorial on the JAX backend is available in the
    :doc:`backends/JAX_Graph.ipynb </examples/backends/JAX_Graph>` notebook.
"""

import logging

import lab.jax  # noqa
import jaxlib
import spherical_harmonics.jax  # noqa

import geometric_kernels.lab_extras.jax  # noqa

# Work around a Python 3.12 dispatch regression where this internal JAX array
# type can be treated as unfaithful by plum, disabling dispatch cache usage.
try:
    jaxlib._jax.ArrayImpl.__faithful__ = True
except Exception:  # pragma: no cover
    pass

logging.getLogger(__name__).info("JAX backend enabled.")
