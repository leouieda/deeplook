"""
DeepLook: Solvers for inverse problems
"""
# Ignore the unused import warning in flake8
# flake8: noqa
# noqa: F401
from __future__ import absolute_import
from .misfit import L2Norm, L2NormLinear
from .models import NonLinearModel, LinearModel
from .regularization import Damping, Smoothness, Smoothness1D, \
                            TotalVariation, TotalVariation1D
from .linear_solver import LinearOptimizer
from .gradient_descent import Newton, LevMarq, SteepestDescent
from .heuristic import ACOR
from .scipy_optimizer import ScipyOptimizer


__version__ = '0.1a0'


def test(doctest=True, verbose=False):
    """
    Run the test suite.

    Uses `py.test <http://pytest.org/>`__ to discover and run the tests.

    Parameters:

    * doctest : bool
        If ``True``, will run the doctests as well (code examples that start
        with a ``>>>`` in the docs).
    * verbose : bool
        If ``True``, will print extra information during the test run.

    Returns:

    * exit_code : int
        The exit code for the test run. If ``0``, then all tests pass.

    """
    import pytest
    args = []
    if verbose:
        args.append('-v')
    if doctest:
        args.append('--doctest-modules')
    args.append('--pyargs deeplook')
    return pytest.main(args)
