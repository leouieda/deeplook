"""
DeepLook: Solvers for inverse problems
"""
# Ignore the unused import warning in flake8
# flake8: noqa
# noqa: F401
from __future__ import absolute_import

try:
    import pytest
except ImportError:
    pytest = None


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

    Raises:

    * ``AssertionError`` if pytest returns a non-zero error code indicating
      that some tests have failed.

    """
    assert pytest is not None, "Must have 'pytest' installed to run tests."
    args = []
    if verbose:
        args.append('-v')
    if doctest:
        args.append('--doctest-modules')
    args.append('--pyargs')
    args.append('deeplook')
    status = pytest.main(args)
    assert status == 0, "Some tests have failed."
