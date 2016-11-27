"""
Use the test function to run the test suite.
"""


def test(doctest=True, verbose=False, coverage=False):
    """
    Run the test suite.

    Uses `py.test <http://pytest.org/>`__ to discover and run the tests.

    Parameters:

    * doctest : bool
        If ``True``, will run the doctests as well (code examples that start
        with a ``>>>`` in the docs).
    * verbose : bool
        If ``True``, will print extra information during the test run.
    * coverage : bool
        If ``True``, will run test coverage analysis on the code as well.
        Requires ``pytest-cov``.

    Returns:

    * exit_code : int
        The exit code for the test run. If ``0``, then all tests pass.

    """
    import pytest
    args = []
    if verbose:
        args.append('-v')
    if coverage:
        args.append('--cov=deeplook')
        args.append('--cov-report term-missing')
    if doctest:
        args.append('--doctest-modules')
    args.append('--pyargs')
    args.append('deeplook')
    return pytest.main(args)
