from __future__ import division
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg



def inv(matrix):
    """
    Calculate the inverse of a matrix.

    Uses the standard ``numpy.linalg.inv`` if *matrix* is dense. If it is
    sparse (from ``scipy.sparse``) then will use ``scipy.sparse.linalg.inv``.
    """
    if scipy.sparse.issparse(matrix):
        return scipy.sparse.linalg.inv(matrix)
    else:
        return np.linalg.inv(matrix)


def solve(matrix, vector):
    """
    Solve a linear system.
    """
    if scipy.sparse.issparse(matrix) or scipy.sparse.issparse(vector):
        estimate, status = scipy.sparse.linalg.cgs(matrix, vector)
        if status >= 0:
            return estimate
        else:
            raise ValueError('CGS exited with input error')
    else:
        return scipy.linalg.solve(matrix, vector)


def dot(a, b):
    """
    Make the dot product using the appropriate method.
    """
    if scipy.sparse.issparse(a) or scipy.sparse.issparse(b):
        return a*b
    else:
        return np.dot(a, b)


def diagonal(matrix):
    """
    Get the diagonal of a matrix using the appropriate method.
    """
    if scipy.sparse.issparse(matrix):
        return np.array(matrix.diagonal())
    else:
        return np.diagonal(matrix)
