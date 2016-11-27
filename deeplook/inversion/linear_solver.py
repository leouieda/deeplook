from __future__ import division
from future.builtins import object, super, range
import warnings
import numpy as np
import scipy.sparse as sp

from fatiando.utils import safe_solve, safe_diagonal, safe_dot


class LinearOptimizer(object):

    def __init__(self, precondition=True):
        self.precondition = precondition

    def minimize(self, objective):
        hessian = objective.hessian(None)
        gradient = objective.gradient_at_null()
        if self.precondition:
            diag = np.abs(safe_diagonal(hessian))
            diag[diag < 10 ** -10] = 10 ** -10
            precond = sp.diags(1. / diag, 0).tocsr()
            hessian = safe_dot(precond, hessian)
            gradient = safe_dot(precond, gradient)
        p = safe_solve(hessian, -gradient)
        self.stats = dict(method="Linear solver")
        return p
