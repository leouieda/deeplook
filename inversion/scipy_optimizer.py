from __future__ import division
from future.builtins import object, super, range
import warnings
import numpy as np
from scipy import optimize


class ScipyOptimizer(object):

    needs_jac = 'CG BFGS Newton-CG L-BFGS-B TNC SLSQP dogleg trust-ncg'.split()
    needs_hess = 'Newton-CG dogleg trust-ncg'.split()

    def __init__(self, method, **kwargs):
        self.method = self._is_valid(method)
        self.kwargs = kwargs

    def _is_valid(self, method):
        return method

    def minimize(self, objective):
        args = dict(self.kwargs)
        if self.method in self.needs_jac:
            args['jac'] = objective.gradient
        if self.method in self.needs_hess:
            args['hess'] = objective.hessian
        res = optimize.minimize(fun=objective.value, method=self.method,
                                **args)
        self.stats = res
        return res.x

