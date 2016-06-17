from __future__ import division
from future.builtins import object, super, range
import warnings
import numpy as np
from scipy import optimize


class ScipyOptimizer(object):

    def __init__(self, method, **kwargs):
        self.method = self._is_valid(method)
        self.kwargs = kwargs

    def _is_valid(self, method):
        return method

    def minimize(self, objective):
        res = optimize.minimize(fun=objective.value,
                                method=self.method,
                                jac=objective.gradient,
                                hess=objective.hessian,
                                **self.kwargs)
        self.stats = res
        return res.x

