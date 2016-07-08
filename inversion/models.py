from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.sparse as sp

from .optimization import LinearOptimizer, ScipyOptimizer
from .misfit import L2NormMisfit


class NonLinearModel(with_metaclass(ABCMeta)):


    def __init__(self, optimizer, misfit=None):
        self.optimizer = optimizer
        if misfit is None:
            self.misfit = L2NormMisfit
        else:
            self.misfit = misfit
        self.p_ = None
        self.custom_regularization = None

    @abstractmethod
    def predict(self, args):
        "Return data predicted by self.p_"
        pass

    def config(self, optimizer=None, misfit=None, regularization=None):
        if optimizer is not None:
            self.optimizer = optimizer
        if misfit is not None:
            self.misfit = misfit
        if regularization is not None:
            self.custom_regularization = regularization
        return self


    def make_misfit(self, data, args, weights=None, jacobian=None):
        "Fit the model to the given data"
        def make_partial(func):
            def partial(p):
                backup = self.p_
                self.p_ = p
                res = getattr(self, func)(*args)
                self.p_ = backup
                return res
            return partial
        misfit_args = dict(data=data,
                           predict=make_partial('predict'),
                           weights=weights,
                           jacobian_cache=jacobian)
        if hasattr(self, 'jacobian'):
            misfit_args['jacobian'] = make_partial('jacobian')
        return self.misfit(**misfit_args)

    def make_objective(self, misfit, regularization):
        "Fit the model to the given data"
        if not isinstance(misfit, list):
            misfit = [[1, misfit]]
        components = misfit + regularization
        if self.custom_regularization is not None:
            components.extend(self.custom_regularization)
        return Objective(components)

    def score(self, *args, **kwargs):
        scorer = kwargs.get('scorer', 'R2')
        assert scorer in ['R2', 'L2'], "Unknown scorer '{}'".format(scorer)
        data = args[-1]
        pred = self.predict(*args[:-1])
        if scorer == 'L2':
            score = np.linalg.norm(data - pred)**2
        elif scorer == 'R2':
            u = ((data - pred)**2).sum()
            v = ((data - data.mean())**2).sum()
            score = 1 - u/v
        return score

    def fit_reweighted(self, *args, **kwargs):
        iterations = kwargs.pop('iterations', 10)
        tol = kwargs.pop('tol', 1e-8)
        data = args[-1]
        self.fit(*args)
        for i in range(iterations):
            residuals = np.abs(data - self.predict(*args[:-1]))
            residuals[residuals < tol] = tol
            weights = sp.diags(1/residuals, format='csr')
            kwargs['weights'] = weights
            self.fit(*args, **kwargs)
        return self



class LinearModel(NonLinearModel):
    def __init__(self, misfit='L2NormMisfit', optimizer='linear',
                 regularization=None, scale=1):
        super().__init__(misfit=misfit, optimizer=optimizer,
                         regularization=regularization, scale=scale)
        self.islinear = True


class Objective(object):
    """
    Objective function composed of a sum of components
    """

    def __init__(self, components):
        self.components = components

    def value(self, p):
        return np.sum(lamb*comp.value(p)
                       for lamb, comp in self.components)

    def gradient(self, p):
        return np.sum(lamb*comp.gradient(p)
                       for lamb, comp in self.components)

    def gradient_at_null(self):
        return np.sum(lamb*comp.gradient_at_null()
                       for lamb, comp in self.components)

    def hessian(self, p):
        return np.sum(lamb*comp.hessian(p)
                       for lamb, comp in self.components)
