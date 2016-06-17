from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.sparse as sp

from .optimization import LinearOptimizer
from .misfit import L2NormMisfit


class NonLinearModel(with_metaclass(ABCMeta)):

    def __init__(self, nparams, misfit='L2NormMisfit', optimizer='Nelder-Mead',
                 regularization=None, scale=1):
        self.scale = scale
        self.nparams = nparams
        self.islinear = False
        self.p_ = None
        self.regularization = None
        self.misfit = None
        self.optimizer = None
        self.config(misfit=misfit,
                    optimizer=optimizer,
                    regularization=regularization)

    @abstractmethod
    def predict(self, args):
        "Return data predicted by self.p_"
        pass

    def config(self, optimizer=None, misfit=None, regularization=None,
               scale=None):
        if optimizer is not None:
            self._set_optimizer(optimizer)
        if misfit is not None:
            self._set_misfit(misfit)
        if regularization is not None:
            self._set_regularization(regularization)
        if scale is not None:
            self.scale = scale
        return self


    def _set_optimizer(self, optimizer):
        "Configure the optimization"
        if optimizer == 'linear':
            self.optimizer = LinearOptimizer()
        else:
            self.optimizer = optimizer

    def _set_misfit(self, misfit):
        "Pass a different misfit function"
        if misfit == 'L2NormMisfit':
            self.misfit = L2NormMisfit
        else:
            self.misfit = misfit

    def _set_regularization(self, reguls):
        "Use the given regularization"
        self.regularization = reguls

    def make_partial(self, args, func):
        def partial(p):
            backup = self.p_
            self.p_ = p
            res = getattr(self, func)(*args)
            self.p_ = backup
            return res
        return partial

    def fit(self, args, data, weights=None, jacobian=None):
        "Fit the model to the given data"
        misfit_args = dict(data=data,
                           predict=self.make_partial(args, 'predict'),
                           weights=weights,
                           islinear=self.islinear,
                           jacobian_cache=jacobian)
        if hasattr(self, 'jacobian'):
            misfit_args['jacobian'] = self.make_partial(args, 'jacobian')
        misfit = self.misfit(**misfit_args)
        components = [[self.scale, misfit]]
        if self.regularization is not None:
            components.extend(self.regularization)
        objective = Objective(components)
        self.p_ = self.optimizer.minimize(objective) # the estimated parameter vector
        return self

    def score(self, *args, **kwargs):
        data = args[-1]
        pred = self.predict(*args[:-1])
        scorer = kwargs.get('scorer', 'R2')
        if scorer == 'L2':
            return np.linalg.norm(data - pred)**2
        elif scorer == 'R2':
            u = ((data - pred)**2).sum()
            v = ((data - data.mean())**2).sum()
            return 1 - u/v
        else:
            assert False, "Unknown scorer '{}'".format(scorer)

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
    def __init__(self, nparams, misfit='L2NormMisfit', optimizer='linear',
                 regularization=None, scale=1):
        super().__init__(nparams, misfit=misfit, optimizer=optimizer,
                         regularization=regularization, scale=scale)
        self.islinear = True


class Objective(object):
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
