from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
import copy
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.sparse

from fatiando.utils import safe_dot

from .base import FitMixin, Objective
from .cache import CachedMethod


class NonLinearMisfit(FitMixin, Objective):

    def __init__(self, nparams, config=None):
        super().__init__(nparams, islinear=False)
        self.p_ = None
        self.stats_ = None
        if config is None:
            config = dict(method='levmarq', initial=np.zeros(nparams))
        self._config = config
        self.set_cache()

    @abstractmethod
    def predict(self):
        pass

    def set_cache(self):
        self.predict = CachedMethod(self, 'predict', optional=['p'])
        if hasattr(self, 'jacobian'):
            self.jacobian = CachedMethod(self, 'jacobian')

    def copy(self, deep=False):
        obj = super().copy(deep)
        # Need to reset the cache because the CachedMethod in the copy is
        # holding an instance of the original object.
        obj.set_cache()
        return obj

    def _make_partials(self, *args, **kwargs):
        data = args[-1]
        args = args[:-1]
        weights = kwargs.get('weights', None)
        if weights is not None:
            if weights.ndim == 1:
                kwargs['weights'] = scipy.sparse.diags(weights, format='csr')
        def value(p):
            return self.value(p, data, *args, **kwargs)
        def gradient(p):
            return self.gradient(p, data, *args, **kwargs)
        def hessian(p):
            return self.hessian(p, data, *args, **kwargs)
        return value, gradient, hessian

    def value(self, p, data, *args, **kwargs):
        weights = kwargs.get('weights', None)
        pred = self.predict(*args, p=p)
        if weights is None:
            value = np.linalg.norm(data - pred)**2
        else:
            r = data - pred
            value = safe_dot(r.T, safe_dot(weights, r))
        value *= self.scale
        return value

    def gradient(self, p, data, *args, **kwargs):
        weights = kwargs.get('weights', None)
        jacobian = self.jacobian(*args, p=p)
        if p is None:
            residuals = data
        else:
            residuals = data - self.predict(*args, p=p)
        if weights is None:
            gradient = safe_dot(jacobian.T, residuals)
        else:
            gradient = safe_dot(jacobian.T, safe_dot(weights, residuals))
        # Check if the gradient isn't a one column matrix
        if len(gradient.shape) > 1:
            # Need to convert it to a 1d array so that hell won't break
            # loose
            gradient = np.array(gradient).ravel()
        gradient *= -2*self.scale
        return gradient

    def hessian(self, p, data, *args, **kwargs):
        weights = kwargs.get('weights', None)
        jacobian = self.jacobian(*args, p=p)
        if weights is None:
            hessian = safe_dot(jacobian.T, jacobian)
        else:
            hessian = safe_dot(jacobian.T, safe_dot(weights, jacobian))
        hessian *= 2*self.scale
        return hessian

    def score(self, *args):
        data = args[-1]
        pred = self.predict(*args[:-1])
        u = ((data - pred)**2).sum()
        v = ((data - data.mean())**2).sum()
        return 1 - u/v

    def fit_reweighted(self, *args, **kwargs):
        iterations = kwargs.get('iterations', 10)
        tol = kwargs.get('tol', 1e-8)
        data = args[-1]
        self.fit(*args)
        for i in range(iterations):
            residuals = np.abs(data - self.predict(*args[:-1]))
            residuals[residuals < tol] = tol
            weights = 1/residuals
            self.fit(*args, weights=weights)
        return self


class LinearMisfit(NonLinearMisfit):

    def __init__(self, nparams, config=None):
        if config is None:
            config = dict(method='linear')
        super().__init__(nparams=nparams, config=config)
        self.islinear = True

    def set_cache(self):
        self.predict = CachedMethod(self, 'predict', optional=['p'])
        if hasattr(self, 'jacobian'):
            self.jacobian = CachedMethod(self, 'jacobian', ignored=['p'])
