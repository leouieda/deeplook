from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
import copy
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.sparse

from fatiando.utils import safe_dot

from . import optimization
from .cache import CachedMethod


class NonLinearMisfit(with_metaclass(ABCMeta)):

    _no_gradient = ['acor']
    _no_hessian = ['steepest']
    _needs_both = ['newton', 'levmarq', 'linear']

    def __init__(self, nparams, config=None):
        self.nparams = nparams
        self.p_ = None
        self.stats_ = None
        self.scale = 1
        if config is None:
            config = dict(method='levmarq', initial=np.zeros(nparams))
        self._config = config
        self._set_cache()

    def _set_cache(self):
        self.predict = CachedMethod(self, 'predict')
        if hasattr(self, 'jacobian'):
            self.jacobian = CachedMethod(self, 'jacobian')

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @property
    def estimate_(self):
        return self.fmt_estimate(self.p_)

    def fmt_estimate(self, p):
        return p

    def optimize(self, data, **kwargs):
        weights = kwargs.get('weights', None)
        if weights is not None:
            if weights.ndim == 1:
                kwargs['weights'] = scipy.sparse.diags(weights, format='csr')
        tmp = self._config
        method = tmp['method']
        config = {k:tmp[k] for k in tmp if k != 'method'}
        optimizer = getattr(optimization, method)
        if method == 'linear':
            gradient = self.gradient(p=None, data=data, **kwargs)
            hessian = self.hessian(p=None, data=data, **kwargs)
            p, stats = optimizer(gradient, hessian, **config)
        elif method in self._needs_both:
            def value(p):
                return self.value(p, data=data, **kwargs)
            def gradient(p):
                return self.gradient(p, data=data, **kwargs)
            def hessian(p):
                return self.hessian(p, data=data, **kwargs)
            p, stats = optimizer(value, gradient, hessian, **config)
        elif method in self._no_hessian:
            def value(p):
                return self.value(p, data=data, **kwargs)
            def gradient(p):
                return self.gradient(p, data=data, **kwargs)
            p, stats = optimizer(value, gradient, **config)
        elif method in self._no_gradient:
            def value(p):
                return self.value(p, data=data, **kwargs)
            p, stats = optimizer(value, nparams=self.nparams, **config)
        self.p_ = p
        self.stats_ = stats

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def value(self, p, data, weights=None, **kwargs):
        pred = self.predict(p=p, **kwargs)
        if weights is None:
            value = np.linalg.norm(data - pred)**2
        else:
            r = data - pred
            value = safe_dot(r.T, safe_dot(weights, r))
        value *= self.scale
        return value

    def gradient(self, p, data, weights=None, **kwargs):
        jacobian = self.jacobian(p=p, **kwargs)
        if p is None:
            residuals = data
        else:
            residuals = data - self.predict(p=p, **kwargs)
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

    def hessian(self, p, data, weights=None, **kwargs):
        jacobian = self.jacobian(p=p, **kwargs)
        if weights is None:
            hessian = safe_dot(jacobian.T, jacobian)
        else:
            hessian = safe_dot(jacobian.T, safe_dot(weights, jacobian))
        hessian *= 2*self.scale
        return hessian

    def score(self, *args):
        data = args[-1]
        pred = self.predict(*args[:-1])
        return np.average((data - pred)**2)

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

    def config(self, method, **kwargs):
        self._config = dict(method=method)
        self._config.update(kwargs)
        return self

class LinearMisfit(NonLinearMisfit):

    def __init__(self, nparams, config=None):
        if config is None:
            config = dict(method='linear')
        super().__init__(nparams=nparams, config=config)

    def _set_cache(self):
        self.predict = CachedMethod(self, 'predict', optional=['p'])
        if hasattr(self, 'jacobian'):
            self.jacobian = CachedMethod(self, 'jacobian', ignored=['p'])
            self.hessian = CachedMethod(self, 'hessian', ignored=['p'],
                                        bypass=['weights'])

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def fit(self):
        pass
