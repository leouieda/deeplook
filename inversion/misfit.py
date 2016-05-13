from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
import copy
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.sparse

from fatiando.utils import safe_dot
from fatiando.inversion import optimization

from .cache import CachedMethod


class NonLinearMisfit(with_metaclass(ABCMeta)):

    def __init__(self, nparams, config=None, cached=True):
        self.nparams = nparams
        self.p_ = None
        self.scale = 1
        if config is None:
            config = dict(method='levmarq', initial=np.zeros(nparams))
        self._config = config
        self.cached = cached
        if self.cached:
            self.predict = CachedMethod(self, 'predict')
            self.residuals = CachedMethod(self, 'residuals')
            if hasattr(self, 'jacobian'):
                self.jacobian = CachedMethod(self, 'jacobian')

    @abstractmethod
    def predict(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, args, data, weights=None):
        self.optimize(data=data, args, weights=None)
        return self

    def residuals(self, data, **kwargs):
        pred = self.predict(**kwargs)
        assert data.shape == pred.shape, \ \
            "Data shape doesn't match auxiliary arguments."
        return data - pred

    def optimize(self, data, **kwargs):
        args = {k:kwargs[k] for k in kwargs if k != 'weights'}
        weights = kwargs.get('weights', None)
        # TODO: Convert the weights to a sparse matrix if 1d
        # Make partials of the methods
        def hessian(p):
            return self.hessian(p, data, weights=weights, **args)
        def gradient(p):
            return self.gradient(p, data, weights=weights, **args)
        def value(p):
            return self.value(p, data, weights=weights, **args)
        # Get the optimization method
        method = self.config['method']
        config = {k:kwargs[k] for k in kwargs if k != 'method'}
        optimizer = getattr(optimization, method)
        if method == 'linear':
            solver = optimizer(hessian(None), gradient(None), **config)
        elif method in ['newton', 'levmarq']:
            solver = optimizer(hessian, gradient, value, **config)
        elif self.fit_method == 'steepest':
            solver = optimizer(gradient, value, **config)
        elif self.fit_method == 'acor':
            solver = optimizer(value, **config)
        # Run the optimizer to the end
        for i, p, stats in solver:
            continue
        self.p_ = p
        self.stats_ = stats
        return self

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def value(self, p, data, weights=None, **kwargs):
        r = self.residuals(data, p=p, **kwargs)
        if weights is None:
            val = np.linalg.norm(r)**2
        else:
            val = np.sum(weights*(r**2))
        return val*self.scale

    def gradient(self, p, data, weights=None, **kwargs):
        jacobian = self.jacobian(p, **kwargs)
        if p is None:
            residuals = data
        else:
            residuals = self.residuals(data, p=p, **kwargs)
        if weights is None:
            grad = safe_dot(jacobian.T, residuals)
        else:
            grad = safe_dot(jacobian.T, weights*residuals)
        # Check if the gradient isn't a one column matrix
        if len(grad.shape) > 1:
            # Need to convert it to a 1d array so that hell won't break loose
            grad = np.array(grad).ravel()
        grad *= -2*self.scale
        return grad

    def hessian(self, p, data, weights=None, **kwargs):
        jacobian = self.jacobian(p, **kwargs)
        if weights is None:
            hessian = safe_dot(jacobian.T, jacobian)
        else:
            hessian = safe_dot(jacobian.T, weights*jacobian)
        hessian *= 2*self.scale
        return hessian
