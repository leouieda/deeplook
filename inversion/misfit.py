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

    def __init__(self, nparams, config=None):
        self.nparams = nparams
        self.p_ = None
        self.stats_ = None
        self.scale = 1
        if config is None:
            config = dict(method='levmarq', initial=np.zeros(nparams))
        self._config = config

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
            g, H = self.evaluate(p=None, data=data, value=False, gradient=True,
                                 hessian=True, **kwargs)
            p, stats = optimizer(H, g, **config)
        elif method in ['newton', 'levmarq']:
            def evaluate(p):
                return self.evaluate(p, data=data, value=True, gradient=True,
                                     hessian=True, **kwargs)
            p, stats = optimizer(evaluate, **config)
        elif self.fit_method == 'steepest':
            def evaluate(p):
                return self.evaluate(p, data=data, value=True, gradient=True,
                                     **kwargs)
            p, stats = optimizer(evaluate, **config)
        elif self.fit_method == 'acor':
            def evaluate(p):
                return self.evaluate(p, data=data, value=True, **kwargs)
            p, stats = optimizer(evaluate, **config)
        self.p_ = p
        self.stats_ = stats
        return self

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def evaluate(self, p, data, value=True, gradient=False, hessian=False,
                 weights=None, **kwargs):
        # Calculate the value and jacobian only if they are required.
        if value or (gradient and p is not None):
            pred = self.predict(p=p, **kwargs)
        if gradient or hessian:
            jacobian = self.jacobian(p=p, **kwargs)
        output = []
        if value:
            if weights is None:
                val = np.linalg.norm(data - pred)**2
            else:
                r = data - pred
                val = safe_dot(r.T, safe_dot(weights, r))
            val *= self.scale
            output.append(val)
        if gradient:
            if p is None:
                residuals = data
            else:
                residuals = data - pred
            if weights is None:
                grad = safe_dot(jacobian.T, residuals)
            else:
                grad = safe_dot(jacobian.T, safe_dot(weights, residuals))
            # Check if the gradient isn't a one column matrix
            if len(grad.shape) > 1:
                # Need to convert it to a 1d array so that hell won't break
                # loose
                grad = np.array(grad).ravel()
            grad *= -2*self.scale
            output.append(grad)
        if hessian:
            if weights is None:
                hess = safe_dot(jacobian.T, jacobian)
            else:
                hess = safe_dot(jacobian.T, safe_dot(weights, jacobian))
            hess *= 2*self.scale
            output.append(hess)
        return output

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


class LinearMisfit(NonLinearMisfit):
    def __init__(self, nparams, config=None):
        if config is None:
            config = dict(method='linear')
        super().__init__(nparams=nparams, config=config)

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def fit(self):
        pass
