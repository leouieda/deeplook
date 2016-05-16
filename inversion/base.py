from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

from . import optimization


class FitMixin(with_metaclass(ABCMeta)):

    _no_gradient = ['acor']
    _no_hessian = ['steepest']
    _needs_both = ['newton', 'levmarq', 'linear']

    @abstractmethod
    def _make_partials(self, *args, **kwargs):
        # Return the value, gradient and hessian functions
        # that take only a p array as argument.
        pass

    def config(self, method, **kwargs):
        self._config = dict(method=method)
        self._config.update(kwargs)
        return self

    def fit(self, *args, **kwargs):
        tmp = self._config
        method = tmp['method']
        config = {k:tmp[k] for k in tmp if k != 'method'}
        optimizer = getattr(optimization, method)
        value, gradient, hessian = self._make_partials(*args, **kwargs)
        if method == 'linear':
            grad = gradient(None)
            hess = hessian(None)
            p, stats = optimizer(grad, hess, **config)
        elif method in self._needs_both:
            p, stats = optimizer(value, gradient, hessian, **config)
        elif method in self._no_hessian:
            p, stats = optimizer(value, gradient, **config)
        elif method in self._no_gradient:
            p, stats = optimizer(value, nparams=self.nparams, **config)
        self.p_ = p
        self.stats_ = stats
        return self

    @property
    def estimate_(self):
        return self.fmt_estimate(self.p_)

    def fmt_estimate(self, p):
        return p
