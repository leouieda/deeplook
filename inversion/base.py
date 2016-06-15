from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
import hashlib
import copy
from abc import ABCMeta, abstractmethod
import scipy.optimize

from . import optimization


class MemoizeDerivs(object):
    def __init__(self, func, with_hessian=False):
        self.func = func
        self.with_hessian = with_hessian
        self.grad = None
        if self.with_hessian:
            self.hess = None
        self.hash = None

    def __call__(self, p, *args):
        self.hash = hashlib.sha1(p).hexdigest()
        res = self.func(p, *args)
        assert len(res) <= 3, 'Too many return arguments'
        self.grad = res[1]
        if self.with_hessian:
            self.hess = res[2]
        return res[0]

    def gradient(self, p, *args):
        newhash = hashlib.sha1(p).hexdigest()
        if self.grad is None or newhash != self.hash:
            self(p, *args)
        return self.grad

    def hessian(self, p, *args):
        newhash = hashlib.sha1(p).hexdigest()
        if self.hess is None or newhash != self.hash:
            self(p, *args)
        return self.hess



class NonLinearModel(with_metaclass(ABCMeta)):

    def __init__(self, nparams, method, method_args):
        self.nparams = nparams
        self.islinear = False
        self.p_ = None
        self.stats_ = None
        self.method = method
        self.method_args = method_args

    @abstractmethod
    def predict(self):
        pass

    def copy(self, deep=False):
        if deep:
            obj = copy.deepcopy(self)
        else:
            obj = copy.copy(self)
        return obj

    def evaluate(self, p, data, aux, weights=None, value=True, grad=False,
                 hess=False):
        backup = self.p_
        self.p_ = p
        returned = []
        if value or grad:
            residuals = data - self.predict(*aux)
        if value:
            if weights is None:
                value = np.linalg.norm(residuals)**2
            else:
                value = safe_dot(residuals.T, safe_dot(weights, residuals))
            returned.append(value)
        if grad or hess:
            A = self.jacobian(*args)
        if grad:
            if weights is None:
                gradient = -2*safe_dot(jacobian.T, residuals)
            else:
                gradient = -2*safe_dot(jacobian.T, safe_dot(weights, residuals))
            # Check if the gradient isn't a one column matrix
            if len(gradient.shape) > 1:
                # Need to convert it to a 1d array so that hell won't break
                # loose
                gradient = np.array(gradient).ravel()
            returned.append(gradient)
        if hess:
            if weights is None:
                hessian = 2*safe_dot(jacobian.T, jacobian)
            else:
                hessian = 2*safe_dot(jacobian.T, safe_dot(weights, jacobian))
            returned.append(hessian)
        self.p_ = backup
        if len(returned) == 1:
            return returned[0]
        else:
            return returned

    def fit(self, data, aux, weights=None):
        sp = """Nelder-Mead Powell CG BFGS Newton-CG L-BFGS-B TNC COBYLA SLSQP
                dogleg trust-ncg""".split()
        sp_jac = """CG BFGS L-BFGS-B TNC SLSQP Newton-CG dogleg
                    trust-ncg""".split()
        sp_hess = 'Newton-CG dogleg trust-ncg'.split()
        if self.method in sp:
            args = [data, aux, weights]
            jac, hess = None, None
            if self.method in sp_jac:
                jac =

            fun = MemoizeDerivs(self.evaluate, with_hessian=False)
            res = scipy.optimize.minimize(fun=fun,
                                          method=method,
                                          jac=fun.gradient,
                                          **self.method_args)
            p = res.x
            stats = dict(method=method, iterations=res.nit)
        elif self.method in sp_hess:
        self.p_ = p
        self.stats_ = stats
        return self

