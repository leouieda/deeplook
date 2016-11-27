from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
import numpy as np
import scipy.sparse as sp

from fatiando.utils import safe_dot


class Damping(object):

    def __init__(self, nparams):
        self.nparams = nparams
        self.islinear = True
        self._identity = sp.identity(nparams).tocsr()

    def value(self, p):
        return np.linalg.norm(p)**2

    def gradient(self, p):
        return 2*p

    def gradient_at_null(self):
        return 0

    def hessian(self, p):
        return 2*self._identity


class Smoothness(object):

    def __init__(self, diffs):
        self.islinear = True
        self.diffs = diffs
        self.RtR = safe_dot(diffs.T, diffs)

    def value(self, p):
        return safe_dot(p.T, safe_dot(self.RtR, p))

    def gradient(self, p):
        return 2*safe_dot(self.RtR, p)

    def gradient_at_null(self):
        return 0

    def hessian(self, p):
        return 2*self.RtR


class Smoothness1D(Smoothness):
    def __init__(self, nparams):
        super().__init__(fd1d(nparams))


def fd1d(size):
    i = list(range(size - 1)) + list(range(size - 1))
    j = list(range(size - 1)) + list(range(1, size))
    v = [1] * (size - 1) + [-1] * (size - 1)
    return sp.coo_matrix((v, (i, j)), (size - 1, size)).tocsr()


class TotalVariation(object):
    def __init__(self, diffs, beta=1e-5):
        if beta <= 0:
            raise ValueError("Invalid beta={:g}. Must be > 0".format(beta))
        self.islinear = False
        self.beta = beta
        self.diffs = diffs

    def value(self, p):
        return np.linalg.norm(safe_dot(self.diffs, p), 1)

    def hessian(self, p):
        derivs = safe_dot(self.diffs, p)
        q = self.beta/((derivs**2 + self.beta)**1.5)
        q_matrix = sp.diags(q, 0).tocsr()
        return safe_dot(self.diffs.T, q_matrix*self.diffs)

    def gradient(self, p):
        derivs = safe_dot(self.diffs, p)
        q = derivs/np.sqrt(derivs**2 + self.beta)
        grad = safe_dot(self.diffs.T, q)
        if len(grad.shape) > 1:
            grad = np.array(grad.T).ravel()
        return grad


class TotalVariation1D(TotalVariation):
    def __init__(self, nparams, beta=1e-5):
        super().__init__(fd1d(nparams), beta=1e-5)
