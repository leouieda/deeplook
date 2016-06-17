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
