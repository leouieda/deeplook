from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
import copy
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.sparse

from fatiando.utils import safe_dot

from .base import Objective


class Damping(Objective):

    def __init__(self, nparams):
        super().__init__(nparams, islinear=True)
        self.identity = scipy.sparse.identity(nparams).tocsr()

    def value(self, p):
        return self.scale*np.linalg.norm(p)**2

    def gradient(self, p):
        if p is None:
            return 0
        else:
            return self.scale*2*p

    def hessian(self, p):
        return (self.scale*2)*self.identity
