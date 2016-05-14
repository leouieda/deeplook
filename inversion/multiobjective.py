from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
import copy
from abc import ABCMeta, abstractmethod
import numpy as np



def MultiObjective(object):

 def __init__(self, *args):
        self._components = self._unpack_components(args)
        self._have_fit = [i for i, c in enumerate(self._components)
                          if hasattr(c, 'fit')]
        self.size = len(self._components)
        self.p_ = None
        nparams = [obj.nparams for obj in self._components]
        assert all(nparams[0] == n for n in nparams[1:]), \
            "Can't add goals with different number of parameters:" \
            + ' ' + ', '.join(str(n) for n in nparams)
        self.nparams = nparams[0]
        if all(obj.islinear for obj in self._components):
            self.islinear = True
        else:
            self.islinear = False
        self._i = 0  # Tracker for indexing
        self.scale = 1

    def fit(self, *args):
        assert len(args) % len(self._have_fit) == 0, \
            'Number of arguments must be divisible by number of misfit funcs'

        for obj in self:
            obj.p_ = self.p_
        return self

    # Pass along the configuration in case the classes need to change something
    # depending on the optimization method.
    def config(self, *args, **kwargs):
        super().config(*args, **kwargs)
        for obj in self:
            if hasattr(obj, 'config'):
                obj.config(*args, **kwargs)
        return self

    config.__doc__ = OptimizerMixin.config.__doc__

    def _unpack_components(self, args):
        """
        Find all the MultiObjective elements in components and unpack them into
        a single list.
        This is needed so that ``D = A + B + C`` can be indexed as ``D[0] == A,
        D[1] == B, D[2] == C``. Otherwise, ``D[1]`` would be a
        ``MultiObjetive == B + C``.
        """
        components = []
        for comp in args:
            if isinstance(comp, MultiObjective):
                components.extend([c*comp.regul_param for c in comp])
            else:
                components.append(comp)
        return components

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self._components[i]

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        """
        Used for iterating over the MultiObjetive.
        """
        if self._i >= self.size:
            raise StopIteration
        comp = self.__getitem__(self._i)
        self._i += 1
        return comp

    def fmt_estimate(self, p):
        """
        Format the current estimated parameter vector into a more useful form.
        Will call the ``fmt_estimate`` method of the first component goal
        function (the first term in the addition that created this object).
        """
        return self._components[0].fmt_estimate(p)

    def value(self, p):
        """
        Return the value of the multi-objective function.
        This will be the sum of all goal functions that make up this
        multi-objective.
        Parameters:
        * p : 1d-array
            The parameter vector.
        Returns:
        * result : scalar (float, int, etc)
            The sum of the values of the components.
        """
        return self.scale*sum(obj.value(p) for obj in self)

    def gradient(self, p):
        """
        Return the gradient of the multi-objective function.
        This will be the sum of all goal functions that make up this
        multi-objective.
        Parameters:
        * p : 1d-array
            The parameter vector.
        Returns:
        * result : 1d-array
            The sum of the gradients of the components.
        """
        return self.scale*sum(obj.gradient(p) for obj in self)

    def hessian(self, p):
        """
        Return the hessian of the multi-objective function.
        This will be the sum of all goal functions that make up this
        multi-objective.
        Parameters:
        * p : 1d-array
            The parameter vector.
        Returns:
        * result : 2d-array
            The sum of the hessians of the components.
        """
        return self.scale*sum(obj.hessian(p) for obj in self)
