from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
import copy
from abc import ABCMeta, abstractmethod
import numpy as np

from .base import FitMixin


class OperatorMixin(object):

    _multiobjective = None

    @property
    def scale(self):
        return getattr(self, '_scale', 1)

    @scale.setter
    def scale(self, value):
        self._scale = value

    def copy(self, deep=False):
        if deep:
            obj = copy.deepcopy(self)
        else:
            obj = copy.copy(self)
        return obj

    def __add__(self, other):
        """
        Add two objective functions to make a MultiObjective.
        """
        assert self.nparams == other.nparams, \
            "Can't add goals with different number of parameters:" \
            + ' {}, {}'.format(self.nparams, other.nparams)
        # Make a shallow copy of self to return. If returned self, doing
        # 'a = b + c' a and b would reference the same object.
        if self._multiobjective is None:
            mo_cls = MultiObjective
        else:
            mo_cls = self._multiobjective
        res = mo_cls(self.copy(), other.copy())
        return res

    def __mul__(self, scale):
        """
        Multiply the objective function by a scalar to set the `scale`
        attribute.
        """
        # Make a shallow copy of self to return. If returned self, doing
        # 'a = 10*b' a and b would reference the same object.
        obj = self.copy()
        obj.scale = obj.scale*scale
        return obj

    def __rmul__(self, scale):
        return self.__mul__(scale)


class MultiObjective(FitMixin, OperatorMixin):

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
            self._config = dict(method='linear')
        else:
            self.islinear = False
            self._config = dict(method='Nelder-Mead', x0=np.ones(self.nparams))
        self._i = 0  # Tracker for indexing
        self.scale = 1

    def _make_partials(self, *args, **kwargs):
        packets = []
        nargs = len(args)//len(self._have_fit)
        for start in range(0, len(args), nargs):
            last = start + nargs - 1
            packets.append(dict(data=args[last],  # The data
                                args=args[start:last],  # The rest
                                kwargs=dict() # ignore the kwargs for now
                                ))
        def value(p):
            return self.value(p, packets)
        def gradient(p):
            return self.gradient(p, packets)
        def hessian(p):
            return self.hessian(p, packets)
        return value, gradient, hessian

    def fit(self, *args, **kwargs):
        """
        """
        assert len(args) % len(self._have_fit) == 0, \
            'Number of arguments must be divisible by number of misfit funcs'
        super().fit(*args, **kwargs)
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

    def value(self, p, packets):
        return self.scale*sum(o.value(p, t['data'], *t['args'], **t['kwargs'])
                              for o, t in zip(self, packets))

    def gradient(self, p, packets):
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
        return self.scale*sum(o.gradient(p, t['data'], *t['args'], **t['kwargs'])
                              for o, t in zip(self, packets))

    def hessian(self, p, packets):
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
        return self.scale*sum(o.hessian(p, t['data'], *t['args'], **t['kwargs'])
                              for o, t in zip(self, packets))
