from __future__ import division
from future.builtins import super, range, object
import hashlib
import numpy as np


class CachedMethod(object):
    pass

# class CachedMethod(object):
    # def __init__(self, instance, meth):
        # self.arg_hashes = None
        # self.kwarg_hashes = None
        # self.cache = None
        # self.instance = instance
        # self.meth = meth
        # method = getattr(self.instance.__class__, self.meth)
        # setattr(self, '__doc__', getattr(method, '__doc__'))

    # def hard_reset(self):
        # """
        # Delete the cached values.
        # """
        # self.cache = None
        # self.arg_hashes = None
        # self.kwarg_hashes = None

    # def __call__(self, *args, **kwargs):
        # arg_hashes = [hashlib.sha1(x).hexdigest() for x in args]
        # kwarg_hashes = {k:hashlib.sha1(kwargs[k]).hexdigest() for k in kwargs}
        # changed = False


            # if self.cache is None or self.array_hash != p_hash:
                # # Update the cache
                # self.array_hash = p_hash
                # # Get the method from the class because the instance will overwrite
                # # it with the CachedMethod instance.
                # method = getattr(self.instance.__class__, self.meth)
                # self.cache = method(self.instance, p)
        # return self.cache
