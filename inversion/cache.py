from __future__ import division
from future.builtins import super, range, object
import hashlib
import numpy as np


class CachedMethod(object):
    def __init__(self, instance, meth):
        self.hashes = None
        self.cache = None
        self.instance = instance
        self.meth = meth
        method = getattr(self.instance.__class__, self.meth)
        setattr(self, '__doc__', getattr(method, '__doc__'))

    def hard_reset(self):
        """
        Delete the cached values.
        """
        self.cache = None
        self.hashes = None

    def __call__(self, *args, **kwargs):
        if 'p' in kwargs:
            p = getattr(self.instance, 'p_')
        p_hash = hashlib.sha1(p).hexdigest()
        if self.cache is None or self.array_hash != p_hash:
            # Update the cache
            self.array_hash = p_hash
            # Get the method from the class because the instance will overwrite
            # it with the CachedMethod instance.
            method = getattr(self.instance.__class__, self.meth)
            self.cache = method(self.instance, p)
        return self.cache
