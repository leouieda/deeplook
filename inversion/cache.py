from __future__ import division
from future.builtins import super, range, object
import hashlib
import numpy as np


class CachedMethod(object):

    def __init__(self, instance, method, ignored=None):
        if ignored is None:
            ignored = []
        self.ignored = ignored
        self.instance = instance
        self.method = method
        meth = getattr(self.instance.__class__, self.method)
        setattr(self, '__doc__', getattr(meth, '__doc__'))
        self.reset()

    def reset(self):
        """
        Delete the cached values.
        """
        self.cache = None
        self.arg_hashes = None
        self.kw_hashes = None

    def _update_cache(self, arg_hashes, kw_hashes, value):
        self.arg_hashes = arg_hashes
        self.kw_hashes = kw_hashes
        self.cache = value

    def _inputs_changed(self, arg_hashes, kw_hashes):
        if arg_hashes != self.arg_hashes:
            return True
        if kw_hashes != self.kw_hashes:
            return True
        return False

    def __call__(self, *args, **kwargs):
        arg_hashes = [hashlib.sha1(x).hexdigest()
                      for x in args if x is not None]
        kw_hashes = {k:hashlib.sha1(kwargs[k]).hexdigest()
                     for k in kwargs
                     if kwargs[k] is not None and k not in self.ignored}
        if self.cache is None or self._inputs_changed(arg_hashes, kw_hashes):
            func = getattr(self.instance.__class__, self.method)
            value = func(self.instance, *args, **kwargs)
            self._update_cache(arg_hashes, kw_hashes, value)
        return self.cache
