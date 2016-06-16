from __future__ import division
from future.builtins import super, range, object
import hashlib
import numpy as np

from fatiando.utils import safe_dot


class L2NormMisfit(object):
    def __init__(self, data, predict, islinear, jacobian=None, weights=None,
                 jacobian_cache=None):
        self.data = data
        self.predict = predict
        self.jacobian = jacobian
        self.weights = weights
        self.islinear = islinear
        self.cache = {'predict': {'hash':None, 'value': None},
                      'jacobian': {'hash':None, 'value': jacobian_cache}}
        
    def from_cache(self, p, func):
        if func == 'jacobian' and self.islinear:
            if self.cache[func]['value'] is None:
                self.cache[func]['value'] = getattr(self, func)(p)
        else:
            new_hash = hashlib.sha1(p).hexdigest()
            old_hash = self.cache[func]['hash']
            if old_hash is None or old_hash != new_hash:
                self.cache[func]['hash'] = new_hash
                self.cache[func]['value'] = getattr(self, func)(p)
        return self.cache[func]['value']            
    
    def value(self, p):
        pred = self.from_cache(p, 'predict')
        residuals = self.data - pred
        if self.weights is None:
            return np.linalg.norm(residuals, ord=2)**2
        else:
            return safe_dot(residuals.T, 
                            safe_dot(self.weights, residuals))
        
    def gradient(self, p):
        jac = self.from_cache(p, 'jacobian')
        pred = self.from_cache(p, 'predict')
        residuals = self.data - pred
        if self.weights is None:
            grad = -2*safe_dot(jac.T, residuals)
        else:
            grad = -2*safe_dot(jac.T, 
                               safe_dot(self.weights, residuals))
        return self._grad_to_1d(grad)
    
    def _grad_to_1d(self, grad):
        # Check if the gradient isn't a one column matrix
        if len(grad.shape) > 1:
            # Need to convert it to a 1d array so that hell won't break
            # loose
            grad = np.array(grad).ravel()
        return grad
    
    def gradient_at_null(self):
        assert self.islinear, 'Can only calculate gradient at the null vector for linear misfits'
        jac = self.from_cache(None, 'jacobian')
        if self.weights is None:
            grad = -2*safe_dot(jac.T, self.data)
        else:
            grad = -2*safe_dot(jac.T, 
                               safe_dot(self.weights, self.data))
        return self._grad_to_1d(grad)        
    
    def hessian(self, p):
        jac = self.from_cache(p, 'jacobian')
        if self.weights is None:
            return 2*safe_dot(jac.T, jac)
        else:
            return 2*safe_dot(jac.T, 
                               safe_dot(self.weights, jac))