from __future__ import division
from future.builtins import super, range, object
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

from . import optimization
from .misfit import L2NormMisfit


class NonLinearModel(with_metaclass(ABCMeta)):
        
    def __init__(self, nparams, misfit='L2NormMisfit', optimizer='Nelder-Mead'):
        self.nparams = nparams
        self.islinear = False
        self.regularization = []
        self.misfit = None
        self.optimizer = None
        self.set_misfit(misfit)
        self.set_optimizer(optimizer)
        self.p_ = None
    
    @abstractmethod
    def predict(self, args):
        "Return data predicted by self.p_"
        pass
        
    def set_optimizer(self, optimizer):
        "Configure the optimization"
        if optimizer == 'linear':
            self.optimizer = LinearOptimizer()
        else:
            self.optimizer = optimizer
        return self
    
    def set_misfit(self, misfit):
        "Pass a different misfit function"
        if misfit == 'L2NormMisfit':
            self.misfit = L2NormMisfit
        else:
            self.misfit = misfit        
        return self
    
    def add_regularization(self, regul_param, regul):
        "Use the given regularization"
        self.regularization.append([regul_param, regul])
        return self
    
    def _make_partial(self, args, func):
        def partial(p):
            backup = self.p_
            self.p_ = p
            res = getattr(self, func)(*args)
            self.p_ = backup
            return res
        return partial
                
    def fit(self, args, data, weights=None, jacobian=None):
        "Fit the model to the given data"
        misfit_args = dict(data=data, 
                           predict=self._make_partial(args, 'predict'),
                           weights=weights, 
                           islinear=self.islinear,
                           jacobian_cache=jacobian)
        if hasattr(self, 'jacobian'):
            misfit_args['jacobian'] = self._make_partial(args, 'jacobian')
        misfit = self.misfit(**misfit_args)         
        objective = Objective([[1, misfit]] + self.regularization)
        self.p_ = self.optimizer.minimize(objective) # the estimated parameter vector
        return self

    
class LinearModel(NonLinearModel):
    def __init__(self, nparams, misfit='L2NormMisfit', optimizer='linear'):
        super().__init__(nparams, misfit=misfit, optimizer=optimizer)
        self.islinear = True
        
        
class Objective(object):
    def __init__(self, components):
        self.components = components
    
    def value(self, p):
        return np.sum(lamb*comp.value(p) 
                       for lamb, comp in self.components)
    
    def gradient(self, p):
        return np.sum(lamb*comp.gradient(p) 
                       for lamb, comp in self.components)
    
    def gradient_at_null(self):
        return np.sum(lamb*comp.gradient_at_null() 
                       for lamb, comp in self.components)
    
    def hessian(self, p):
        return np.sum(lamb*comp.hessian(p) 
                       for lamb, comp in self.components)