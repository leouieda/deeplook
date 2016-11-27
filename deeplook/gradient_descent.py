from __future__ import division
from future.builtins import object, super, range
import warnings
import numpy as np
import scipy.sparse as sp

from fatiando.utils import safe_solve, safe_diagonal, safe_dot


class Newton(object):

    def __init__(self, initial, tol=1e-5, maxit=30, precondition=True):
        self.initial = initial
        self.tol = tol
        self.maxit = maxit
        self.precondition = precondition

    def minimize(self, objective):
        stats = dict(method="Newton's method",
                 iterations=0,
                 objective=[])
        p = np.array(self.initial)
        value = objective.value(p)
        stats['objective'].append(value)
        for iteration in range(self.maxit):
            grad = objective.gradient(p)
            hess = objective.hessian(p)
            if self.precondition:
                diag = np.abs(safe_diagonal(hess))
                diag[diag < 10 ** -10] = 10 ** -10
                precond = sp.diags(1. / diag, 0).tocsr()
                hess = safe_dot(precond, hess)
                grad = safe_dot(precond, grad)
            p = p + safe_solve(hess, -grad)
            new_value = objective.value(p)
            stats['objective'].append(new_value)
            stats['iterations'] += 1
            stop = (new_value > value
                    or abs(new_value - value) < self.tol*abs(value))
            if stop:
                break
            value = new_value
        if iteration == self.maxit - 1:
            warnings.warn(
                'Newton optimizer exited because maximum iterations reached. '
                + 'Might not have achieved convergence. '
                + 'Try increasing the maximum number of iterations allowed.',
                RuntimeWarning)
        self.stats = stats
        return p


class LevMarq(object):

    def __init__(self, initial, tol=1e-5, maxit=30, maxsteps=20, lamb=2,
                 dlamb=2, precondition=True):
        self.initial = initial
        self.tol = tol
        self.maxit = maxit
        self.precondition = precondition
        self.maxsteps = maxsteps
        self.lamb = lamb
        self.dlamb = dlamb

    def minimize(self, objective):
        stats = dict(method="Levemberg-Marquardt",
                     iterations=0,
                     objective=[],
                     step_attempts=[],
                     step_size=[])
        p = np.array(self.initial)
        value = objective.value(p)
        lamb = self.lamb
        stats['objective'].append(value)
        stats['step_attempts'].append(0)
        stats['step_size'].append(lamb)
        for iteration in range(self.maxit):
            grad = objective.gradient(p)
            hess = objective.hessian(p)
            if self.precondition:
                diag = np.abs(safe_diagonal(hess))
                diag[diag < 1e-10] = 1e-10
                precond = sp.diags(1/diag, 0).tocsr()
                hess = safe_dot(precond, hess)
                grad = safe_dot(precond, grad)
            diag = sp.diags(safe_diagonal(hess), 0).tocsr()
            # Try to take a step
            took_step = False
            for step in range(self.maxsteps):
                newp = p + safe_solve(hess + lamb*diag, -grad)
                newvalue = objective.value(newp)
                decrease = newvalue < value
                if not decrease:
                    if lamb < 1e15:
                        lamb = lamb*self.dlamb
                else:
                    if lamb > 1e-15:
                        lamb = lamb/self.dlamb
                    took_step = True
                    break
            if not took_step:
                stop = True
                warnings.warn(
                    "LevMarq optimizer exited because couldn't take a step "
                    + 'without increasing the objective function. '
                    + 'Might not have achieved convergence. '
                    + 'Try increasing the max number of step attempts.',
                    RuntimeWarning)
            else:
                stop = abs(newvalue - value) < self.tol*abs(value)
                p = newp
                value = newvalue
                stats['objective'].append(value)
                stats['iterations'] += 1
                stats['step_attempts'].append(step + 1)
                stats['step_size'].append(lamb)
            if stop:
                break
        if iteration == self.maxit - 1:
            warnings.warn(
                'LevMarq optmizer exited because maximum iterations reached. '
                + 'Might not have achieved convergence. '
                + 'Try increasing the maximum number of iterations allowed.',
                RuntimeWarning)
        self.stats = stats
        return p


class SteepestDescent(object):

    def __init__(self, initial, maxit=1000, linesearch=True, maxsteps=30,
                 beta=0.1, tol=1e-5):
        self.initial = initial
        self.maxit = maxit
        self.linesearch = linesearch
        self.maxsteps = maxsteps
        self.beta = beta
        self.tol = tol
        assert 1 > beta > 0, \
            "Invalid 'beta' parameter {}. Must be 1 > beta > 0".format(beta)

    def minimize(self, objective):
        stats = dict(method='Steepest Descent',
                     iterations=0,
                     objective=[],
                     step_attempts=[])
        p = np.array(self.initial)
        value = objective.value(p)
        stats['objective'].append(value)
        if self.linesearch:
            stats['step_attempts'].append(0)
        alpha = 1e-4  # This is a mystic parameter of the Armijo rule
        for iteration in range(self.maxit):
            grad = objective.gradient(p)
            if self.linesearch:
                # Calculate now to avoid computing inside the loop
                gradnorm = np.linalg.norm(grad)**2
                # Determine the best step size
                took_step = False
                for i in range(self.maxsteps):
                    stepsize = self.beta**i
                    newp = p - stepsize*grad
                    newvalue = objective.value(newp)
                    decreased = newvalue < value
                    good_step = newvalue - value < alpha*stepsize*gradnorm
                    if decreased and good_step:
                        took_step = True
                        break
            else:
                newp = p - grad
                newvalue = objective.value(newp)
                decreased = newvalue < value
                if decreased:
                    took_step = True
                else:
                    took_step = False
            if not took_step:
                stop = True
                warnings.warn(
                    "SteepestDescent optimizer exited because couldn't take a"
                    + " step without increasing the objective function. "
                    + 'Might not have achieved convergence. '
                    + 'Try increasing the max number of step attempts allowed.',
                    RuntimeWarning)
            else:
                stop = abs(newvalue - value) < self.tol*abs(value)
                p = newp
                value = newvalue
                stats['objective'].append(value)
                stats['iterations'] += 1
                if self.linesearch:
                    stats['step_attempts'].append(i + 1)
            if stop:
                break
        if iteration == self.maxit - 1:
            warnings.warn(
                'SteepestDescent optimizer exited because maximum iterations '
                + 'reached. Might not have achieved convergence. '
                + 'Try increasing the maximum number of iterations allowed.',
                RuntimeWarning)
        self.stats = stats
        return p
