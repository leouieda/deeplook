from __future__ import division
from future.builtins import object, super, range
import warnings
import numpy as np
import scipy.sparse as sp

from fatiando.utils import safe_solve, safe_diagonal, safe_dot


class LinearOptimizer(object):
    
    def __init__(self, precondition=True):
        self.precondition = precondition
        
    def minimize(self, objective):        
        hessian = objective.hessian(None)
        gradient = objective.gradient_at_null()
        if self.precondition:
            diag = np.abs(safe_diagonal(hessian))
            diag[diag < 10 ** -10] = 10 ** -10
            precond = sp.diags(1. / diag, 0).tocsr()
            hessian = safe_dot(precond, hessian)
            gradient = safe_dot(precond, gradient)
        p = safe_solve(hessian, -gradient)
        self.stats = dict(method="Linear solver")
        return p

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
        for iteration in xrange(self.maxit):
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
            if new_value > value or abs((new_value - value)/value) < self.tol:
                break
            value = new_value
        if iteration == self.maxit - 1:
            warnings.warn(
                'Exited because maximum iterations reached. '
                + 'Might not have achieved convergence. '
                + 'Try inscreasing the maximum number of iterations allowed.',
                RuntimeWarning)
        self.stats = stats
        return p
    
    
class LevMarq(object):
    
    def __init__(self, initial, tol=1e-5, maxit=30, maxsteps=20, lamb=10,
                 dlamb=2, precondition=True):
        self.initial = initial
        self.tol = tol
        self.maxit = maxit
        self.precondition = precondition
        self.maxsteps = maxsteps
        self.lamb = lamb
        self.dlamb = dlamb
        
    def minimize(self, objective):
        stats = dict(method="Newton's method",
                 iterations=0,
                 objective=[])
        p = np.array(self.initial)
        value = objective.value(p)
        stats['objective'].append(value)
        for iteration in xrange(self.maxit):
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
            if new_value > value or abs((new_value - value)/value) < self.tol:
                break
            value = new_value
        if iteration == self.maxit - 1:
            warnings.warn(
                'Exited because maximum iterations reached. '
                + 'Might not have achieved convergence. '
                + 'Try inscreasing the maximum number of iterations allowed.',
                RuntimeWarning)
        self.stats = stats
        return p
    
    
        
class ScipyOptimizer(object):
    def __init__(self, method, **kwargs):
        self.method = self._is_valid(method)
        self.args = kwargs
        
    def _is_valid(self, method):
        return method
    
    def minimize(self, objective):
        pass    












def levmarq(value, gradient, hessian, initial, maxit=30, maxsteps=20, lamb=10,
            dlamb=2, tol=10**-5, precondition=True):
    stats = dict(method="Levemberg-Marquardt",
                 iterations=0,
                 objective=[],
                 step_attempts=[],
                 step_size=[])
    p = numpy.array(initial, dtype=numpy.float)
    misfit = value(p)
    stats['objective'].append(misfit)
    stats['step_attempts'].append(0)
    stats['step_size'].append(lamb)
    for iteration in xrange(maxit):
        grad, hess = gradient(p), hessian(p)
        if precondition:
            diag = numpy.abs(safe_diagonal(hess))
            diag[diag < 10 ** -10] = 10 ** -10
            precond = scipy.sparse.diags(1. / diag, 0).tocsr()
            hess = safe_dot(precond, hess)
            grad = safe_dot(precond, grad)
        stagnation = True
        diag = scipy.sparse.diags(safe_diagonal(hess), 0).tocsr()
        for step in xrange(maxsteps):
            newp = p + safe_solve(hess + lamb*diag, -grad)
            newmisfit = value(newp)
            if newmisfit >= misfit:
                if lamb < 10 ** 15:
                    lamb = lamb*dlamb
            else:
                if lamb > 10 ** -15:
                    lamb = lamb/dlamb
                stagnation = False
                break
        if stagnation:
            stop = True
            warnings.warn(
                "Exited because couldn't take a step without increasing "
                + 'the objective function. '
                + 'Might not have achieved convergence. '
                + 'Try inscreasing the max number of step attempts allowed.',
                RuntimeWarning)
        else:
            stop = newmisfit > misfit or abs(
                (newmisfit - misfit) / misfit) < tol
            p = newp
            misfit = newmisfit
            # Getting inside here means that I could take a step, so this is
            # where the yield goes.
            stats['objective'].append(misfit)
            stats['iterations'] += 1
            stats['step_attempts'].append(step + 1)
            stats['step_size'].append(lamb)
        if stop:
            break
    if iteration == maxit - 1:
        warnings.warn(
            'Exited because maximum iterations reached. '
            + 'Might not have achieved convergence. '
            + 'Try inscreasing the maximum number of iterations allowed.',
            RuntimeWarning)
    return p, stats


def steepest(value, gradient, initial, maxit=1000, linesearch=True,
             maxsteps=30, beta=0.1, tol=10**-5):
    assert 1 > beta > 0, \
        "Invalid 'beta' parameter {}. Must be 1 > beta > 0".format(beta)
    stats = dict(method='Steepest Descent',
                 iterations=0,
                 objective=[],
                 step_attempts=[])
    p = numpy.array(initial, dtype=numpy.float)
    misfit = value(p)
    stats['objective'].append(misfit)
    if linesearch:
        stats['step_attempts'].append(0)
    # This is a mystic parameter of the Armijo rule
    alpha = 10 ** (-4)
    stagnation = False
    for iteration in xrange(maxit):
        grad = gradient(p)
        if linesearch:
            # Calculate now to avoid computing inside the loop
            gradnorm = numpy.linalg.norm(grad) ** 2
            stagnation = True
            # Determine the best step size
            for i in xrange(maxsteps):
                stepsize = beta**i
                newp = p - stepsize*grad
                newmisfit = value(newp)
                if newmisfit - misfit < alpha*stepsize*gradnorm:
                    stagnation = False
                    break
        else:
            newp = p - grad
            newmisfit = value(newp)
        if stagnation:
            stop = True
            warnings.warn(
                "Exited because couldn't take a step without increasing "
                + 'the objective function. '
                + 'Might not have achieved convergence. '
                + 'Try inscreasing the max number of step attempts allowed.',
                RuntimeWarning)
        else:
            stop = abs((newmisfit - misfit) / misfit) < tol
            p = newp
            misfit = newmisfit
            # Getting inside here means that I could take a step, so this is
            # where the yield goes.
            stats['objective'].append(misfit)
            stats['iterations'] += 1
            if linesearch:
                stats['step_attempts'].append(i + 1)
        if stop:
            break
    if iteration == maxit - 1:
        warnings.warn(
            'Exited because maximum iterations reached. '
            + 'Might not have achieved convergence. '
            + 'Try inscreasing the maximum number of iterations allowed.',
            RuntimeWarning)
    return p, stats


def acor(value, bounds, nparams, nants=None, archive_size=None, maxit=1000,
         diverse=0.5, evap=0.85, seed=None):
    stats = dict(method="Ant Colony Optimization for Continuous Domains",
                 iterations=0,
                 objective=[])
    numpy.random.seed(seed)
    # Set the defaults for number of ants and archive size
    if nants is None:
        nants = nparams
    if archive_size is None:
        archive_size = 10 * nants
    # Check is giving bounds for each parameter or one for all
    bounds = numpy.array(bounds)
    if bounds.size == 2:
        low, high = bounds
        archive = numpy.random.uniform(low, high, (archive_size, nparams))
    else:
        archive = numpy.empty((archive_size, nparams))
        bounds = bounds.reshape((nparams, 2))
        for i, bound in enumerate(bounds):
            low, high = bound
            archive[:, i] = numpy.random.uniform(low, high, archive_size)
    # Compute the initial pheromone trail based on the objective function value
    trail = numpy.fromiter((value(p) for p in archive), dtype=numpy.float)
    # Sort the archive of initial random solutions
    order = numpy.argsort(trail)
    archive = [archive[i] for i in order]
    trail = trail[order].tolist()
    stats['objective'].append(trail[0])
    # Compute the weights (probabilities) of the solutions in the archive
    amp = 1. / (diverse * archive_size * numpy.sqrt(2 * numpy.pi))
    variance = 2 * diverse ** 2 * archive_size ** 2
    weights = amp * numpy.exp(-numpy.arange(archive_size) ** 2 / variance)
    weights /= numpy.sum(weights)
    for iteration in xrange(maxit):
        for k in xrange(nants):
            # Sample the probabilities to produce new estimates
            ant = numpy.empty(nparams, dtype=numpy.float)
            # 1. Choose a pdf from the archive
            pdf = numpy.searchsorted(
                numpy.cumsum(weights),
                numpy.random.uniform())
            for i in xrange(nparams):
                # 2. Get the mean and stddev of the chosen pdf
                mean = archive[pdf][i]
                std = (evap / (archive_size - 1)) * numpy.sum(
                    abs(p[i] - archive[pdf][i]) for p in archive)
                # 3. Sample the pdf until the samples are in bounds
                for atempt in xrange(100):
                    ant[i] = numpy.random.normal(mean, std)
                    if bounds.size == 2:
                        low, high = bounds
                    else:
                        low, high = bounds[i]
                    if ant[i] >= low and ant[i] <= high:
                        break
            pheromone = value(ant)
            # Place the new estimate in the archive
            place = numpy.searchsorted(trail, pheromone)
            if place == archive_size:
                continue
            trail.insert(place, pheromone)
            trail.pop()
            archive.insert(place, ant)
            archive.pop()
        stats['objective'].append(trail[0])
        stats['iterations'] += 1
    return archive[0], stats
