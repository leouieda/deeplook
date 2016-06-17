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
            stagnation = True
            diag = sp.diags(safe_diagonal(hess), 0).tocsr()
            for step in range(self.maxsteps):
                newp = p + safe_solve(hess + lamb*diag, -grad)
                newvalue = objective.value(newp)
                if newvalue >= value:
                    if lamb < 1e15:
                        lamb = lamb*self.dlamb
                else:
                    if lamb > 1e-15:
                        lamb = lamb/self.dlamb
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
                stop = (newvalue > value
                        or abs((newvalue - value)/value) < self.tol)
                p = newp
                value = newvalue
                # Getting inside here means that I could take a step, so this is
                # where the yield goes.
                stats['objective'].append(value)
                stats['iterations'] += 1
                stats['step_attempts'].append(step + 1)
                stats['step_size'].append(lamb)
            if stop:
                break
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


class ACOR(object):
    def __init__(self, bounds, nparams, nants=None, archive_size=None,
                 maxit=1000, diverse=0.5, evap=0.85, seed=None):
        self.nparams = nparams
        # Set the defaults for number of ants and archive size
        if nants is None:
            self.nants = nparams
        else:
            self.nants = nants
        if archive_size is None:
            self.archive_size = 10*self.nants
        else:
            self.archive_size = archive_size
        self.bounds = np.array(bounds)
        self.maxit = maxit
        self.diverse = diverse
        self.evap = evap
        self.seed = seed

    def minimize(self, objective):
        stats = dict(method="Ant Colony Optimization for Continuous Domains",
                     iterations=0,
                     objective=[])
        np.random.seed(self.seed)
        bounds = self.bounds
        archive_size = self.archive_size
        nparams = self.nparams
        nants = self.nants
        maxit = self.maxit
        diverse = self.diverse
        evap = self.evap
        # Check is giving bounds for each parameter or one for all
        if bounds.size == 2:
            low, high = bounds
            archive = np.random.uniform(low, high, (archive_size, nparams))
        else:
            archive = np.empty((archive_size, nparams))
            bounds = bounds.reshape((nparams, 2))
            for i, bound in enumerate(bounds):
                low, high = bound
                archive[:, i] = np.random.uniform(low, high, archive_size)
        # Compute the initial pheromone trail based on the objective function
        trail = np.fromiter((objective.value(p) for p in archive),
                            dtype=np.float)
        # Sort the archive of initial random solutions
        order = np.argsort(trail)
        archive = [archive[i] for i in order]
        trail = trail[order].tolist()
        stats['objective'].append(trail[0])
        # Compute the weights (probabilities) of the solutions in the archive
        amp = 1. / (diverse * archive_size * np.sqrt(2 * np.pi))
        variance = 2 * diverse ** 2 * archive_size ** 2
        weights = amp * np.exp(-np.arange(archive_size) ** 2 / variance)
        weights /= np.sum(weights)
        for iteration in range(maxit):
            for k in range(nants):
                # Sample the probabilities to produce new estimates
                ant = np.empty(nparams, dtype=np.float)
                # 1. Choose a pdf from the archive
                pdf = np.searchsorted(
                    np.cumsum(weights),
                    np.random.uniform())
                for i in range(nparams):
                    # 2. Get the mean and stddev of the chosen pdf
                    mean = archive[pdf][i]
                    std = (evap / (archive_size - 1)) * np.sum(
                        abs(p[i] - archive[pdf][i]) for p in archive)
                    # 3. Sample the pdf until the samples are in bounds
                    for atempt in range(100):
                        ant[i] = np.random.normal(mean, std)
                        if bounds.size == 2:
                            low, high = bounds
                        else:
                            low, high = bounds[i]
                        if ant[i] >= low and ant[i] <= high:
                            break
                pheromone = objective.value(ant)
                # Place the new estimate in the archive
                place = np.searchsorted(trail, pheromone)
                if place == archive_size:
                    continue
                trail.insert(place, pheromone)
                trail.pop()
                archive.insert(place, ant)
                archive.pop()
            stats['objective'].append(trail[0])
            stats['iterations'] += 1
        self.stats = stats
        return archive[0]











def steepest(value, gradient, initial, maxit=1000, linesearch=True,
             maxsteps=30, beta=0.1, tol=10**-5):
    assert 1 > beta > 0, \
        "Invalid 'beta' parameter {}. Must be 1 > beta > 0".format(beta)
    stats = dict(method='Steepest Descent',
                 iterations=0,
                 objective=[],
                 step_attempts=[])
    p = np.array(initial, dtype=np.float)
    misfit = value(p)
    stats['objective'].append(misfit)
    if linesearch:
        stats['step_attempts'].append(0)
    # This is a mystic parameter of the Armijo rule
    alpha = 10 ** (-4)
    stagnation = False
    for iteration in range(maxit):
        grad = gradient(p)
        if linesearch:
            # Calculate now to avoid computing inside the loop
            gradnorm = np.linalg.norm(grad) ** 2
            stagnation = True
            # Determine the best step size
            for i in range(maxsteps):
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

