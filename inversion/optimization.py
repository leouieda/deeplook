from __future__ import division
import copy
import warnings
import numpy
import scipy.sparse

from fatiando.utils import safe_solve, safe_diagonal, safe_dot


def linear(hessian, gradient, precondition=True):
    if precondition:
        diag = numpy.abs(safe_diagonal(hessian))
        diag[diag < 10 ** -10] = 10 ** -10
        precond = scipy.sparse.diags(1. / diag, 0).tocsr()
        hessian = safe_dot(precond, hessian)
        gradient = safe_dot(precond, gradient)
    p = safe_solve(hessian, -gradient)
    return p, dict(method="Linear solver")


def newton(evaluate, initial, maxit=30, tol=10 ** -5, precondition=True):
    stats = dict(method="Newton's method",
                 iterations=0,
                 objective=[])
    p = numpy.array(initial, dtype=numpy.float)
    stats['objective'].append(misfit)
    misfit, grad, hess = evaluate(p)
    for iteration in xrange(maxit):
        if precondition:
            diag = numpy.abs(safe_diagonal(hess))
            diag[diag < 10 ** -10] = 10 ** -10
            precond = scipy.sparse.diags(1. / diag, 0).tocsr()
            hess = safe_dot(precond, hess)
            grad = safe_dot(precond, grad)
        p += safe_solve(hess, -grad)
        newmisfit, grad, hess = evaluate(p)
        stats['objective'].append(newmisfit)
        stats['iterations'] += 1
        if newmisfit > misfit or abs((newmisfit - misfit) / misfit) < tol:
            break
        misfit = newmisfit
    if iteration == maxit - 1:
        warnings.warn(
            'Exited because maximum iterations reached. '
            + 'Might not have achieved convergence. '
            + 'Try inscreasing the maximum number of iterations allowed.',
            RuntimeWarning)
    return p, stats


def levmarq(evaluate, initial, maxit=30, maxsteps=20, lamb=10, dlamb=2,
            tol=10**-5, precondition=True):
    stats = dict(method="Levemberg-Marquardt",
                 iterations=0,
                 objective=[],
                 step_attempts=[],
                 step_size=[])
    p = numpy.array(initial, dtype=numpy.float)
    misfit, grad, hess = evaluate(p)
    stats['objective'].append(misfit)
    stats['step_attempts'].append(0)
    stats['step_size'].append(lamb)
    for iteration in xrange(maxit):
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
            newmisfit, newgrad, newhess = evaluate(newp)
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
            grad = newgrad
            hess = newhess
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


def steepest(gradient, value, initial, maxit=1000, linesearch=True,
             maxsteps=30, beta=0.1, tol=10**-5):
    r"""
    Minimize an objective function using the Steepest Descent method.

    The increment to the initial estimate of the parameter vector
    :math:`\bar{p}` is calculated by (Kelley, 1999)

    .. math::

        \Delta\bar{p} = -\lambda\bar{g}

    where :math:`\lambda` is the step size and :math:`\bar{g}` is the gradient
    vector.

    The step size can be determined thought a line search algorithm using the
    Armijo rule (Kelley, 1999). In this case,

    .. math::

        \lambda = \beta^m

    where :math:`1 > \beta > 0` and :math:`m \ge 0` is an integer that controls
    the step size. The line search finds the smallest :math:`m` that satisfies
    the Armijo rule

    .. math::

        \phi(\bar{p} + \Delta\bar{p}) - \Gamma(\bar{p}) <
        \alpha\beta^m ||\bar{g}(\bar{p})||^2

    where :math:`\phi(\bar{p})` is the objective function evaluated at
    :math:`\bar{p}` and :math:`\alpha = 10^{-4}`.

    Parameters:

    * gradient : function
        A function that returns the gradient vector of the objective function
        when given a parameter vector.
    * value : function
        A function that returns the value of the objective function evaluated
        at a given parameter vector.
    * initial : 1d-array
        The initial estimate for the gradient descent.
    * maxit : int
        The maximum number of iterations allowed.
    * linesearch : True or False
        Whether or not to perform the line search to determine an optimal step
        size.
    * maxsteps : int
        The maximum number of times to try to take a step before giving up.
    * beta : float
        The base factor used to determine the step size in line search
        algorithm. Must be 1 > beta > 0.
    * tol : float
        The convergence criterion. The lower it is, the more steps are
        permitted.

    Yields:

    * i, estimate, stats:
        * i : int
            The current iteration number
        * estimate : 1d-array
            The current estimated parameter vector
        * stats : dict
            Statistics about the optimization so far. Keys:

            * method : stf
                The name of the optimization algorithm
            * iterations : int
                The total number of iterations so far
            * objective : list
                Value of the objective function per iteration. First value
                corresponds to the inital estimate
            * step_attempts : list
                Number of attempts at taking a step per iteration. First number
                is zero, reflecting the initial estimate. Will be empty if
                ``linesearch==False``.

    References:

    Kelley, C. T., 1999, Iterative methods for optimization: Raleigh: SIAM.

    """
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
            yield iteration, p, copy.deepcopy(stats)
        if stop:
            break
    if iteration == maxit - 1:
        warnings.warn(
            'Exited because maximum iterations reached. '
            + 'Might not have achieved convergence. '
            + 'Try inscreasing the maximum number of iterations allowed.',
            RuntimeWarning)


def acor(value, bounds, nparams, nants=None, archive_size=None, maxit=1000,
         diverse=0.5, evap=0.85, seed=None):
    """
    Minimize the objective function using ACO-R.

    ACO-R stands for Ant Colony Optimization for Continuous Domains (Socha and
    Dorigo, 2008).

    Parameters:

    * value : function
        Returns the value of the objective function at a given parameter vector
    * bounds : list
        The bounds of the search space. If only two values are given, will
        interpret as the minimum and maximum, respectively, for all parameters.
        Alternatively, you can given a minimum and maximum for each parameter,
        e.g., for a problem with 3 parameters you could give
        `bounds = [min1, max1, min2, max2, min3, max3]`.
    * nparams : int
        The number of parameters that the objective function takes.
    * nants : int
        The number of ants to use in the search. Defaults to the number of
        parameters.
    * archive_size : int
        The number of solutions to keep in the solution archive. Defaults to
        10 x nants
    * maxit : int
        The number of iterations to run.
    * diverse : float
        Scalar from 0 to 1, non-inclusive, that controls how much better
        solutions are favored when constructing new ones.
    * evap : float
        The pheromone evaporation rate (evap > 0). Controls how spread out the
        search is.
    * seed : None or int
        Seed for the random number generator.

    Yields:

    * i, estimate, stats:
        * i : int
            The current iteration number
        * estimate : 1d-array
            The current best estimated parameter vector
        * stats : dict
            Statistics about the optimization so far. Keys:

            * method : stf
                The name of the optimization algorithm
            * iterations : int
                The total number of iterations so far
            * objective : list
                Value of the objective function corresponding to the best
                estimate per iteration.

    """
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
    # Compute the inital pheromone trail based on the objetive function value
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
            # Sample the propabilities to produce new estimates
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
        yield iteration, archive[0], copy.deepcopy(stats)
