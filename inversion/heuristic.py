from __future__ import division
from future.builtins import object, super, range
import warnings
import numpy as np


class ACOR(object):
    def __init__(self, bounds, nparams, nants=None, archive_size=None,
                 maxit=2000, diverse=0.5, evap=0.85, seed=None):
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
                    # if std is zero, this means that all solutions in the
                    # archive for this parameter are the same.
                    if std == 0:
                        ant[i] = mean
                    else:
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
