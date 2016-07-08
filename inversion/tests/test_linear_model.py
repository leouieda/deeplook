from __future__ import division
import numpy as np
import numpy.testing as npt
from fatiando.utils import contaminate
from .. import LinearModel, Damping, \
               ScipyOptimizer, LevMarq, ACOR, Newton, SteepestDescent


# Make a test inversion using the simplest model possible: fitting a straight
# line.
class Regression(LinearModel):

    def predict(self, x):
        a, b = self.p_
        return a*x + b

    def jacobian(self, x):
        A = np.empty((x.size, 2))
        A[:, 0] = x
        A[:, 1] = 1
        return A

    def fit(self, x, y, weights=None, jacobian=None):
        misfit = self.make_misfit(data=y, args=[x], weights=weights,
                                  jacobian=jacobian)
        self.p_ = self.optimizer.minimize(misfit)
        self.residuals_ = y - self.predict(x)
        return self


# Add damping regularization to test if that works as well.
class RegressionDamped(Regression):

    def __init__(self, damping=0):
        self.damping = damping

    def fit(self, x, y, weights=None, jacobian=None):
        misfit = self.make_misfit(data=y, args=[x], weights=weights,
                                  jacobian=jacobian)
        regul = []
        if self.damping > 0:
            regul.append([self.damping, Damping(nparams=2)])
        obj = self.make_objective(misfit, regul)
        self.p_ = self.optimizer.minimize(obj)
        self.residuals_ = y - self.predict(x)
        return self


# Make some test data
x = np.linspace(0, 1000, 100)
true_p = np.array([20, 500])
true_a, true_b = true_p
y = true_a*x + true_b
y_noisy, error = contaminate(y, 0.05, percent=True, return_stddev=True,
                             seed=42)
y_outlier = np.array(y)
y_outlier[20] += 0.5*y[20]
y_outlier[50] -= y[50]
y_outlier[70] += y[70]


def test_fit_clean():
    "Fit a line to noise-free data"
    inv = Regression()
    inv.fit(x, y)
    npt.assert_allclose(inv.predict(x), y)
    npt.assert_allclose(inv.p_, true_p)
    npt.assert_allclose(inv.residuals_, np.zeros_like(x))
    npt.assert_allclose(inv.residuals_.mean(), 0)
    npt.assert_allclose(inv.score(x, y), 1)


def test_fit_noisy():
    "Fit a line to noisy data"
    inv = Regression()
    inv.fit(x, y_noisy)
    npt.assert_allclose(inv.predict(x), y)
    npt.assert_allclose(inv.p_, true_p)
    npt.assert_allclose(inv.residuals_.std(), error)
    npt.assert_allclose(inv.residuals_.mean(), 0)
    npt.assert_allclose(inv.score(x, y), 1)


def test_fit_outlier():
    "Fit a line to data with outliers using fit_reweighted"
    inv = Regression()
    inv.fit_reweighted(x, y_outlier)
    npt.assert_allclose(inv.predict(x), y)
    npt.assert_allclose(inv.p_, true_p)
    npt.assert_allclose(inv.score(x, y), 1)
    # The biggest residuals should be on the outliers in order
    largest = np.argsort(inv.residuals_)
    assert largest[-1] == 70
    assert largest[-2] == 50
    assert largest[-3] == 20


def test_fit_damped():
    "Fit a line to noisy data using damping regularization"
    # Using little damping shouldn't change the results
    # This tests if adding damping doesn't break the existing code, not really
    # if it gives the right results. Need an example that actually needs
    # regularization for this.
    inv = RegressionDamped(damping=1e-10)
    inv.fit(x, y_noisy)
    npt.assert_allclose(inv.predict(x), y)
    npt.assert_allclose(inv.p_, true_p)
    npt.assert_allclose(inv.residuals_.std(), error)
    npt.assert_allclose(inv.residuals_.mean(), 0)
    npt.assert_allclose(inv.score(x, y), 1)


def test_optimizers():
    "Fit a line to noise-free data using different optimizers"
    def fit_n_test(opt):
        inv = Regression().config(optimizer=opt).fit(x, y)
        npt.assert_allclose(inv.predict(x), y)
        npt.assert_allclose(inv.p_, true_p)
        npt.assert_allclose(inv.residuals_, np.zeros_like(x))
        npt.assert_allclose(inv.residuals_.mean(), 0)
        npt.assert_allclose(inv.score(x, y), 1)
    initial = [5, 100]
    fit_n_test(Newton(initial=initial))
    fit_n_test(LevMarq(initial=initial))
    fit_n_test(SteepestDescent(initial=initial))
    fit_n_test(ACOR(bounds=[-1000, 1000], nparams=2))
    fit_n_test(ScipyOptimizer('Nelder-Mead', x0=initial))
    fit_n_test(ScipyOptimizer('BFGS', x0=initial))


def test_jacobian_warm_start():
    "Pass a pre-computed Jacobian to fit"
    A = np.ones((x.size, 2))
    A[:, 0] = x
    inv = Regression()
    inv.fit(x, y, jacobian=A)
    npt.assert_allclose(inv.predict(x), y)
    npt.assert_allclose(inv.p_, true_p)
    npt.assert_allclose(inv.residuals_, np.zeros_like(x))
    npt.assert_allclose(inv.residuals_.mean(), 0)
    npt.assert_allclose(inv.score(x, y), 1)
    # If the jacobian is divided by 2, the estimated parameters should be half
    # the true value.  All the rest should be the same.
    inv.fit(x, y, jacobian=A*0.5)
    npt.assert_allclose(inv.predict(x), y)
    npt.assert_allclose(inv.p_, true_p/2)
    npt.assert_allclose(inv.residuals_, np.zeros_like(x))
    npt.assert_allclose(inv.residuals_.mean(), 0)
    npt.assert_allclose(inv.score(x, y), 1)
