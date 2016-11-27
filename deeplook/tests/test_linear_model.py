from __future__ import division
from future.builtins import super, range
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
        return self


# Add damping regularization to test if that works as well.
class RegressionDamped(Regression):

    def __init__(self, damping=0):
        super().__init__()
        self.damping = damping

    def fit(self, x, y, weights=None, jacobian=None):
        misfit = self.make_misfit(data=y, args=[x], weights=weights,
                                  jacobian=jacobian)
        regul = []
        if self.damping > 0:
            regul.append([self.damping, Damping(nparams=2)])
        obj = self.make_objective(misfit, regul)
        self.p_ = self.optimizer.minimize(obj)
        return self


# Make some test data
x = np.linspace(0, 1000, 100)
true_p = np.array([3, 500])
true_a, true_b = true_p
y = true_a*x + true_b
y_noisy, error = contaminate(y, 0.01, percent=True, return_stddev=True,
                             seed=42)
y_outlier = np.array(y)
y_outlier[20] += 0.5*y[20]
y_outlier[50] -= y[50]
y_outlier[70] += y[70]


def test_fit_clean():
    "Fit a line to noise-free data"
    inv = Regression()
    inv.fit(x, y)
    residuals = y - inv.predict(x)
    npt.assert_allclose(inv.predict(x), y)
    npt.assert_allclose(inv.p_, true_p)
    npt.assert_almost_equal(np.zeros_like(x), residuals, decimal=10)
    npt.assert_almost_equal(0, residuals.mean(), decimal=10)
    npt.assert_allclose(inv.score(x, y), 1)


def test_fit_noisy():
    "Fit a line to noisy data"
    inv = Regression()
    inv.fit(x, y_noisy)
    residuals = y_noisy - inv.predict(x)
    npt.assert_allclose(inv.predict(x), y, rtol=1e-2)
    npt.assert_allclose(inv.p_, true_p, rtol=1e-2)
    npt.assert_allclose(residuals.std(), error, rtol=1e-1)
    npt.assert_almost_equal(0, residuals.mean(), decimal=10)
    npt.assert_allclose(inv.predict(x), y, rtol=1e-2)
    npt.assert_almost_equal(inv.score(x, y), 1, decimal=5)


def test_fit_outlier():
    "Fit a line to data with outliers using fit_reweighted"
    inv = Regression()
    inv.fit_reweighted(x, y_outlier)
    residuals = y_outlier - inv.predict(x)
    npt.assert_allclose(inv.predict(x), y)
    npt.assert_allclose(inv.p_, true_p)
    npt.assert_allclose(inv.score(x, y), 1)
    # The biggest residuals should be on the outliers in order
    largest = np.argsort(np.abs(residuals))
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
    residuals = y_noisy - inv.predict(x)
    npt.assert_allclose(inv.predict(x), y, rtol=1e-2)
    npt.assert_allclose(inv.p_, true_p, rtol=1e-2)
    npt.assert_allclose(residuals.std(), error, rtol=1e-1)
    npt.assert_almost_equal(0, residuals.mean(), decimal=8)
    npt.assert_almost_equal(inv.score(x, y), 1, decimal=5)


def test_optimizers():
    "Fit a line to noise-free data using different optimizers"
    def fit_n_test(opt):
        inv = Regression().config(optimizer=opt).fit(x, y)
        residuals = y - inv.predict(x)
        npt.assert_allclose(inv.p_, true_p, rtol=1e-2)
        npt.assert_allclose(inv.predict(x), y, rtol=1e-2)
        npt.assert_almost_equal(np.zeros_like(x), residuals, decimal=3)
        npt.assert_almost_equal(0, residuals.mean(), decimal=3)
        npt.assert_almost_equal(inv.score(x, y), 1, decimal=5)
    initial = [5, 100]
    fit_n_test(Newton(initial=initial, tol=1e1))
    fit_n_test(LevMarq(initial=initial, tol=1e-1))
    fit_n_test(ACOR(bounds=[0, 5, 100, 800], nparams=2, seed=30))
    fit_n_test(ScipyOptimizer('Nelder-Mead', x0=initial))
    fit_n_test(ScipyOptimizer('BFGS', x0=initial))
    # Can't get this to work, it just won't take a step
    # fit_n_test(SteepestDescent(initial=initial))


def test_fit_outlier_optimizers():
    "Fit data with outliers using fit_reweighted and non-linear optimization"
    def fit_n_test(opt):
        inv = Regression().config(optimizer=opt)
        inv.fit_reweighted(x, y_outlier)
        residuals = y_outlier - inv.predict(x)
        npt.assert_allclose(inv.p_, true_p, rtol=1e-2)
        npt.assert_allclose(inv.predict(x), y, rtol=1e-2)
        npt.assert_almost_equal(inv.score(x, y), 1, decimal=5)
        # The biggest residuals should be on the outliers in order
        largest = np.argsort(np.abs(residuals))
        assert largest[-1] == 70
        assert largest[-2] == 50
        assert largest[-3] == 20
    initial = [5, 100]
    fit_n_test(Newton(initial=initial, tol=1e1))
    fit_n_test(LevMarq(initial=initial, tol=1e-3))
    fit_n_test(ACOR(bounds=[0, 1000], nparams=2, seed=30))
    fit_n_test(ScipyOptimizer('Nelder-Mead', x0=initial))
    fit_n_test(ScipyOptimizer('BFGS', x0=initial))


def test_fit_damped_optimizers():
    "Fit noisy data using damping regularization with non-linear optimization"
    # Using little damping shouldn't change the results
    # This tests if adding damping doesn't break the existing code, not really
    # if it gives the right results. Need an example that actually needs
    # regularization for this.
    def fit_n_test(opt):
        inv = RegressionDamped(damping=1e-10).config(optimizer=opt)
        inv.fit(x, y_noisy)
        residuals = y_noisy - inv.predict(x)


        npt.assert_allclose(inv.predict(x), y, rtol=1e-2)
        npt.assert_allclose(inv.p_, true_p, rtol=1e-2)
        npt.assert_allclose(residuals.std(), error, rtol=1e-1)
        npt.assert_almost_equal(0, residuals.mean(), decimal=3)
        npt.assert_almost_equal(inv.score(x, y), 1, decimal=5)
    initial = [5, 100]
    fit_n_test(Newton(initial=initial, tol=1e-5))
    fit_n_test(LevMarq(initial=initial, tol=1e-5))
    fit_n_test(ACOR(bounds=[1, 10, 100, 1000], nparams=2, seed=30))
    fit_n_test(ScipyOptimizer('Nelder-Mead', x0=initial))
    fit_n_test(ScipyOptimizer('BFGS', x0=initial))


def test_jacobian_warm_start():
    "Pass a pre-computed Jacobian to fit"
    A = np.ones((x.size, 2))
    A[:, 0] = x
    inv = Regression()
    inv.fit(x, y, jacobian=A)
    residuals = y - inv.predict(x)
    npt.assert_allclose(inv.p_, true_p)
    npt.assert_allclose(inv.predict(x), y, rtol=1e-2)
    npt.assert_almost_equal(np.zeros_like(x), residuals, decimal=10)
    npt.assert_almost_equal(0, residuals.mean(), decimal=10)
    npt.assert_allclose(inv.score(x, y), 1)
    # If the jacobian is divided by 2, the estimated parameters should be
    # double the true value to compensate.
    half_jacobian = A*0.5
    inv.fit(x, y, jacobian=half_jacobian)
    npt.assert_allclose(inv.p_, true_p*2)
    # The predicted should not return the correct values because the parameters
    # are wrong.
    npt.assert_array_less(y, inv.predict(x))
    # Calculate the true predicted by multiplying the Jacobian by p and check
    # the fit.
    predicted = half_jacobian.dot(inv.p_)
    residuals = y -  predicted
    npt.assert_allclose(predicted, y, rtol=1e-2)
    npt.assert_almost_equal(np.zeros_like(x), residuals, decimal=10)
    npt.assert_almost_equal(0, residuals.mean(), decimal=10)
