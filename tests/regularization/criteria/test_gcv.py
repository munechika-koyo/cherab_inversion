# ruff: noqa: N802
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cherab.inversion.regularization import GCV, SVD
from cherab.inversion.regularization.criteria._base import Criterion


class TestGCV:
    def test_gcv(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        gcv = GCV()

        # try to compute gcv with some log10(lambda) values
        betas = np.logspace(-20, 2, num=500)
        for beta in betas:
            gcv.gcv(svd, beta)

    def test_solve(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        gcv = GCV()

        bounds = (-20.0, 2.0)
        stepsize = 10

        svd.solve(gcv, bounds=bounds, stepsize=stepsize, disp=True)

        # TODO: check the solution
        # this test is not passed because the gcv optimization is not converged at this ill-posed
        # problem. we need to find a better test case.

    def test_plot_gcv(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        gcv = GCV()
        svd.solve(gcv, bounds=(-20.0, 2.0), stepsize=10, disp=True)

        bounds = (-20.0, 2.0)
        n_beta = 500

        ax, fig = gcv.plot_gcv(svd, bounds=bounds, n_beta=n_beta)
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_gcv_on_existing_axes(self, test_data, computed_svd):
        """Passing an existing axes returns just the Axes (fig=None branch)."""
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        gcv = GCV()
        svd.solve(gcv, bounds=(-20.0, 2.0), stepsize=10, disp=True)

        _, existing_ax = plt.subplots()
        result = gcv.plot_gcv(svd, axes=existing_ax)
        assert isinstance(result, Axes)
        plt.close("all")

    def test_plot_gcv_no_solver_raises_RuntimeError(self):
        """Calling plot_gcv without solver or prior optimize raises RuntimeError."""
        gcv = GCV()
        with pytest.raises(RuntimeError, match="No solver available"):
            gcv.plot_gcv()

    def test_plot_gcv_uses_cached_solver(self, test_data, computed_svd):
        """plot_gcv without solver uses _solver cached by optimize."""
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        gcv = GCV()
        svd.solve(gcv, bounds=(-20.0, 2.0), stepsize=10, disp=False)
        # _solver is now cached; call plot_gcv() without explicit solver
        result = gcv.plot_gcv()
        plt.close("all")
        assert result is not None


def test_criterion_base_init():
    """Criterion.__init__ sets _solver = None (covers _base.py line 22)."""

    class MyCriterion(Criterion):
        def optimize(self, solver, bounds, stepsize=10, **kwargs):
            return 0.0, None

    crit = MyCriterion()
    assert crit._solver is None
