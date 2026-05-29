# ruff: noqa: N802
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cherab.inversion import SVD, Lcurve
from cherab.inversion.regularization import TSVD


class TestLcurve:
    def test_carvature(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        lcurve = Lcurve()

        # try to compute curvature with some log10(lambda) values
        betas = np.logspace(-20, 2, num=500)
        for beta in betas:
            lcurve.curvature(svd, beta)

    def test_curvature_eta_dif_zero_returns_neg_inf(self, test_data, computed_svd):
        """TSVD has eta_diff == 0 everywhere; curvature should return -inf (line 74)."""
        u, sigma, vh = computed_svd
        tsvd = TSVD(sigma, u, vh.T, data=test_data.b)
        lcurve = Lcurve()
        result = lcurve.curvature(tsvd, 1.0)
        assert result == -np.inf

    def test_solve(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        lcurve = Lcurve()

        bounds = (-20.0, 2.0)
        stepsize = 10

        sol, status = svd.solve(lcurve, bounds=bounds, stepsize=stepsize, disp=True)

        assert status["success"] is True
        np.testing.assert_allclose(sol, test_data.x_true, rtol=0, atol=1.0)

    @pytest.mark.parametrize(("scatter_plot", "scatter_annotate"), [(None, False), (5, True)])
    def test_plot_L_curve(self, test_data, computed_svd, scatter_plot, scatter_annotate):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        lcurve = Lcurve()
        svd.solve(lcurve, bounds=(-20.0, 2.0), stepsize=10, disp=True)

        bounds = (-20.0, 2.0)
        n_beta = 500

        ax, fig = lcurve.plot_L_curve(
            svd,
            bounds=bounds,
            n_beta=n_beta,
            scatter_plot=scatter_plot,
            scatter_annotate=scatter_annotate,
        )
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_L_curve_on_existing_axes(self, test_data, computed_svd):
        """Passing existing axes returns just Axes (fig=None branch, line 254)."""
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        lcurve = Lcurve()
        svd.solve(lcurve, bounds=(-20.0, 2.0), stepsize=10, disp=True)

        _, existing_ax = plt.subplots()
        result = lcurve.plot_L_curve(svd, axes=existing_ax)
        assert isinstance(result, Axes)
        plt.close("all")

    def test_plot_curvature(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        lcurve = Lcurve()
        svd.solve(lcurve, bounds=(-20.0, 2.0), stepsize=10, disp=True)

        bounds = (-20.0, 2.0)
        n_beta = 500

        ax, fig = lcurve.plot_curvature(svd, bounds=bounds, n_beta=n_beta)
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)
        plt.close("all")

    def test_plot_curvature_on_existing_axes(self, test_data, computed_svd):
        """Passing existing axes returns just Axes (fig=None branch, line 350)."""
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        lcurve = Lcurve()
        svd.solve(lcurve, bounds=(-20.0, 2.0), stepsize=10, disp=True)

        _, existing_ax = plt.subplots()
        result = lcurve.plot_curvature(svd, axes=existing_ax)
        assert isinstance(result, Axes)
        plt.close("all")

    def test_plot_L_curve_no_solver_raises_RuntimeError(self):
        """plot_L_curve without solver or prior optimize raises RuntimeError."""
        lcurve = Lcurve()
        with pytest.raises(RuntimeError, match="No solver available"):
            lcurve.plot_L_curve()

    def test_plot_curvature_no_solver_raises_RuntimeError(self):
        """plot_curvature without solver or prior optimize raises RuntimeError."""
        lcurve = Lcurve()
        with pytest.raises(RuntimeError, match="No solver available"):
            lcurve.plot_curvature()
