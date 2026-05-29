# ruff: noqa: N802
import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cherab.inversion.regularization import SVD
from cherab.inversion.regularization.criteria import PRESS

matplotlib.use("Agg")


class TestPRESS:
    def test_press(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        press = PRESS()

        betas = np.logspace(-20, 2, num=500)
        for beta in betas:
            val = press.press(svd, beta)
            assert isinstance(val, float)
            assert np.isfinite(val)

    def test_press_raises_without_data(self, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T)  # no data
        press = PRESS()

        with pytest.raises(RuntimeError, match="solver.data is not set"):
            press.press(svd, 1.0e-5)

    def test_solve(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        press = PRESS()

        bounds = (-20.0, 2.0)
        stepsize = 10

        svd.solve(press, bounds=bounds, stepsize=stepsize, disp=True)

    def test_plot_press(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        press = PRESS()
        svd.solve(press, bounds=(-20.0, 2.0), stepsize=10, disp=True)

        bounds = (-20.0, 2.0)
        n_beta = 500

        ax, fig = press.plot_press(svd, bounds=bounds, n_beta=n_beta)
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)

    def test_plot_press_on_existing_axes(self, test_data, computed_svd):
        import matplotlib.pyplot as plt

        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        press = PRESS()
        svd.solve(press, bounds=(-20.0, 2.0), stepsize=10, disp=True)

        _, ax = plt.subplots()
        result = press.plot_press(svd, axes=ax)
        assert isinstance(result, Axes)
        plt.close("all")

    def test_press_with_B(self, test_data, computed_svd):
        """Test press() with non-None B (covers if solver.B is not None branch, line 92)."""
        u, sigma, vh = computed_svd
        M = u.shape[0]
        B = np.eye(M)  # identity matrix as B
        svd = SVD(sigma, u, vh.T, B)
        svd.data = test_data.b
        press = PRESS()
        val = press.press(svd, 1.0e-5)
        assert np.isfinite(val)

    def test_plot_press_show_min_line_false(self, test_data, computed_svd):
        """Test plot_press with show_min_line=False (covers False branch at line 194)."""
        import matplotlib.pyplot as plt

        u, sigma, vh = computed_svd
        svd = SVD(sigma, u, vh.T, data=test_data.b)
        press = PRESS()
        svd.solve(press, bounds=(-20.0, 2.0), stepsize=10, disp=True)
        ax, fig = press.plot_press(svd, bounds=(-20.0, 2.0), n_beta=50, show_min_line=False)
        assert isinstance(ax, Axes)
        plt.close("all")
