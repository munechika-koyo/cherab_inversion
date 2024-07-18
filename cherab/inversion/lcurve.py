"""Module for L-curve crietrion."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import _SVDBase
from .tools.scientific_format import parse_scientific_notation

__all__ = ["Lcurve"]


class Lcurve(_SVDBase):
    """L-curve criterion for regularization parameter optimization.

    The L-curve is a log-log plot of the residual norm versus the regularization norm.
    The L-curve criterion for tikhnov regularization gives the optimal regularization
    parameter :math:`\\lambda` as the corner point of the L-curve by maximizing the
    curvature of the L-curve.

    .. note::
        The theory and implementation of the L-curve criterion are described here_.

    .. _here: ../user/theory/lcurve.ipynb

    Parameters
    ----------
    *args, **kwargs
        Parameters are the same as :class:`~cherab.inversion.core._SVDBase`.

    Examples
    --------
    >>> lcurve = Lcurve(s, U, basis, data=data)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_L_curve(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        scatter_plot: int | None = None,
        scatter_annotate: bool = True,
        plot_lambda_opt: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plotting the L-curve in log-log scale.

        The points :math:`(\\sqrt{\\rho}, \\sqrt{\\eta})` are plotted in log-log scale.

        Parameters
        ----------
        fig : :obj:`~matplotlib.figure.Figure`, optional
            A matplotlib figure object, by default None.
        axes : :obj:`~matplotlib.axes.Axes`, optional
            A matplotlib Axes object, by default None.
        bounds : Collection[float | None], optional
            Boundary pair of log10 of regularization parameters, by default :obj:`.bounds`.
            If you set the bounds like ``(-10, None)``, the higher bound is set to
            :math:`\\log_{10}\\sigma_1^2`.
            Raise an error if a >= b in (a, b).
        n_beta : int, optional
            Nnumber of regularization parameters, by default 500.
        scatter_plot : int, optional
            Whether or not to plot some L-curve points as a 10 :sup:`x` format, by default None.
            If you want to manually define the number of points,
            enter the numbers like ``scatter_plot=10`` then around 10 points corresponding to
            10 :sup:`x` format are plotted.
        scatter_annotate : bool, optional
            Whether or not to annotate the scatter_points, by default True.
            This key argument is valid if only ``scatter_plot`` is not None.
        plot_lambda_opt : bool, optional
            Whether or not to plot the L-curve corner point, by default True.

        Returns
        -------
        fig : :obj:`~matplotlib.figure.Figure`
            A matplotlib figure object.
        axes : :obj:`~matplotlib.axes.Axes`
            A matplotlib Axes object.
        """
        # get bounds of log10 of regularization parameter
        bounds = self._generate_bounds(bounds)

        # define regularization parameters
        lambdas = np.logspace(*bounds, n_beta)

        # compute norms
        residual_norms = np.array([self.residual_norm(i) for i in lambdas])
        regularization_norms = np.array([self.regularization_norm(i) for i in lambdas])

        # validation
        if not isinstance(fig, Figure):
            fig = plt.figure()
        if not isinstance(axes, Axes):
            axes = fig.add_subplot()

        # plotting
        axes.loglog(residual_norms, regularization_norms, color="C0", zorder=0)

        # plot some points of L-curve and annotate with regularization parameters label
        if isinstance(scatter_plot, int) and scatter_plot > 0:
            a, b = np.ceil(bounds[0]), np.floor(bounds[1])
            interval = round((b - a) / scatter_plot)
            betas = 10 ** np.arange(a, b, interval)
            for beta in betas:
                x, y = self.residual_norm(beta), self.regularization_norm(beta)
                axes.scatter(
                    x,
                    y,
                    edgecolors="C0",
                    marker="o",
                    facecolor="none",
                    zorder=1,
                )
                if scatter_annotate is True:
                    _lambda_sci = parse_scientific_notation(f"{beta:.2e}", scilimits=(0, 0))
                    _lambda_sci = _lambda_sci.split("\\times ")
                    _lambda_sci = _lambda_sci[0] if len(_lambda_sci) == 1 else _lambda_sci[1]
                    axes.annotate(
                        f"$\\lambda = {_lambda_sci}$",
                        xy=(x, y),
                        xytext=(0.25, 0.25),
                        textcoords="offset fontsize",
                        color="k",
                        zorder=2,
                    )

        # plot L-curve corner if already optimize method excuted
        if self.lambda_opt is not None and plot_lambda_opt is True:
            _lambda_opt_sci = parse_scientific_notation(f"{self.lambda_opt:.2e}")
            axes.scatter(
                self.residual_norm(self.lambda_opt),
                self.regularization_norm(self.lambda_opt),
                c="r",
                marker="x",
                zorder=2,
                label=f"$\\lambda = {_lambda_opt_sci}$",
            )
            axes.legend()

        # labels
        axes.set_xlabel("Residual norm")
        axes.set_ylabel("Regularization norm")

        # set axis properties
        axes.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        return (fig, axes)

    def plot_curvature(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        show_max_curvature_line: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plotting the curvature of L-curve as function of regularization parameter.

        The curvature of L-curve is calculated by :meth:`.curvature` method.

        Parameters
        ----------
        fig : :obj:`~matplotlib.figure.Figure`, optional
            A matplotlib figure object, by default None.
        axes : :obj:`~matplotlib.axes.Axes`, optional
            A matplotlib Axes object, by default None.
        bounds : Collection[float | None], optional
            Boundary pair of log10 of regularization parameters, by default :obj:`.bounds`.
            If you set the bounds like ``(-10, None)``, the higher bound is set to
            :math:`\\log_{10}\\sigma_1^2`.
            Raise an error if a >= b in (a, b).
        n_beta : int, optional
            Number of regularization parameters, by default 500.
        show_max_curvature_line : bool, optional
            Whether or not to plot the vertical red dashed line at the maximum curvature point,
            by default True.

        Returns
        -------
        fig : :obj:`~matplotlib.figure.Figure`
            A matplotlib figure object.
        axes : :obj:`~matplotlib.axes.Axes`
            A matplotlib Axes object.
        """
        # get bounds of log10 of regularization parameter
        bounds = self._generate_bounds(bounds)

        # define regularization parameters
        lambdas = np.logspace(*bounds, n_beta)

        # compute the curvature
        curvatures = np.array([self.curvature(beta) for beta in lambdas])

        # validation
        if not isinstance(fig, Figure):
            fig = plt.figure()
        if not isinstance(axes, Axes):
            axes = fig.add_subplot()

        # plot the curvature
        axes.semilogx(lambdas, curvatures, color="C0", zorder=0)

        # indicate the maximum curvature point as a vertical red dashed line
        if self.lambda_opt is not None and show_max_curvature_line is True:
            axes.axvline(self.lambda_opt, color="r", linestyle="dashed", linewidth=1, zorder=1)

        # draw a y=0 dashed line
        axes.axhline(0, color="k", linestyle="dashed", linewidth=1, zorder=-1)

        # x range limitation
        axes.set_xlim(lambdas.min(), lambdas.max())

        # labels
        axes.set_xlabel("Regularization parameter $\\lambda$")
        axes.set_ylabel("Curvature of L-curve")

        # set axis properties
        axes.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        return (fig, axes)

    def curvature(self, beta: float) -> float:
        """Calculate L-curve curvature.

        This method calculates the curvature of L-curve at the point
        :math:`(\\sqrt{\\rho}, \\sqrt{\\eta})` as function of regularization parameter
        :math:`\\lambda`.

        If the curvature is positive, the L-curve is concave at the point.
        If the curvature is negative, the L-curve is convex at the point.

        Parameters
        ----------
        beta : float
            Regularization parameter :math:`\\lambda`.

        Returns
        -------
        float
            Value of calculated curvature.

        Examples
        --------
        >>> lcurve = Lcurve(s, U, basis, data=data)
        >>> curvature = lcurve.curvature(1.0e-5)
        """
        rho = self.rho(beta)
        eta = self.eta(beta)
        eta_dif = self.eta_diff(beta)

        numerator = -2.0 * rho * eta * (eta * beta**2.0 + beta * rho + rho * eta / eta_dif)
        denominator = ((beta * eta) ** 2.0 + rho**2.0) ** 1.5

        return numerator / denominator

    def _objective_function(self, logbeta: float) -> float:
        """Objective function for optimization.

        The optimal regularization parameter corresponds to the maximum curvature of L-curve.
        To apply the minimization solver, this method returns the negative value of curvature.

        Parameters
        ----------
        logbeta : float
            A log10 of regularization parameter.

        Returns
        -------
        float
            Negative value of curvature.
        """
        return -self.curvature(10**logbeta)
