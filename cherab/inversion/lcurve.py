"""Module for L-curve crietrion."""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import _SVDBase

__all__ = ["Lcurve"]


class Lcurve(_SVDBase):
    """L-curve criterion for regularization parameter optimization.

    The L-curve is a log-log plot of the residual norm versus the regularization norm.
    The L-curve criterion for tikhnov regularization gives the optimal regularization
    parameter :math:`\\lambda` as the corner point of the L-curve by maximizing the
    curvature of the L-curve.

    .. note::
        The theory and implementation of L-curve criterion is described in here_.

    .. _here: ../user/theory/lcurve.ipynb

    Parameters
    ----------
    s : vector_like
        singular values like :math:`\\mathbf{s} = (\\sigma_1, \\sigma_2, ...) \\in \\mathbb{R}^r`
    u : array_like
        left singular vectors like :math:`\\mathbf{U}\\in\\mathbb{R}^{M\\times r}`
    basis : array_like
        inverted solution basis like :math:`\\tilde{\\mathbf{V}} \\in \\mathbb{R}^{N\\times r}`.
    **kwargs : :py:class:`._SVDBase` properties, optional
        *kwargs* are used to specify properties like a `data`

    Examples
    --------
    >>> lcurve = Lcurve(s, u, basis, data=data)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_L_curve(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        bounds: tuple[float, float] = (-20.0, 2.0),
        n_beta: int = 100,
        scatter_plot: int | None = None,
        scatter_annotate: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plotting the L-curve in log-log scale.

        The points :math:`(\\sqrt{\\rho}, \\sqrt{\\eta})` are plotted in log-log scale.

        Parameters
        ----------
        fig
            matplotlib figure object, by default None.
        axes
            matplotlib Axes object, by default None.
        bounds
            bounds of log10 of regularization parameter, by default (-20.0, 2.0).
        n_beta
            number of regularization parameters, by default 100.
        scatter_plot
            whether or not to plot some L-curve points, by default None.
            If you want to manually define the number of points,
            put in the numbers. e.g.) ``scatter_plot=10``.
        scatter_annotate
            whether or not to annotate the scatter_points, by default True.
            This key argument is valid if only ``scatter_plot`` is not None.

        Returns
        -------
        tuple of :obj:`~matplotlib.figure.Figure` and :obj:`~matplotlib.axes.Axes`
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
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

        # plot some points of L curve and annotate with regularization parameters label
        if isinstance(scatter_plot, int) and scatter_plot > 0:
            betas = np.logspace(*bounds, scatter_plot)
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
                    axes.annotate(
                        "$\\lambda$ = {:.4g}".format(beta),
                        xy=(x, y),
                        xytext=(0.25, 0.25),
                        textcoords="offset fontsize",
                        color="k",
                        zorder=2,
                    )

        # plot L curve corner if already optimize method excuted
        if self.lambda_opt is not None:
            axes.scatter(
                self.residual_norm(self.lambda_opt),
                self.regularization_norm(self.lambda_opt),
                c="r",
                marker="x",
                zorder=2,
                label=f"$\\lambda = {self.lambda_opt:.2e}$",
            )
            axes.legend()

        # labels
        axes.set_xlabel("Residual norm")
        axes.set_ylabel("Regularization norm")

        return (fig, axes)

    def plot_curvature(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        bounds: tuple[float, float] = (-20.0, 2.0),
        n_beta: int = 100,
        show_max_curvature_line: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plotting the curvature of L-curve as function of regularization parameter.

        The curvature of L-curve is calculated by :meth:`.curvature` method.

        Parameters
        ----------
        fig
            matplotlib figure object, by default None.
        axes
            matplotlib Axes object, by default None.
        bounds
            bounds of log10 of regularization parameter, by default (-20.0, 2.0).
        n_beta
            number of regularization parameters, by default 100.
        show_max_curvature_line
            whether or not to plot the vertical red dashed line at the maximum curvature point,
            by default True.

        Returns
        -------
        tuple[:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`]
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
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
        beta
            regularization parameter :math:`\\lambda`

        Returns
        -------
        float
            the value of calculated curvature

        Examples
        --------
        >>> lcurve = Lcurve(s, u, basis, data=data)
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
        logbeta
            log10 of regularization parameter

        Returns
        -------
        float
            negative value of curvature
        """
        return -self.curvature(10**logbeta)
