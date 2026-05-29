"""L-curve criterion for regularization parameter selection."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, overload

import matplotlib.axes
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, basinhopping

from ...tools import parse_scientific_notation
from ._base import Criterion

if TYPE_CHECKING:
    from .._base import _SVDBase

__all__ = ["Lcurve"]


class Lcurve(Criterion):
    r"""L-curve criterion.

    The L-curve is a log-log plot of the residual norm versus the regularization norm.
    The L-curve criterion for Tikhonov regularization gives the optimal regularization
    parameter :math:`\lambda` as the corner point of the L-curve by maximizing the
    curvature of the L-curve.

    .. note::
        The theory and implementation of the L-curve criterion are described
        :doc:`here </theory/lcurve>`.
    """

    def __init__(self) -> None:
        self._solver: _SVDBase | None = None

    def curvature(self, solver: _SVDBase, beta: float) -> float:
        r"""Calculate L-curve curvature.

        This method calculates the curvature of L-curve at the point
        :math:`(\sqrt{\rho}, \sqrt{\eta})` as function of regularization parameter
        :math:`\lambda`.

        If the curvature is positive, the L-curve is concave at the point.
        If the curvature is negative, the L-curve is convex at the point.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance providing ``rho``, ``eta``, and ``eta_diff``.
        beta
            Regularization parameter :math:`\lambda`.

        Returns
        -------
        float
            Value of calculated curvature.

        Examples
        --------
        >>> svd = SVD(s, U, basis, data=data)
        >>> lcurve = Lcurve()
        >>> curvature = lcurve.curvature(svd, 1.0e-5)
        """
        rho = solver.rho(beta)
        eta = solver.eta(beta)
        eta_dif = solver.eta_diff(beta)

        # TSVD can produce eta'==0 on plateau regions; skip such points.
        if eta_dif == 0:
            return -np.inf

        numerator = -2.0 * rho * eta * (eta * beta**2.0 + beta * rho + rho * eta / eta_dif)
        denominator = ((beta * eta) ** 2.0 + rho**2.0) ** 1.5
        return numerator / denominator

    def optimize(
        self,
        solver: _SVDBase,
        bounds: tuple[float, float],
        stepsize: float = 10,
        **kwargs: Any,
    ) -> tuple[float, OptimizeResult]:
        """Find optimal parameter by maximizing curvature (continuous).

        This method finds the optimal regularization parameter by maximizing the curvature
        of the L-curve using the basinhopping global optimization algorithm.  The search is
        performed in log-space, so the bounds are specified as ``(log10_lower, log10_upper)``.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance providing ``rho``, ``eta``, and ``eta_diff``.
        bounds
            ``(log10_lower, log10_upper)`` search interval for the regularization parameter in log-space.
        stepsize
            Step-size for the underlying global optimizer.
        **kwargs
            Extra arguments forwarded to the optimizer.

        Returns
        -------
        lambda_opt : float
            Optimal regularization parameter that maximizes the curvature of the L-curve.
        result : `~scipy.optimize.OptimizeResult`
            Optimizer result object returned by :obj:`~scipy.optimize.basinhopping`.
        """
        self._solver = solver
        init_logbeta = 0.5 * (bounds[0] + bounds[1])
        res: OptimizeResult = basinhopping(
            lambda logb: -self.curvature(solver, 10 ** logb[0]),
            x0=[init_logbeta],
            minimizer_kwargs={"bounds": [bounds]},  # type: ignore[arg-type]
            stepsize=stepsize,
            **kwargs,
        )
        return float(10 ** res.x[0]), res

    def optimize_discrete(
        self, solver: _SVDBase
    ) -> tuple[float, dict[str, int | NDArray[np.float64]]]:
        r"""Find optimal TSVD truncation via the discrete Menger curvature of the L-curve.

        The TSVD filter factors are piecewise-constant (0 or 1), so the analytical derivative
        ``eta_diff`` is identically zero.
        Instead the **Menger curvature** — the reciprocal of the circumradius of three consecutive
        points — is used.
        For three points :math:`P_{k-1}, P_k, P_{k+1}` in log–log space
        (:math:`\xi = \log\|\mathbf{r}\|`, :math:`\chi = \log\|\mathbf{x}\|`):

        .. math::

            \kappa_k =
            \frac{2\,A(P_{k-1}, P_k, P_{k+1})}
                 {|P_{k-1}P_k|\,|P_k P_{k+1}|\,|P_{k-1}P_{k+1}|}

        where :math:`A` is the signed area of the triangle (positive when the turn
        :math:`P_{k-1}\to P_k\to P_{k+1}` is counter-clockwise).
        The endpoint indices (:math:`k=1` and :math:`k=r`) are assigned :math:`\kappa = 0`.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance providing ``rho`` and ``eta``.

        Returns
        -------
        lambda_opt : float
            Optimal regularization parameter corresponding to the index of maximum curvature.
        result : dict[str, int | numpy.ndarray]
            Dictionary containing the following keys:
            - ``k``: Index of the optimal truncation point (1-based).
            - ``curvatures``: Array of curvature values for all candidate truncation points.

        Notes
        -----
        This approach avoids computing second-order finite differences, which
        can amplify noise.  It matches the method of
        :cite:t:`castellanos2002triangular`.
        """
        self._solver = solver
        betas = solver.s**2  # descending: σ₁² ≥ σ₂² ≥ … ≥ σ_r²
        r = len(betas)

        # Log-squared norms at every candidate truncation point
        rho_vals = np.array([solver.rho(float(b)) for b in betas])
        eta_vals = np.array([solver.eta(float(b)) for b in betas])

        # Only points where both norms are strictly positive are valid
        valid = (rho_vals > 0) & (eta_vals > 0)
        curvatures = np.full(r, np.nan)

        if valid.sum() >= 3:
            xi = np.log(rho_vals[valid])  # log residual norm²
            chi = np.log(eta_vals[valid])  # log solution norm²
            n = len(xi)

            # Vectorised Menger curvature for interior indices 1 .. n-2
            x1, y1 = xi[:-2], chi[:-2]
            x2, y2 = xi[1:-1], chi[1:-1]
            x3, y3 = xi[2:], chi[2:]

            d12 = np.hypot(x2 - x1, y2 - y1)
            d23 = np.hypot(x3 - x2, y3 - y2)
            d13 = np.hypot(x3 - x1, y3 - y1)

            # Signed 2× area: positive when P1→P2→P3 is CCW
            area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

            denom = d12 * d23 * d13
            curv_interior = np.where(denom > 0, area2 / denom, 0.0)

            curv = np.zeros(n)
            curv[1:-1] = curv_interior
            curvatures[valid] = curv

        k_opt = int(np.nanargmax(curvatures))
        return float(betas[k_opt]), {"k": k_opt + 1, "curvatures": curvatures}

    @overload
    def plot_L_curve(
        self,
        solver: _SVDBase | None = None,
        axes: None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        scatter_plot: int | None = None,
        scatter_annotate: bool = True,
        plot_lambda_opt: bool = True,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure]: ...

    @overload
    def plot_L_curve(
        self,
        solver: _SVDBase | None = None,
        axes: matplotlib.axes.Axes | None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        scatter_plot: int | None = None,
        scatter_annotate: bool = True,
        plot_lambda_opt: bool = True,
    ) -> matplotlib.axes.Axes: ...

    def plot_L_curve(
        self,
        solver: _SVDBase | None = None,
        axes: matplotlib.axes.Axes | None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        scatter_plot: int | None = None,
        scatter_annotate: bool = True,
        plot_lambda_opt: bool = True,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure] | matplotlib.axes.Axes:
        r"""Plot the curvature of L-curve as function of regularization parameter.

        The curvature of L-curve is calculated by :meth:`.curvature` method.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance.
        axes
            A matplotlib Axes object, by default None.
        bounds
            Boundary pair of log10 of regularization parameters, by default None.
            If None, the bounds are generated by the solver as described in ``SVD._generate_bounds()``.
            If you set the bounds like ``(-10, None)``, the higher bound is set to
            :math:`\log_{10}\sigma_1^2`.
            Raise an error if a >= b in (a, b).
        n_beta
            Number of regularization parameters, by default 500.
        scatter_plot
            Whether or not to plot some L-curve points as a 10 :sup:`x` format, by default None.
            If you want to manually define the number of points,
            enter the numbers like ``scatter_plot=10`` then around 10 points corresponding to
            10 :sup:`x` format are plotted.
        scatter_annotate
            Whether or not to annotate the scatter_points, by default True.
            This key argument is valid if only `.scatter_plot` is not None.
        plot_lambda_opt
            Whether or not to plot the L-curve corner point, by default True.

        Returns
        -------
        axes : :obj:`~matplotlib.axes.Axes`
            A matplotlib Axes object.
        fig : :obj:`~matplotlib.figure.Figure`
            A matplotlib figure object if axes is None, otherwise None.
        """
        solver = self._resolve_solver(solver)

        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = None

        bounds_resolved = solver._generate_bounds(bounds)
        lambdas = np.logspace(*bounds_resolved, n_beta)

        residual_norms = np.array([solver.residual_norm(i) for i in lambdas])
        regularization_norms = np.array([solver.regularization_norm(i) for i in lambdas])

        axes.loglog(residual_norms, regularization_norms, color="C0", zorder=0)

        if isinstance(scatter_plot, int) and scatter_plot > 0:
            a, b = np.ceil(bounds_resolved[0]), np.floor(bounds_resolved[1])
            interval = max(1, round((b - a) / scatter_plot))
            betas = 10 ** np.arange(a, b, interval)
            for beta in betas:
                x, y = solver.residual_norm(beta), solver.regularization_norm(beta)
                axes.scatter(
                    x,
                    y,
                    fc="C0",
                    marker=".",
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

        # plot L-curve corner if already optimize method executed
        if (lambda_opt := solver.lambda_opt) is not None and plot_lambda_opt:
            _lambda_opt_sci = parse_scientific_notation(f"{lambda_opt:.2e}")
            axes.scatter(
                solver.residual_norm(lambda_opt),
                solver.regularization_norm(lambda_opt),
                c="r",
                marker="x",
                zorder=2,
                label=f"$\\lambda = {_lambda_opt_sci}$",
            )
            axes.legend()

        axes.set_xlabel("Residual norm")
        axes.set_ylabel("Regularization norm")
        axes.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        return (axes, fig) if fig is not None else axes

    @overload
    def plot_curvature(
        self,
        solver: _SVDBase | None = None,
        axes: None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        show_max_curvature_line: bool = True,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure]: ...

    @overload
    def plot_curvature(
        self,
        solver: _SVDBase | None = None,
        axes: matplotlib.axes.Axes | None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        show_max_curvature_line: bool = True,
    ) -> matplotlib.axes.Axes: ...

    def plot_curvature(
        self,
        solver: _SVDBase | None = None,
        axes: matplotlib.axes.Axes | None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        show_max_curvature_line: bool = True,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure] | matplotlib.axes.Axes:
        r"""Plot the curvature of L-curve as function of regularization parameter.

        The curvature of L-curve is calculated by :meth:`.curvature` method.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance. If ``None``, the last solver used in :meth:`optimize` is used.
        axes
            A matplotlib Axes object, by default None.
        bounds
            Boundary pair of log10 of regularization parameters, by default None.
            If None, the bounds are generated by the solver as described in ``SVD._generate_bounds()``.
            If you set the bounds like ``(-10, None)``, the higher bound is set to
            :math:`\log_{10}\sigma_1^2`.
            Raise an error if a >= b in (a, b).
        n_beta
            Number of regularization parameters, by default 500.
        show_max_curvature_line
            Whether or not to plot the vertical red dashed line at the maximum curvature point,
            by default True.

        Returns
        -------
        axes : :obj:`~matplotlib.axes.Axes`
            A matplotlib Axes object.
        fig : :obj:`~matplotlib.figure.Figure`
            A matplotlib figure object if axes is None, otherwise None.
        """
        solver = self._resolve_solver(solver)

        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = None

        bounds_resolved = solver._generate_bounds(bounds)
        lambdas = np.logspace(*bounds_resolved, n_beta)
        curvatures = np.array([self.curvature(solver, beta) for beta in lambdas])

        axes.semilogx(lambdas, curvatures, color="C0", zorder=0)

        if (lambda_opt := solver.lambda_opt) is not None and show_max_curvature_line:
            axes.axvline(lambda_opt, color="r", linestyle="dashed", linewidth=1, zorder=1)

        axes.axhline(0, color="k", linestyle="dashed", linewidth=1, zorder=-1)
        axes.set_xlim(lambdas.min(), lambdas.max())
        axes.set_xlabel("Regularization parameter, $\\lambda$")
        axes.set_ylabel("Curvature of L-curve")
        axes.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        return (axes, fig) if fig is not None else axes
