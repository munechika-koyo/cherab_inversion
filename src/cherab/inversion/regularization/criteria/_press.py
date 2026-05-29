"""Predicted Residual Error Sum of Squares (PRESS) criterion."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, overload

import matplotlib.axes
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeResult, basinhopping

from ._base import Criterion

if TYPE_CHECKING:
    from .._base import _SVDBase

__all__ = ["PRESS"]


class PRESS(Criterion):
    r"""Predicted Residual Error Sum of Squares (PRESS) criterion.

    The PRESS criterion, also known as ordinary cross-validation, selects the
    regularization parameter :math:`\lambda` by minimizing the leave-one-out
    cross-validation error.

    .. note::
        The theory and implementation of the PRESS criterion are described in
        the :doc:`PRESS theory page </theory/press>`.
    """

    def __init__(self) -> None:
        self._solver: _SVDBase | None = None

    def press(self, solver: _SVDBase, beta: float) -> float:
        r"""Calculate the PRESS criterion function.

        The PRESS function :math:`\mathcal{P}(\lambda)` can be expressed with SVD
        components as:

        .. math::

            \mathcal{P}(\lambda)
            =
            \left\|
            \left[
                \mathrm{Diag}\!\left(
                    \mathbf{I} - \mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T}
                \right)
            \right]^{-1}
            \!\left(
                \mathbf{I} - \mathbf{U}\mathbf{F}_\lambda\mathbf{U}^\mathsf{T}
            \right)\hat{\mathbf{b}}
            \right\|_2^2,

        where :math:`\hat{\mathbf{b}} = \mathbf{B}\mathbf{b}` (or simply
        :math:`\mathbf{b}` when :math:`\mathbf{B} = \mathbf{I}`) and
        :math:`\mathrm{Diag}(\mathbf{M})` denotes the diagonal matrix formed
        from the diagonal entries of :math:`\mathbf{M}`.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance providing ``U``, ``filter``, ``data``, etc.
        beta
            Regularization parameter :math:`\lambda`.

        Returns
        -------
        float
            Value of PRESS function at a given regularization parameter.

        Raises
        ------
        RuntimeError
            If ``solver.data`` is not set.
        """
        if solver.data is None:
            raise RuntimeError(
                "solver.data is not set. Assign the data vector before calling press()."
            )
        assert solver._ub is not None

        F = solver.filter(beta)  # (r,)
        U = solver.U  # (M, r)

        # b_hat = B @ b (or just b when B is None)
        # type: ignore needed because scipy sparray stubs lack __matmul__
        if solver.B is not None:
            b_hat: np.ndarray = np.asarray(solver.B @ solver.data)  # type: ignore[operator]
        else:
            b_hat = solver.data  # (M,)

        # ub = U^T @ b_hat — already cached as solver._ub, reuse it
        ub: np.ndarray = solver._ub  # (r,)

        # numerator: (I - U diag(F) U^T) b_hat = b_hat - U @ (F * ub)
        num = b_hat - U @ (F * ub)  # (M,)

        # denominator: diagonal of (I - U diag(F) U^T)
        denom = 1.0 - (U**2) @ F  # (M,)

        return float(np.sum((num / denom) ** 2))

    def optimize(
        self,
        solver: _SVDBase,
        bounds: tuple[float, float],
        stepsize: float = 10,
        **kwargs: Any,
    ) -> tuple[float, OptimizeResult]:
        """Find the optimal parameter by minimizing PRESS with basinhopping.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance.
        bounds
            ``(log10_lower, log10_upper)`` search interval.
        stepsize
            Step-size for the underlying global optimizer.
        **kwargs
            Extra arguments forwarded to the optimizer.

        Returns
        -------
        lambda_opt : float
            Optimal regularization parameter.
        result : `~scipy.optimize.OptimizeResult`
            Optimizer result object returned by :obj:`~scipy.optimize.basinhopping`.
        """
        self._solver = solver
        init_logbeta = 0.5 * (bounds[0] + bounds[1])
        res: OptimizeResult = basinhopping(
            lambda logb: self.press(solver, 10 ** logb[0]),
            x0=[init_logbeta],
            minimizer_kwargs={"bounds": [bounds]},  # type: ignore[arg-type]
            stepsize=stepsize,
            **kwargs,
        )
        return float(10 ** res.x[0]), res

    @overload
    def plot_press(
        self,
        solver: _SVDBase | None = None,
        axes: None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        show_min_line: bool = True,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure]: ...

    @overload
    def plot_press(
        self,
        solver: _SVDBase | None = None,
        axes: matplotlib.axes.Axes | None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        show_min_line: bool = True,
    ) -> matplotlib.axes.Axes: ...

    def plot_press(
        self,
        solver: _SVDBase | None = None,
        axes: matplotlib.axes.Axes | None = None,
        bounds: Collection[float | None] | None = None,
        n_beta: int = 500,
        show_min_line: bool = True,
    ) -> tuple[matplotlib.axes.Axes, matplotlib.figure.Figure] | matplotlib.axes.Axes:
        r"""Plot PRESS as a function of the regularization parameter.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance. If ``None``, the last solver used in
            :meth:`optimize` is used.
        axes
            Matplotlib axes to plot on. If ``None``, a new figure and axes are created.
        bounds
            Boundary pair of log10 of regularization parameters, by default None.
            If None, the bounds are generated by the solver as described in
            ``SVD._generate_bounds()``.
            If you set the bounds like ``(-10, None)``, the higher bound is set to
            :math:`\log_{10}\sigma_1^2`.
            Raise an error if a >= b in (a, b).
        n_beta
            Number of regularization parameters, by default 500.
        show_min_line
            Whether to show a vertical line at the optimal regularization parameter.

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
        press_vals = np.array([self.press(solver, beta) for beta in lambdas])

        axes.loglog(lambdas, press_vals, color="C0", zorder=0)

        if solver.lambda_opt is not None and show_min_line:
            axes.axvline(solver.lambda_opt, color="r", linestyle="dashed", linewidth=1, zorder=1)

        axes.set_xlim(lambdas.min(), lambdas.max())
        axes.set_xlabel(r"Regularization parameter, $\lambda$")
        axes.set_ylabel("PRESS function")
        axes.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        return (axes, fig) if fig is not None else axes
