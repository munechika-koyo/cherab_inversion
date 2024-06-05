"""Module for GCV crieterion inversion."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import _SVDBase

__all__ = ["GCV"]


class GCV(_SVDBase):
    """Generalized Cross-Validation (GCV) criterion for regularization parameter optimization.

    .. note::
        The theory and implementation of GCV criterion can be seen `here`_.

    .. _here: ../user/theory/gcv.ipynb

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
    >>> gcv = GCV(s, u, basis, data=data)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def gcv(self, beta: float) -> float:
        """Calculate of GCV criterion function.

        GCV function :math:`\\mathcal{G}(\\lambda)` can be expressed with SVD components as:

        .. math::

            \\mathcal{G}(\\lambda) = \\frac{\\rho}{\\left[1 - \\sum_{i=1}^r f_{\\lambda, i}\\right]^2}.

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        float
            the value of GCV function at a given regularization parameter
        """
        return self.rho(beta) / (self.s.size - np.sum(self.filter(beta))) ** 2.0

    def _objective_function(self, logbeta: float) -> float:
        """Objective function for optimization.

        The optimal regularization parameter corresponds to the minimum value of GCV function.

        Parameters
        ----------
        logbeta
            log10 of regularization parameter

        Returns
        -------
        float
            the value of GCV function at a given regularization parameter
        """
        return self.gcv(10**logbeta)

    def plot_gcv(
        self,
        fig: Figure | None = None,
        axes: Axes | None = None,
        bounds: tuple[float | None, float | None] | None = None,
        n_beta: int = 500,
        show_min_line: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plotting GCV as a function of the regularization parameter in log-log scale.

        Parameters
        ----------
        fig
            matplotlib figure object, by default None.
        axes
            matplotlib Axes object, by default None.
        bounds
            bounds of log10 of regularization parameter, by default
            :obj:`~cherab.inversion.core._SVDBase.bounds`.
            If you set the bounds like ``(-10, None)``, the higher bound is set to
            :math:`\\log_{10}\\sigma_1^2`.
            Raise an error if a >= b in (a, b).
        n_beta
            number of regularization parameters, by default 500.
        show_min_line
            whether or not to plot the vertical red dashed line at the minimum GCV point,
            by default True.

        Returns
        -------
        tuple[:obj:`~matplotlib.figure.Figure`, :obj:`~matplotlib.axes.Axes`]
            (fig, axes), each of which is matplotlib objects applied some properties.
        """
        # get bounds of log10 of regularization parameter
        bounds = self._generate_bounds(bounds)

        # define regularization parameters
        lambdas = np.logspace(*bounds, n_beta)

        # calculate GCV values
        gcvs = np.array([self.gcv(beta) for beta in lambdas])

        # validation
        if not isinstance(fig, Figure):
            fig = plt.figure()
        if not isinstance(axes, Axes):
            axes = fig.add_subplot()

        # plot
        axes.loglog(lambdas, gcvs, color="C0", zorder=0)

        # indicate the minimum gcv curve point as a vertical red dashed line
        if self.lambda_opt is not None and show_min_line is True:
            axes.axvline(self.lambda_opt, color="r", linestyle="dashed", linewidth=1, zorder=1)

        # x range limitation
        axes.set_xlim(lambdas.min(), lambdas.max())

        # labels
        axes.set_xlabel("Regularization parameter $\\lambda$")
        axes.set_ylabel("GCV function")

        # set axis properties
        axes.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        return (fig, axes)
