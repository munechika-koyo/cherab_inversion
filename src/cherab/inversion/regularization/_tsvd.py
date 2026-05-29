from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from ._svd import SVD

if TYPE_CHECKING:
    from .criteria._base import Criterion


class TSVD(SVD):
    r"""Truncated SVD solver.

    Parameters
    ----------
    *args, **kwargs
        Same as `~.SVD`.

    Notes
    -----
    For TSVD, an L-curve criterion can be applied in a discrete manner by evaluating candidate
    truncation points :math:`\beta_k = \sigma_k^2`.
    If the criterion object provides ``optimize_discrete`` (as `~.Lcurve` does),
    :meth:`solve` dispatches to that method automatically.
    """

    def filter(self, beta: float) -> NDArray[np.float64]:
        r"""Calculate the filter factors :math:`f_{\lambda, i}`.

        The filter factors are diagonal elements of the filter matrix :math:`\mathbf{F}_\lambda`,
        and can be expressed with TSVD components as follows:

        .. math::

            f_{\lambda, i}
            =
            \begin{cases}
                1 & \mathrm{if}\; \sigma_i^2 \geq \beta, \\
                0 & \mathrm{if}\; \sigma_i^2 < \beta.
            \end{cases}

        Parameters
        ----------
        beta
            Regularization parameter.

        Returns
        -------
        (r, ) ndarray
            1-D array containing filter factors, the length of which is the same as the number of
            singular values.
        """
        factors = np.ones_like(self.s)
        factors[self.s**2 < beta] = 0
        return factors

    def solve(
        self,
        criterion: Criterion | None = None,
        bounds=None,
        stepsize: float = 10,
        **kwargs: Any,
    ) -> tuple[ndarray, Any]:
        """Solve with TSVD regularization.

        Parameters
        ----------
        criterion
            Criterion object. If it has ``optimize_discrete``, that method is used.
        bounds
            Bounds for continuous criteria.
        stepsize
            Step-size for continuous criteria.
        **kwargs
            Extra arguments forwarded to criterion optimizer.

        Returns
        -------
        sol : (N,) ndarray
            Optimal solution vector.
        result : :obj:`~scipy.optimize.OptimizeResult` | dict
            Object returned by :obj:`~scipy.optimize.basinhopping` function for continuous criteria,
            or a dictionary of relevant information for discrete criteria.
        """
        if criterion is None:
            # Keep all components.
            self._lambda_opt = float(self.s[-1] ** 2) * (1.0 - 1.0e-10)
            return self.solution(self._lambda_opt), {}

        if hasattr(criterion, "optimize_discrete"):
            lambda_opt, result = criterion.optimize_discrete(self)
            self._lambda_opt = lambda_opt
            return self.solution(lambda_opt), result

        return super().solve(criterion, bounds=bounds, stepsize=stepsize, **kwargs)
