"""Abstract base class for regularization parameter selection criteria."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._base import _SVDBase


class Criterion(ABC):
    r"""Abstract base class for regularization parameter selection criteria.

    Concrete subclasses implement :meth:`optimize` to find the optimal regularization parameter
    :math:`\lambda` for a given `~._SVDBase` solver.

    Subclasses that support discrete solvers (e.g. `~.TSVD`) may also implement :meth:`optimize_discrete`.
    """

    def __init__(self) -> None:
        self._solver: _SVDBase | None = None

    @abstractmethod
    def optimize(
        self,
        solver: _SVDBase,
        bounds: tuple[float, float],
        stepsize: float = 10,
        **kwargs: Any,
    ) -> tuple[float, Any]:
        """Find the optimal regularization parameter.

        Parameters
        ----------
        solver
            An `~._SVDBase` instance providing ``filter``, ``rho``, etc.
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
        result
            Optimizer result object (implementation-dependent).
        """

    def _resolve_solver(self, solver: _SVDBase | None) -> _SVDBase:
        if solver is not None:
            return solver
        if self._solver is None:
            raise RuntimeError("No solver available. Call optimize() or pass solver.")
        return self._solver
