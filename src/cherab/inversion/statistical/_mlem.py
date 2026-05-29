from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from rich.progress import Progress, TimeRemainingColumn
from scipy.sparse import csc_array, csr_array


class MLEM:
    r"""Maximum Likelihood Expectation Maximization (MLEM) algorithm.

    This class provides a simple implementation of the MLEM algorithm for solving the inverse problem
    :math:`\mathbf{T} \mathbf{x} = \mathbf{b}` where :math:`\mathbf{T}` is the forward problem matrix,
    :math:`\mathbf{x}` is the unknown solution, and :math:`\mathbf{b}` is the given data.

    .. note::
        The detailed mathematical formulation and implementation of the MLEM algorithm can be found
        in :doc:`here </theory/mlem>`.

    Parameters
    ----------
    T
        `(M, N)` Matrix :math:`\mathbf{T}\in\mathbb{R}^{M \times N}` of the forward problem.
    data
        Given data :math:`\mathbf{b}\in\mathbb{R}^M` or :math:`\mathbf{b}\in\mathbb{R}^{M \times K}`,
        where `K` is typically the number of time slices, by default None.
    """

    def __init__(
        self,
        T: NDArray[np.float64] | csr_array[np.float64, tuple[int, int]],
        data: ArrayLike | None = None,
    ) -> None:
        # validate arguments
        if not hasattr(T, "ndim"):
            raise TypeError("T must be an ndarray object")
        if T.ndim != 2:
            raise ValueError("T must be a 2D array")

        # set matrix attributes
        self._T = T
        self._data: NDArray[np.float64] | None = None

        # set data attribute
        if data is not None:
            self.data = data

    @property
    def T(self) -> NDArray[np.float64] | csr_array[np.float64, tuple[int, int]]:
        r"""Matrix :math:`\mathbf{T}\in\mathbb{R}^{M \times N}` of the forward problem."""
        return self._T

    @property
    def data(self) -> NDArray[np.float64] | None:
        r"""Data vector :math:`\mathbf{b}\in\mathbb{R}^M` or matrix :math:`\mathbf{b}\in\mathbb{R}^{M \times K}`."""
        return self._data

    @data.setter
    def data(self, value: ArrayLike) -> None:
        data = np.asarray(value, dtype=float)
        if data.ndim == 1:
            data = data.transpose()
            size = data.size
        elif data.ndim == 2:
            size = data.shape[0]
        else:
            raise ValueError("data must be a vector or a matrix")
        if size != self._T.shape[0]:
            raise ValueError("data size must be the same as the number of rows of geometry matrix")
        self._data = data

    def solve(
        self,
        x0: NDArray[np.float64] | None = None,
        tol: float = 1e-5,
        max_iter: int = 100,
        quiet: bool = False,
        store_temp: bool = False,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        r"""Solve the inverse problem using the MLEM algorithm.

        Parameters
        ----------
        x0
            Initial guess of the solution :math:`\mathbf{x}\in\mathbb{R}^N` or
            :math:`\mathbf{x}\in\mathbb{R}^{N \times K}`.
            If not given, a vector of ones is used.
        tol
            Tolerance for convergence, by default 1e-5.
            The iteration stops when the maximum difference between the current and previous
            solutions is less than this value.
        max_iter
            Maximum number of iterations, by default 100.
        quiet
            Whether to suppress the progress bar, by default False.
        store_temp
            Whether to store the temporary solutions during the iteration, by default False.

        Returns
        -------
        x : (N, ) or (N, K) `numpy.ndarray`
            Solution of the inverse problem.
            If the data has K time slices, the solution is a matrix.
        status : `dict`
            Dictionary containing the status of the iteration.

        Raises
        ------
        ValueError
            If data is not set before calling this method.
        """
        if self._data is None:
            raise ValueError("data must be set before calling solve method")

        # set initial guess
        if x0 is None:
            if self._data.ndim == 2:
                x0 = np.ones((self._T.shape[1], self._data.shape[1]))
            else:
                x0 = np.ones(self._T.shape[1])
        elif isinstance(x0, np.ndarray):
            if x0.ndim == 1:
                size = x0.size
            elif x0.ndim == 2:
                size = x0.shape[0]
            else:
                raise ValueError("x0 must be a vector or a matrix.")
            if size != self._T.shape[1]:
                raise ValueError("x0 must have the same size as the rows of T")

        assert x0 is not None

        # set tolerance
        def _tolerance(x):
            return tol * np.amax(x)

        # set progress bar
        console = Console(quiet=quiet)
        progress = Progress(
            *Progress.get_default_columns()[:3],
            TimeRemainingColumn(elapsed_when_finished=True),
            auto_refresh=False,
            console=console,
        )
        task_id = progress.add_task("Pre-processing...", total=max_iter)
        progress.refresh()

        # set iteration counter and status
        niter = 0
        status: dict[str, Any] = {}
        self._converged = False
        diffs: list[float] = []
        x_temp: list[NDArray[np.float64]] = []  # temporary solutions
        T_t = self._T.T  # transpose of T
        if isinstance(T_t, csc_array):
            T_t = T_t.tocsr()  # convert to csr for faster row access
        T_t1_recip: NDArray[np.float64] = 1.0 / T_t.sum(axis=1)  # 1 / (T^T @ 1)

        # start iteration
        with progress:
            # update progress bar
            progress.update(task_id, description="Solving...")
            progress.refresh()
            while niter < max_iter and not self._converged:
                data: NDArray[np.float64] = self._T @ x0  # type: ignore[bad-assignment]
                ratio = self._data / data
                x: NDArray[np.float64] = x0 * (T_t @ ratio * T_t1_recip)  # type: ignore[bad-assignment]

                # store temporary solution
                x_temp.append(x) if store_temp else None

                # check convergence
                diff_max: float = np.amax(np.abs(x - x0))
                diffs.append(diff_max)
                _tol = _tolerance(x0)
                self._converged = bool(diff_max < _tol)

                # update solution
                x0 = x
                text = f"(Max Diff: {diff_max:.2e}, Tol: {_tol:.2e})"

                niter += 1
                progress.update(task_id, description=f"Solving...{text}", advance=1)
                progress.refresh()

            else:
                # stop progress bar
                progress.update(task_id, description=f"Completed {text}", completed=max_iter)
                progress.refresh()

        # set status
        status["elapsed_time"] = progress.tasks[task_id].elapsed
        status["niter"] = niter
        status["tol"] = _tol
        status["converged"] = self._converged
        status["diffs"] = np.asarray(diffs)
        status["x_temp"] = x_temp

        return x, status
