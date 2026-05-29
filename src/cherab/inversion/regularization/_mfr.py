from __future__ import annotations

import contextlib
import pickle
from collections.abc import Collection
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from scipy.sparse import csc_array, diags_array, issparse
from scipy.sparse._base import _spbase

from ._base import _SVDBase
from ._svd import SVD
from .criteria import Lcurve
from .criteria._base import Criterion
from .utility import compute_svd


class MFR:
    r"""Inverses provided data using Minimum Fisher Regularization (MFR) scheme.

    .. note::
        The theory and implementation of the MFR are described :doc:`here </theory/mfr>`.

    Parameters
    ----------
    T
        Matrix :math:`\mathbf{T}\in\mathbb{R}^{M\times N}` of the forward problem
        (geometry matrix, ray transfer matrix, etc.).
    dmats
        Iterable of pairs of derivative matrices :math:`\mathbf{D}_i` and :math:`\mathbf{D}_j`
        along to :math:`i` and :math:`j` coordinate directions, respectively.
    Q
        Weighted matrix for the residual norm :math:`\mathbf{Q}\in\mathbb{R}^{M\times M}`,
        by default None (meaning :math:`\mathbf{Q} = \mathbf{I}`).
        This matrix must be a symmetric positive semi-definite matrix.
    data
        Given data as a vector :math:`\mathbf{b}\in\mathbb{R}^M`, by default None.

    Examples
    --------
    >>> mfr = MFR(T, dmats, data=data)
    """

    def __init__(
        self,
        T: NDArray[np.float64] | _spbase[np.float64, tuple[int, int]],
        dmats: Collection[tuple[csc_array, csc_array]],
        *,
        Q: NDArray[np.float64] | _spbase[np.float64, tuple[int, int]] | None = None,
        data: NDArray[np.float64] | None = None,
    ) -> None:
        # validate arguments
        if not hasattr(T, "ndim"):
            raise TypeError("T must be an array-like object")
        if T.ndim != 2:
            raise ValueError("T must be a 2D array")

        if not isinstance(dmats, Collection):
            raise TypeError("dmats must be a collection of derivative matrices pair.")
        for dmat1, dmat2 in dmats:
            if not issparse(dmat1):
                raise TypeError("one of the matrices in dmats is not a scipy sparse array.")
            if not issparse(dmat2):
                raise TypeError("one of the matrices in dmats is not a scipy sparse array.")
            if dmat1.shape != dmat2.shape:
                raise ValueError("dmats must have the same shape")
            if dmat1.shape[0] != dmat1.shape[1] or dmat1.shape[0] != T.shape[1]:
                raise ValueError("dmats must be square matrices with the same size as columns of T")

        # set matrix attributes
        self._T = T
        self._dmats = dmats
        self._Q = Q

        # set data attribute
        if data is not None:
            self.data = data

    @property
    def T(self) -> NDArray[np.float64] | _spbase[np.float64, tuple[int, int]]:
        r"""Matrix :math:`\mathbf{T}` of the forward problem."""
        return self._T

    @property
    def dmats(self) -> Collection[tuple[csc_array, csc_array]]:
        r"""List of pairs of derivative matrices :math:`\mathbf{D}_i` and :math:`\mathbf{D}_j`.

        Each derivative matrix's subscript represents the coordinate direction.
        """
        return self._dmats

    @property
    def data(self) -> NDArray[np.float64]:
        r"""Given data as a vector :math:`\mathbf{b}`."""
        return self._data

    @property
    def Q(self) -> NDArray[np.float64] | _spbase[np.float64, tuple[int, int]] | None:
        r"""Weighted matrix :math:`\mathbf{Q}` for the residual norm."""
        return self._Q

    @data.setter
    def data(self, value: ArrayLike) -> None:
        data = np.asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._T.shape[0]:
            raise ValueError("data size must be the same as the number of rows of geometry matrix")
        self._data = data

    def solve(
        self,
        x0: NDArray[np.float64] | None = None,
        derivative_weights: Collection[float] | None = None,
        eps: float = 1.0e-6,
        tol: float = 1e-3,
        miter: int = 4,
        regularizer: type[_SVDBase] | None = None,
        criterion: Criterion | None = None,
        store_regularizers: bool = False,
        path: str | Path | None = None,
        use_gpu: bool = False,
        dtype: DTypeLike | None = None,
        verbose: bool = False,
        spinner: bool = True,
        **kwargs,
    ) -> tuple[NDArray[np.float64] | None, dict]:
        r"""Solve the inverse problem using MFR scheme.

        MFR is an iterative scheme that combines Singular Value Decomposition (SVD) and a
        optimizer to find the optimal regularization parameter.

        The detailed workflow of the MFR scheme is described in :doc:`here </theory/mfr>`.

        Parameters
        ----------
        x0
            Initial solution vector, by default ones vector.
        derivative_weights
            Allows to specify anisotropy by assigning weights for each matrix,
            by default ones vector.
        eps
            Small number to avoid division by zero, by default 1e-6.
        tol
            Tolerance for solution convergence, by default 1e-3.
        miter
            Maximum number of MFR iterations, by default 4.
        regularizer
            SVD-family solver class to use, by default `~.SVD`.
        criterion
            Regularization parameter criterion object, by default `~.Lcurve()`.
        store_regularizers
            If True, store regularizer objects at each iteration, by default False.
            The path to store regularizer objects can be specified using `path` argument.
        path
            Directory path to store regularizer objects, by default None.
            If `path` is None, the regularizer objects will be stored in the current directory
            if `store_regularizers` is True.
        use_gpu
            Same as `~.compute_svd`'s `use_gpu` argument, by default False.
        dtype
            Same as `~.compute_svd`'s `dtype` argument, by default numpy.float64.
        verbose
            If True, print iteration information regarding SVD computation, by default False.
        spinner
            If True, show spinner during the computation, by default True.
        **kwargs
            Additional keyword arguments passed to the regularizer class's `~._SVDBase.solve`
            method.

        Returns
        -------
        x : (N, ) array or None
            Optimal solution vector :math:`\mathbf{x}` found by the MFR scheme.
            If the unintended error occurs during the first MFR iteration, the solution will be None.
        status : dict[str, Any]
            Dictionary containing the following keys:

            :`elapsed_time`: elapsed time for the inversion calculation.
            :`niter`: number of iterations.
            :`diffs`: list of differences between the current and previous solutions.
            :`converged`: boolean value indicating the convergence.
            :`regularizer`: regularizer object.

        Raises
        ------
        TypeError
            If `regularizer` is not a subclass of `_SVDBase`, or if `x0` is not a numpy array.
        ValueError
            If the data attribute is not set, or if the initial solution `x0` has incorrect shape.

        Examples
        --------
        >>> x, status = mfr.solve()
        """
        # validate regularizer
        if regularizer is None:
            regularizer = SVD
        if not issubclass(regularizer, _SVDBase):
            raise TypeError("regularizer must be a subclass of _SVDBase")

        if criterion is None:
            criterion = Lcurve()
        elif not isinstance(criterion, Criterion):
            raise TypeError("criterion must be a Criterion instance")

        # check data attribute
        if self._data is None:
            raise ValueError("data attribute is not set")

        # check initial solution
        if x0 is None:
            x0 = np.ones(self._T.shape[1])
        elif isinstance(x0, np.ndarray):
            if x0.ndim != 1:
                raise ValueError("Initial solution must be a 1D array")
            if x0.shape[0] != self._T.shape[1]:
                raise ValueError("Initial solution must have same size as the columns of T")
        else:
            raise TypeError("Initial solution must be a numpy array")

        # check store_regularizers
        _save_path: Path | None = None
        if store_regularizers:
            _save_path = Path.cwd() if path is None else Path(path)

        # set iteration counter and status
        niter = 0
        status: dict[str, Any] = {}
        self._converged = False
        diffs = []
        reg = None
        x = None

        # set timer
        start_time = time()

        # start MFR iteration
        _progress = (
            Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                TimeElapsedColumn(),
            )
            if spinner
            else None
        )
        with _progress or contextlib.nullcontext():
            while niter < miter and not self._converged:
                _task_desc = f"{niter:02}-th MFR iteration"
                task: TaskID | None = _progress.add_task(_task_desc) if _progress else None
                try:
                    # compute regularization matrix
                    H = self.regularization_matrix(
                        x0, eps=eps, derivative_weights=derivative_weights
                    )

                    # compute SVD components
                    svds = compute_svd(
                        self._T,
                        H,
                        Q=self._Q,
                        dtype=dtype,
                        use_gpu=use_gpu,
                        progress=_progress if verbose else None,
                        task_id=task if verbose else None,
                    )

                    # find optimal solution using regularizer class
                    if _progress is not None and task is not None:
                        _progress.update(task, description=f"{_task_desc} (Solving regularizer)")
                    reg = regularizer(*svds, data=self._data)
                    x, _ = reg.solve(criterion, **kwargs)

                    # check convergence
                    diff = np.linalg.norm(x - x0, axis=0)
                    diffs.append(diff)
                    self._converged = bool(diff < tol)

                    # update solution
                    x0 = x

                    # store regularizer object at each iteration
                    if _save_path is not None:
                        with (_save_path / f"regularizer_{niter}.pickle").open("wb") as f:
                            pickle.dump(reg, f)

                    # print iteration information
                    _text = f"(Diff: {diff:.3e}, lambda: {reg.lambda_opt:.3e})"
                    if _progress is not None and task is not None:
                        _progress.update(task, description=f"{_task_desc} {_text}")

                    # update iteration counter
                    niter += 1

                except Exception as e:
                    print(e)
                    break

        elapsed_time = time() - start_time

        # set status
        status["elapsed_time"] = elapsed_time
        status["niter"] = niter
        status["diffs"] = diffs
        status["converged"] = self._converged
        status["regularizer"] = reg

        return x, status

    def regularization_matrix(
        self,
        x: NDArray[np.float64],
        eps: float = 1.0e-6,
        derivative_weights: Collection[float] | None = None,
    ) -> csc_array:
        r"""Compute nonlinear regularization matrix from provided derivative matrices and a solution vector.

        Multiple derivative matrices can be used allowing to combine matrices computed by
        different numerical schemes.

        Each matrix can have different weight coefficients assigned to introduce anisotropy.

        The expression of the regularization matrix :math:`\mathbf{H}(\mathbf{x})` with a solution
        vector :math:`\mathbf{x}` is:

        .. math::

            \mathbf{H}(\mathbf{x})
                = \sum_{\mu,\nu}
                  \alpha_{\mu\nu}
                  \mathbf{D}_\mu^\mathsf{T}
                  \mathbf{W}(\mathbf{x})
                  \mathbf{D}_\nu

        where :math:`\mathbf{D}_\mu` and :math:`\mathbf{D}_\nu` are derivative matrices along to
        :math:`\mu` and :math:`\nu` directions, respectively, :math:`\alpha_{\mu\nu}` is the
        anisotropic coefficient, and :math:`\mathbf{W}` is the diagonal weight matrix defined as
        the inverse of :math:`\mathbf{x}_i`:

        .. math::

            \left[\mathbf{W}\right]_{ij}
                = \frac{\delta_{ij}}{ \max\left(\mathbf{x}_i, \epsilon_0\right) },

        where :math:`\delta_{ij}` is the Kronecker delta, :math:`\mathbf{x}_i` is the :math:`i`-th
        element of the solution vector :math:`\mathbf{x}`, and :math:`\epsilon_0` is a small
        number to avoid division by zero and to push the solution to be positive.

        Parameters
        ----------
        x
            Solution vector :math:`\mathbf{x}`.
        eps
            Small number :math:`\epsilon_0` to avoid division by zero, by default 1.0e-6.
        derivative_weights
            Allows to specify anisotropy by assigning weights :math:`\alpha_{ij}` for each matrix,
            by default ones vector (:math:`\alpha_{ij}=1` for all matrices).

        Returns
        -------
        `scipy.sparse.csc_array`
            Regularization matrix :math:`\mathbf{H}(\mathbf{x})`.

        Raises
        ------
        ValueError
            If `eps` is not positive, or if the number of derivative weight coefficients is not
            equal to the number of derivative matrices.
        """
        # validate eps
        if eps <= 0:
            raise ValueError("eps must be positive small number")

        # set weighting matrix
        w = np.zeros_like(x)
        w[x > eps] = 1 / x[x > eps]
        w[x <= eps] = 1 / eps
        w = diags_array(w)

        if derivative_weights is None:
            derivative_weights = [1.0] * len(self._dmats)
        elif len(derivative_weights) != len(self._dmats):
            raise ValueError(
                "Number of derivative weight coefficients must be equal to number of derivative matrices"
            )

        regularization = csc_array(w.shape, dtype=float)

        for (dmat1, dmat2), aniso in zip(self._dmats, derivative_weights, strict=True):
            regularization = csc_array(regularization + (aniso * dmat1.T @ w @ dmat2))

        return regularization
