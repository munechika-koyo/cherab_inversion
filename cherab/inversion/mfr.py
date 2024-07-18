"""Inverses provided data using Minimum Fisher Regularization (MFR) scheme."""

from __future__ import annotations

import pickle
from collections.abc import Collection
from pathlib import Path
from time import time
from typing import Type

import numpy as np
from scipy.sparse import csc_matrix, issparse, sparray, spmatrix
from scipy.sparse import diags as spdiags

from .core import _SVDBase, compute_svd
from .lcurve import Lcurve
from .tools.spinner import DummySpinner, Spinner

__all__ = ["Mfr"]


class Mfr:
    """Inverses provided data using Minimum Fisher Regularization (MFR) scheme.

    .. note::
        The theory and implementation of the MFR are described here_.

    .. _here: ../user/theory/mfr.ipynb

    Parameters
    ----------
    T : (M, N) array_like
        Matrix :math:`\\mathbf{T}\\in\\mathbb{R}^{M\\times N}` of the forward problem
        (geometry matrix, ray transfer matrix, etc.).
    dmats : Collection[Collection[scipy.sparse.spmatrix, scipy.sparse.spmatrix]]
        Iterable of pairs of derivative matrices :math:`\\mathbf{D}_i` and :math:`\\mathbf{D}_j`
        along to :math:`i` and :math:`j` coordinate directions, respectively.
    Q : (M, M) array_like, optional
        Weighted matrix for the residual norm :math:`\\mathbf{Q}\\in\\mathbb{R}^{M\\times M}`,
        by default None (meaning :math:`\\mathbf{Q} = \\mathbf{I}`).
        This matrix must be a symmetric positive semi-definite matrix.
    data : (M, ) array_like, optional
        Given data as a vector :math:`\\mathbf{b}\\in\\mathbb{R}^M`, by default None.

    Examples
    --------
    >>> mfr = Mfr(T, dmats, data=data)
    """

    def __init__(
        self,
        T,
        dmats: Collection[Collection[spmatrix | sparray]],
        *,
        Q=None,
        data=None,
    ):
        # validate arguments
        if not hasattr(T, "ndim"):
            raise TypeError("T must be an array-like object")
        if T.ndim != 2:
            raise ValueError("T must be a 2D array")

        if not isinstance(dmats, Collection):
            raise TypeError("dmats must be a collection of derivative matrices pair.")
        for dmat1, dmat2 in dmats:
            if not issparse(dmat1):
                raise TypeError("one of the matrices in dmats is not a scipy sparse matrix")
            if not issparse(dmat2):
                raise TypeError("one of the matrices in dmats is not a scipy sparse matrix")
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
    def T(self) -> np.ndarray | spmatrix:
        """Matrix :math:`\\mathbf{T}` of the forward problem."""
        return self._T

    @property
    def dmats(self) -> Collection[Collection[spmatrix | sparray]]:
        """List of pairs of derivative matrices :math:`\\mathbf{D}_i` and :math:`\\mathbf{D}_j`.

        Each derivative matrix's subscript represents the coordinate direction.
        """
        return self._dmats

    @property
    def data(self) -> np.ndarray:
        """Given data as a vector :math:`\\mathbf{b}`."""
        return self._data

    @property
    def Q(self) -> np.ndarray | spmatrix | sparray | None:
        """Weighted matrix :math:`\\mathbf{Q}` for the residual norm."""
        return self._Q

    @data.setter
    def data(self, value):
        data = np.asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._T.shape[0]:
            raise ValueError("data size must be the same as the number of rows of geometry matrix")
        self._data = data

    def solve(
        self,
        x0: np.ndarray | None = None,
        derivative_weights: Collection[float] | None = None,
        eps: float = 1.0e-6,
        tol: float = 1e-3,
        miter: int = 4,
        regularizer: Type["_SVDBase"] = Lcurve,
        store_regularizers: bool = False,
        path: str | Path | None = None,
        use_gpu: bool = False,
        dtype=None,
        verbose: bool = False,
        spinner: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray | None, dict]:
        """Solve the inverse problem using MFR scheme.

        MFR is an iterative scheme that combines Singular Value Decomposition (SVD) and a
        optimizer to find the optimal regularization parameter.

        The detailed workflow of the MFR scheme is described in `MFR theory`_.

        .. _MFR theory: ../../user/theory/mfr.ipynb

        Parameters
        ----------
        x0 : (N, ) array, optional
            Initial solution vector, by default ones vector.
        derivative_weights : Collection[float], optional
            Allows to specify anisotropy by assigning weights for each matrix,
            by default ones vector.
        eps : float, optional
            Small number to avoid division by zero, by default 1e-6.
        tol : float, optional
            Tolerance for solution convergence, by default 1e-3.
        miter : int, optional
            Maximum number of MFR iterations, by default 4.
        regularizer : Type[_SVDBase], optional
            Regularizer class to use, by default :obj:`~.Lcurve`.
        store_regularizers : bool, optional
            If True, store regularizer objects at each iteration, by default False.
            The path to store regularizer objects can be specified using `path` argument.
        path : str or `.~pathlib.Path`, optional
            Directory path to store regularizer objects, by default None.
            If `path` is None, the regularizer objects will be stored in the current directory
            if `store_regularizers` is True.
        use_gpu : bool, optional
            Same as :obj:`~.compute_svd`'s `use_gpu` argument, by default False.
        dtype : str or numpy dtype, optional
            Same as :obj:`~.compute_svd`'s `dtype` argument, by default numpy.float64.
        verbose : bool, optional
            If True, print iteration information regarding SVD computation, by default False.
        spinner : bool, optional
            If True, show spinner during the computation, by default True.
        **kwargs : dict, optional
            Additional keyword arguments passed to the regularizer class's :obj:`~._SVDBase.solve`
            method.

        Returns
        -------
        x : (N, ) array or None
            Optimal solution vector :math:`\\mathbf{x}` found by the MFR scheme.
            If the unintended error occurs during the first MFR iteration, the solution will be None.
        status : dict[str, Any]
            Dictionary containing the following keys:
            - `elapsed_time`: elapsed time for the inversion calculation.
            - `niter`: number of iterations.
            - `diffs`: list of differences between the current and previous solutions.
            - `converged`: boolean value indicating the convergence.
            - `regularizer`: regularizer object.

        Examples
        --------
        >>> x, status = mfr.solve()
        """
        # validate regularizer
        if not issubclass(regularizer, _SVDBase):
            raise TypeError("regularizer must be a subclass of _SVDBase")

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
                raise ValueError("Initial solution must have same size as the rows of T")
        else:
            raise TypeError("Initial solution must be a numpy array")

        # check store_regularizers
        if store_regularizers:
            if path is None:
                path: Path = Path.cwd()
            else:
                path: Path = Path(path)

        # set spinner
        if spinner:
            _spinner = Spinner
        else:
            _spinner = DummySpinner

        # set iteration counter and status
        niter = 0
        status = {}
        self._converged = False
        diffs = []
        reg = None
        x = None

        # set timer
        start_time = time()

        # start MFR iteration
        while niter < miter and not self._converged:
            with _spinner(f"{niter:02}-th MFR iteration", timer=True) as sp:
                try:
                    sp_base_text = sp.text + " "

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
                        sp=sp if verbose else None,
                    )

                    # find optimal solution using regularizer class
                    sp.text = sp_base_text + " (Solving regularizer)"
                    reg = regularizer(*svds, data=self._data)
                    x, _ = reg.solve(**kwargs)

                    # check convergence
                    diff = np.linalg.norm(x - x0, axis=0)
                    diffs.append(diff)
                    self._converged = bool(diff < tol)

                    # update solution
                    x0 = x

                    # store regularizer object at each iteration
                    if store_regularizers:
                        with (path / f"regularizer_{niter}.pickle").open("wb") as f:  # type: ignore
                            pickle.dump(reg, f)

                    # print iteration information
                    _text = f"(Diff: {diff:.3e}, lambda: {reg.lambda_opt:.3e})"
                    sp.text = sp_base_text + _text
                    sp.ok()

                    # update iteration counter
                    niter += 1

                except Exception as e:
                    sp.fail()
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
        x: np.ndarray,
        eps: float = 1.0e-6,
        derivative_weights: Collection[float] | None = None,
    ) -> csc_matrix:
        """Compute nonlinear regularization matrix from provided derivative matrices and a solution
        vector.

        Multiple derivative matrices can be used allowing to combine matrices computed by
        different numerical schemes.

        Each matrix can have different weight coefficients assigned to introduce anisotropy.

        The expression of the regularization matrix :math:`\\mathbf{H}(\\mathbf{x})` with a solution
        vector :math:`\\mathbf{x}` is:

        .. math::

            \\mathbf{H}(\\mathbf{x})
                = \\sum_{\\mu,\\nu}
                  \\alpha_{\\mu\\nu}
                  \\mathbf{D}_\\mu^\\mathsf{T}
                  \\mathbf{W}(\\mathbf{x})
                  \\mathbf{D}_\\nu

        where :math:`\\mathbf{D}_\\mu` and :math:`\\mathbf{D}_\\nu` are derivative matrices along to
        :math:`\\mu` and :math:`\\nu` directions, respectively, :math:`\\alpha_{\\mu\\nu}` is the
        anisotropic coefficient, and :math:`\\mathbf{W}` is the diagonal weight matrix defined as
        the inverse of :math:`\\mathbf{x}_i`:

        .. math::

            \\left[\\mathbf{W}\\right]_{ij}
                = \\frac{\\delta_{ij}}{ \\max\\left(\\mathbf{x}_i, \\epsilon_0\\right) },

        where :math:`\\delta_{ij}` is the Kronecker delta, :math:`\\mathbf{x}_i` is the :math:`i`-th
        element of the solution vector :math:`\\mathbf{x}`, and :math:`\\epsilon_0` is a small
        umber to avoid division by zero and to push the solution to be positive.

        Parameters
        ----------
        x : (N, ) array
            Solution vector :math:`\\mathbf{x}`.
        eps : float, optional
            Small number :math:`\\epsilon_0` to avoid division by zero, by default 1.0e-6.
        derivative_weights : Collection[float], optional
            Allows to specify anisotropy by assigning weights :math:`\\alpha_{ij}` for each matrix,
            by default ones vector (:math:`\\alpha_{ij}=1` for all matrices).

        Returns
        -------
        :obj:`scipy.sparse.csc_matrix`
            Regularization matrix :math:`\\mathbf{H}(\\mathbf{x})`.
        """
        # validate eps
        if eps <= 0:
            raise ValueError("eps must be positive small number")

        # set weighting matrix
        w = np.zeros_like(x)
        w[x > eps] = 1 / x[x > eps]
        w[x <= eps] = 1 / eps
        w = spdiags(w)

        if derivative_weights is None:
            derivative_weights = [1.0] * len(self._dmats)
        elif len(derivative_weights) != len(self._dmats):
            raise ValueError(
                "Number of derivative weight coefficients must be equal to number of derivative matrices"
            )

        regularization = csc_matrix(w.shape, dtype=float)

        for (dmat1, dmat2), aniso in zip(self._dmats, derivative_weights):  # noqa: B905  for py39
            regularization += aniso * dmat1.T @ w @ dmat2

        return regularization
