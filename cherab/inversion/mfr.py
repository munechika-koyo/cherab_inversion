"""Inverses provided data using Minimum Fisher Regularization (MFR) scheme."""
from __future__ import annotations

import pickle
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Type

import numpy as np
from scipy.sparse import csc_matrix, issparse, spmatrix
from scipy.sparse import diags as spdiags

from .core import _SVDBase, compute_svd
from .lcurve import Lcurve
from .tools import Spinner

__all__ = ["Mfr"]


class Mfr:
    """Inverses provided data using Minimum Fisher Regularization (MFR) scheme.

    .. note::
        The theory and implementation of the MFR are described here_.

    .. _here: ../user/theory/mfr.ipynb

    Parameters
    ----------
    gmat : numpy.ndarray (M, N) | scipy.sparse.spmatrix (M, N)
        matrix :math:`\\mathbf{T}\\in\\mathbb{R}^{M\\times N}` of the forward problem
        (geometry matrix, ray transfer matrix, etc.)
    dmats : list[tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix]]
        list of pairs of derivative matrices :math:`\\mathbf{D}_i` and :math:`\\mathbf{D}_j` along
        to :math:`i` and :math:`j` coordinate directions, respectively
    data : numpy.ndarray (M, ), optional
        given data for inversion calculation, by default None

    Examples
    --------
    >>> mfr = Mfr(gmat, dmats, data)
    """

    def __init__(
        self, gmat: np.ndarray | spmatrix, dmats: list[tuple[spmatrix, spmatrix]], data=None
    ):
        # validate arguments
        if not isinstance(gmat, (np.ndarray, spmatrix)):
            raise TypeError("gmat must be a numpy array or a scipy sparse matrix")
        if gmat.ndim != 2:
            raise ValueError("gmat must be a 2D array")

        if not isinstance(dmats, list):
            raise TypeError("dmats must be a list of tuples")
        for dmat1, dmat2 in dmats:
            if not issparse(dmat1):
                raise TypeError("one of the matrices in dmats is not a scipy sparse matrix")
            if not issparse(dmat2):
                raise TypeError("one of the matrices in dmats is not a scipy sparse matrix")

        # set matrix attributes
        self._gmat = gmat
        self._dmats = dmats

        # set data attribute
        if data is not None:
            self.data = data

    @property
    def gmat(self) -> np.ndarray | spmatrix:
        """Geometry matrix :math:`\\mathbf{T}` of the forward problem."""
        return self._gmat

    @property
    def dmats(self) -> list[tuple[spmatrix, spmatrix]]:
        """List of pairs of derivative matrices :math:`\\mathbf{D}_i` and :math:`\\mathbf{D}_j`
        along to :math:`i` and :math:`j` coordinate directions, respectively."""
        return self._dmats

    @property
    def data(self) -> np.ndarray:
        """Given data vector :math:`\\mathbf{b}` for inversion calculation."""
        return self._data

    @data.setter
    def data(self, value):
        data = np.asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._gmat.shape[0]:
            raise ValueError("data size must be the same as the number of rows of geometry matrix")
        self._data = data

    def solve(
        self,
        x0: np.ndarray | None = None,
        derivative_weights: list[float] | tuple[float, ...] | None = None,
        eps: float = 1.0e-6,
        tol: float = 1e-3,
        miter: int = 4,
        regularizer: Type["_SVDBase"] = Lcurve,
        store_regularizers: bool = False,
        path: str | Path | None = None,
        use_gpu: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, dict]:
        """Solves the inverse problem using MFR scheme.

        MFR is an iterative scheme that combines Singular Value Decomposition (SVD) and a
        optimizer to find the optimal regularization parameter.

        The detailed workflow of the MFR scheme is described in `MFR theory`_.

        .. _MFR theory: ../../user/theory/mfr.ipynb

        Parameters
        ----------
        x0 : numpy.ndarray
            initial solution vector, by default ones vector
        derivative_weights
            allows to specify anisotropy by assigning weights for each matrix, by default ones vector
        eps
            small number to avoid division by zero, by default 1e-6
        tol
            tolerance for solution convergence, by default 1e-3
        miter
            maximum number of MFR iterations, by default 4
        regularizer
            regularizer class to use, by default :obj:`~.Lcurve`
        store_regularizers
            if True, store regularizer objects at each iteration, by default False.
            The path to store regularizer objects can be specified using `path` argument.
        path
            directory path to store regularizer objects, by default None.
            If `path` is None, the regularizer objects will be stored in the current directory
            if `store_regularizers` is True.
        use_gpu
            same as :obj:`~.compute_svd`'s `use_gpu` argument, by default False
        verbose
            If True, print iteration information regarding SVD computation, by default False
        **kwargs
            additional keyword arguments passed to the regularizer class's :obj:`~._SVDBase.solve`
            method

        Returns
        -------
        tuple[numpy.ndarray, dict]
            optimal solution vector :math:`\\mathbf{x}` and status dictionary which includes
            the following keys:

            - `elapsed_time`: elapsed time for the inversion calculation
            - `niter`: number of iterations
            - `diffs`: list of differences between the current and previous solutions
            - `converged`: boolean value indicating the convergence
            - `regularizer`: regularizer object

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
            x0 = np.ones(self._gmat.shape[1])
        elif isinstance(x0, np.ndarray):
            if x0.ndim != 1:
                raise ValueError("Initial solution must be a 1D array")
            if x0.shape[0] != self._gmat.shape[1]:
                raise ValueError("Initial solution must have same size as the rows of gmat")
        else:
            raise TypeError("Initial solution must be a numpy array")

        # check store_regularizers
        if store_regularizers:
            if path is None:
                path: Path = Path.cwd()
            else:
                path: Path = Path(path)

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
            with Spinner(f"{niter:02}-th MFR iteration", timer=True) as sp:
                try:
                    sp_base_text = sp.text + " "

                    # compute regularization matrix
                    hmat = self.regularization_matrix(
                        x0, eps=eps, derivative_weights=derivative_weights
                    )

                    # compute SVD components
                    spinner = sp if verbose else None
                    singular, u_vecs, basis = compute_svd(
                        self._gmat, hmat, use_gpu=use_gpu, sp=spinner
                    )

                    # find optimal solution using regularizer class
                    sp.text = sp_base_text + " (Solving regularizer)"
                    reg = regularizer(singular, u_vecs, basis, data=self._data)
                    x, _ = reg.solve(**kwargs)

                    # check convergence
                    diff = np.linalg.norm(x - x0, axis=0)
                    diffs.append(diff)
                    self._converged = bool(diff < tol)

                    # update solution
                    x0 = x

                    # store regularizer object at each iteration
                    if store_regularizers:
                        with (path / f"regularizer_{niter}.pickle").open("wb") as f:
                            pickle.dump(reg, f)

                    # print iteration information
                    _text = (
                        f"(Diff: {diff:.3e}, Tolerance: {tol:.3e}, lambda: {reg.lambda_opt:.3e})"
                    )
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

        print(f"Total elapsed time: {timedelta(seconds=elapsed_time)}")

        return x, status

    def regularization_matrix(
        self,
        x: np.ndarray,
        eps: float = 1.0e-6,
        derivative_weights: list[float] | tuple[float, ...] | None = None,
    ) -> csc_matrix:
        """Computes nonlinear regularization matrix from provided derivative matrices and a solution
        vector.

        Multiple derivative matrices can be used allowing to combine matrices computed by
        different numerical schemes.

        Each matrix can have different weight coefficients assigned to introduce anisotropy.

        The expression of the regularization matrix :math:`\\mathbf{H}(\\mathbf{x})` with a solution
        vector :math:`\\mathbf{x}` is:

        .. math::

            \\mathbf{H}(\\mathbf{x})
                = \\sum_{\\mu,\\nu} \\alpha_{\\mu\\nu} \\mathbf{D}_\\mu^\\mathsf{T} \\mathbf{W}(\\mathbf{x}) \\mathbf{D}_\\nu

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
        x : numpy.ndarray
            solution vector :math:`\\mathbf{x}`
        eps
            small number :math:`\\epsilon_0` to avoid division by zero, by default 1.0e-6
        derivative_weights
            allows to specify anisotropy by assigning weights :math:`\\alpha_{ij}` for each matrix,
            by default ones vector (:math:`\\alpha_{ij}=1` for all matrices)

        Returns
        -------
        :obj:`scipy.sparse.csc_matrix`
            regularization matrix :math:`\\mathbf{H}(\\mathbf{x})`
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

        regularization = csc_matrix(self._dmats[0][0].shape, dtype=float)

        for (dmat1, dmat2), aniso in zip(self._dmats, derivative_weights):  # noqa: B905  for py39
            regularization += aniso * dmat1.T @ w @ dmat2

        return regularization
