"""Module to offer the Core functionalities for the ill-posed inversion calculation.

This module includes the usefull functions or base classes for the ill-posed inversion calculation
based on Singular Value Decomposition (SVD) method.

The implementation is based on the `inversion theory`_.

.. _inversion theory: ../user/theory/inversion.ipynb
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from numpy import arange, asarray, float64, log10, ndarray, ones_like, sqrt
from numpy import dtype as np_dtype
from scipy.optimize import basinhopping
from scipy.sparse import csc_matrix as sp_csc_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix
from scipy.sparse import issparse, spmatrix
from sksparse.cholmod import cholesky

from .tools.spinner import DummySpinner, Spinner

__all__ = ["_SVDBase", "compute_svd"]


class _SVDBase:
    """Base class for inversion calculation based on Singular Value Decomposition (SVD) method.

    .. note::

        The implementation of this class is based on the `inversion theory`_.

    .. _inversion theory: ../user/theory/inversion.ipynb

    .. note::

        This class is designed to be inherited by subclasses which define the objective function
        to optimize the regularization parameter :math:`\\lambda` using the
        :obj:`~scipy.optimize.basinhopping` function.

    Parameters
    ----------
    s : (r, ) array_like
        Singular values like :math:`\\mathbf{s} = (\\sigma_1, \\sigma_2, ...) \\in \\mathbb{R}^r`.
    U : (M, r) array_like
        Left singular vectors like :math:`\\mathbf{U}\\in\\mathbb{R}^{M\\times r}`.
    basis : (N, r) array_like
        Inverted solution basis like :math:`\\tilde{\\mathbf{V}} \\in \\mathbb{R}^{N\\times r}`.
    B : (M, M) array_like
        Matrix :math:`\\mathbf{B}` coming from :math:`\\mathbf{Q} = \\mathbf{B}^\\mathsf{T}\\mathbf{B}`,
        by default None, i.e. :math:`\\mathbf{B} = \\mathbf{I}`.
    data : (M, ) array_like
        Given data as a vector :math:`\\mathbf{b}\\in\\mathbb{R}^M`, by default None.
    """

    def __init__(self, s, U, basis, B=None, *, data=None):
        # validate SVD components
        s = asarray(s, dtype=float)
        if s.ndim != 1:
            raise ValueError("s must be a vector.")

        U = asarray(U, dtype=float)
        if U.ndim != 2:
            raise ValueError("U must have two dimensions.")
        if s.size != U.shape[1]:
            raise ValueError(
                "the number of columns of U must be same as that of singular values.\n"
                + f"({U.shape[1]=} != {s.size=})"
            )

        # set SVD components
        self._s = s
        self._U = U

        # set inverted solution basis
        self.basis = basis

        # define _B, _data, _ub
        self._B = None
        self._data = None
        self._ub = None

        # set B matrix
        if B is not None:
            self.B = B

        # set data values
        if data is not None:
            self.data = data

        # set initial regularization parameter
        self._beta = 0.0

        # set initial optimal regularization parameter
        self._lambda_opt: float | None = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(s:{self._s.shape}, U:{self._U.shape}, basis:{self._basis.shape})"
        )

    def __getstate__(self):
        """Return the state of the _SVDBase object."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Set the state of the _SVDBase object."""
        self.__dict__.update(state)

    def __reduce__(self):
        return self.__new__, (self.__class__,), self.__getstate__()

    @property
    def s(self) -> ndarray:
        """Singular values :math:`\\mathbf{s}`.

        Singular values form a vector array like
        :math:`\\mathbf{s} = \\begin{bmatrix}\\sigma_1&\\cdots &\\sigma_r\\end{bmatrix}^\\mathsf{T}\\in\\mathbb{R}^r`.
        """
        return self._s

    @property
    def U(self) -> ndarray:
        """Left singular vectors :math:`\\mathbf{U}`.

        Left singular vactors form a matrix containing column vectors like
        :math:`\\mathbf{U}=\\begin{bmatrix}\\mathbf{u}_1 & \\cdots &\\mathbf{u}_r\\end{bmatrix}\\in\\mathbb{R}^{M\\times r}`.
        """
        return self._U

    @property
    def basis(self) -> ndarray:
        """Inverted solution basis :math:`\\tilde{\\mathbf{V}}`.

        The inverted solution basis is a matrix containing column vectors like
        :math:`\\tilde{\\mathbf{V}}=\\begin{bmatrix}\\tilde{\\mathbf{v}}_1&\\cdots\\tilde{\\mathbf{v}}_r\\end{bmatrix}\\in\\mathbb{R}^{N\\times r}`.
        """
        return self._basis

    @basis.setter
    def basis(self, mat):
        if not isinstance(mat, ndarray):
            raise TypeError("basis must be a numpy.ndarray")
        if mat.shape[1] != self._s.size:
            raise ValueError(
                "the number of columns of inverted solution basis must be same as that of singular values.\n"
                + f"({mat.shape[1]=} != {self._s.size=})"
            )
        self._basis = mat

    @property
    def B(self) -> ndarray | spmatrix | None:
        """Matrix :math:`\\mathbf{B}` from :math:`\\mathbf{Q} = \\mathbf{B}^\\mathsf{T}\\mathbf{B}`.

        If users do not specify the matrix :math:`\\mathbf{B}`, this property is None.
        """
        return self._B

    @B.setter
    def B(self, mat):
        if not hasattr(mat, "shape"):
            raise AttributeError("B must have the attribute 'shape'.")
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("B must be a square matrix.")
        if mat.shape[0] != self._U.shape[0]:
            raise ValueError(
                "the number of rows of B must be same as that of U matrix.\n"
                + f"({mat.shape[0]=} != {self._U.shape[0]=})"
            )
        self._B = mat
        if self._data is not None:
            self._ub = self._U.T @ self._B @ self._data

    @property
    def data(self) -> ndarray | None:
        """Given data for inversion calculation :math:`\\mathbf{b}`.

        The given data is a vector array like :math:`\\mathbf{b} \\in \\mathbb{R}^M`.
        """
        return self._data

    @data.setter
    def data(self, value):
        data = asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._U.shape[0]:
            raise ValueError(
                "data size must be the same as the number of rows of U matrix.\n"
                + f"({data.size=} != {self._U.shape[0]=})"
            )
        self._data = data
        if self._B is not None:
            self._ub = self._U.T @ self._B @ data
        else:
            self._ub = self._U.T @ data

    @property
    def bounds(self) -> tuple[float, float]:
        """Bound of log10 of regularization parameter :math:`\\lambda`.

        :math:`\\lambda` is defined in the range of this bounds like
        :math:`\\log_{10}\\lambda \\in (\\log_{10}\\sigma_r^2, \\log_{10}\\sigma_1^2)`,
        where :math:`\\sigma_i` is the :math:`i`-th singular value.

        If users do not specify the bounds for the optimization, this bounds are used by default.
        """
        return (2.0 * log10(self._s[-1]), 2.0 * log10(self._s[0]))

    # -------------------------------------------------------------------------
    # Define methods calculating the residual norm, regularization norm, etc.
    # -------------------------------------------------------------------------

    def filter(self, beta: float) -> ndarray:
        """Calculate the filter factors :math:`f_{\\lambda, i}`.

        The filter factors are diagonal elements of the filter matrix :math:`\\mathbf{F}_\\lambda`,
        and can be expressed with SVD components as follows:

        .. math::

            f_{\\lambda, i} = \\left( 1 + \\frac{\\lambda}{\\sigma_i^2} \\right)^{-1}.

        Parameters
        ----------
        beta : float
            Regularization parameter.

        Returns
        -------
        (r, ) array
            1-D array containing filter factors, the length of which is the same as the number of
            singular values.
        """
        return 1.0 / (1.0 + beta / self._s**2.0)

    def rho(self, beta: float) -> float:
        """Calculate squared residual norm :math:`\\rho`.

        :math:`\\rho` can be expressed with SVD components as follows:

        .. math::

            \\rho &= \\| \\mathbf{T}\\mathbf{x}_\\lambda - \\mathbf{b} \\|_\\mathbf{Q}^2\\\\
                  &= \\|
                        (\\mathbf{F}_\\lambda - \\mathbf{I}_r)
                        \\mathbf{U}^\\mathsf{T}\\mathbf{B}\\mathbf{b}
                     \\|^2.

        Parameters
        ----------
        beta : float
            Regularization parameter.

        Returns
        -------
        float
            Squared residual norm :math:`\\rho`.
        """
        factor = (self.filter(beta) - 1.0) ** 2.0
        return self._ub.dot(factor * self._ub)

    def eta(self, beta: float) -> float:
        """Calculate squared regularization norm :math:`\\eta`.

        :math:`\\eta` can be expressed with SVD components as follows:

        .. math::

            \\eta &= \\|\\mathbf{x}_\\lambda\\|_\\mathbf{H}^2\\\\
                  &= \\|
                        \\mathbf{F}_\\lambda\\mathbf{S}^{-1}
                        \\mathbf{U}^\\mathsf{T}\\mathbf{B}\\mathbf{b}
                     \\|^2

        Parameters
        ----------
        beta : float
            Regularization parameter.

        Returns
        -------
        float
            Squared regularization norm :math:`\\eta`.
        """
        factor = (self.filter(beta) / self._s) ** 2.0
        return self._ub.dot(factor * self._ub)

    def eta_diff(self, beta: float) -> float:
        """Calculate differential of :math:`\\eta`.

        :math:`\\eta'\\equiv\\frac{\\partial\\eta}{\\partial\\lambda}` can be calculated with SVD
        components as follows:

        .. math::

            \\eta' =
                \\frac{2}{\\lambda}
                (\\mathbf{U}^\\mathsf{T}\\mathbf{B}\\mathbf{b})^\\mathsf{T}
                (\\mathbf{F}_\\lambda - \\mathbf{I}_r)
                \\mathbf{F}_\\lambda^2\\mathbf{S}^{-2}\\
                \\mathbf{U}^\\mathsf{T}\\mathbf{B}\\mathbf{b}.

        Parameters
        ----------
        beta : float
            Regularization parameter :math:`\\lambda`.

        Returns
        -------
        float
            Differential of :math:`\\eta` with respect to :math:`\\lambda`.
        """
        filters = self.filter(beta)
        factor = (filters - 1.0) * (filters / self._s) ** 2.0
        return 2.0 * self._ub.dot(factor * self._ub) / beta

    def residual_norm(self, beta: float) -> float:
        """Return the residual norm: :math:`\\sqrt{\\rho} = \\|\\mathbf{T}\\mathbf{x}_\\lambda - \\mathbf{b}\\|_{\\mathbf{Q}}`.

        Parameters
        ----------
        beta : float
            Reguralization parameter.

        Returns
        -------
        float
            Residual norm :math:`\\sqrt{\\rho}`.
        """
        return sqrt(self.rho(beta))

    def regularization_norm(self, beta: float) -> float:
        """Return the regularization norm: :math:`\\sqrt{\\eta} = \\|\\mathbf{x}_\\lambda\\|_\\mathbf{H}`.

        Parameters
        ----------
        beta : float
            Reguralization parameter.

        Returns
        -------
        float
            Regularization norm :math:`\\sqrt{\\eta}`.
        """
        return sqrt(self.eta(beta))

    # ------------------------------------------------------
    # calculating the inverted solution using SVD components
    # ------------------------------------------------------

    def solution(self, beta: float) -> ndarray:
        """Calculate the solution vector :math:`\\mathbf{x}_\\lambda`.

        The solution vector :math:`\\mathbf{x}_\\lambda` can be expressed with SVD components as

        .. math::

            \\mathbf{x}_\\lambda
            =
            \\tilde{\\mathbf{V}}
            \\mathbf{F}_\\lambda
            \\mathbf{S}^{-1}
            \\mathbf{U}^\\mathsf{T}\\mathbf{B}\\mathbf{b}.

        Parameters
        ----------
        beta : float
            Regularization parameter.

        Returns
        -------
        (N, ) array
            Slution vector :math:`\\mathbf{x}_\\lambda`.
        """
        return self._basis @ ((self.filter(beta) / self._s) * self._ub)

    # ------------------------------------------------------
    # Optimization for the regularization parameter
    # ------------------------------------------------------
    @property
    def lambda_opt(self) -> float | None:
        """Optimal regularization parameter defined after `.solve` is executed."""
        return self._lambda_opt

    def solve(
        self,
        bounds: Collection[float | None] | None = None,
        stepsize: float = 10,
        **kwargs,
    ) -> tuple[ndarray, dict]:
        """Solve the ill-posed inversion equation.

        This method is used to seek the optimal regularization parameter finding the global minimum
        of an objective function using the :obj:`~scipy.optimize.basinhopping` function.

        An objective function `_objective_function` must be defined in the subclass.

        Parameters
        ----------
        bounds : Collection[float | None], optional
            Boundary pair of log10 of regularization parameters, by default :obj:`.bounds`.
            Users can specify either lower or upper bound or both like `(1.0, None)`.
        stepsize : float, optional
            Stepsize of optimization, by default 10.
        **kwargs : dict, optional
            Keyword arguments for :obj:`~scipy.optimize.basinhopping` function.

        Returns
        -------
        sol : (N, ) array
            1-D array of the solution vector.
        res : :obj:`~scipy.optimize.OptimizeResult`
            Object returned by :obj:`~scipy.optimize.basinhopping` function.
        """
        # generate bounds
        bounds = self._generate_bounds(bounds)

        # initial guess of log10 of regularization parameter
        init_logbeta = 0.5 * (bounds[0] + bounds[1])

        # optimization
        res = basinhopping(
            self._objective_function,
            x0=10**init_logbeta,
            minimizer_kwargs={"bounds": [bounds]},
            stepsize=stepsize,
            **kwargs,
        )

        # set property of optimal lambda
        _lambda_opt: float = 10 ** res.x[0]
        self._lambda_opt = _lambda_opt

        # optmized solution
        sol = self.solution(_lambda_opt)

        return sol, res

    def _objective_function(self, logbeta: float) -> float:
        raise NotImplementedError("To be defined in subclass.")

    def _generate_bounds(self, bounds: Collection[float | None] | None) -> tuple[float, float]:
        default_lower = self.bounds[0]
        default_upper = self.bounds[1]

        if bounds is None:
            bounds = (default_lower, default_upper)

        if len(bounds) != 2:
            raise ValueError("bounds must contain two elements.")

        lower, upper = bounds

        if lower is None:
            lower = default_lower
        if upper is None:
            upper = default_upper

        if lower >= upper:
            raise ValueError(
                "the first element of bounds must be smaller than the second one. "
                f"({lower} >= {upper})"
            )

        return (lower, upper)


def compute_svd(
    T,
    H,
    Q=None,
    use_gpu=False,
    dtype=None,
    sp: Spinner | DummySpinner | None = None,
) -> tuple[Any, Any, Any] | tuple[Any, Any, Any, Any]:
    """Compute singular value decomposition (SVD) components of the generalized Tikhonov
    regularization problem.

    This function returns the :math:`\\mathbf{s}`, :math:`\\mathbf{U}`, :math:`\\tilde{\\mathbf{V}}`
    and :math:`\\mathbf{B}` from the given matrix :math:`\\mathbf{T}`, :math:`\\mathbf{Q}`, and
    :math:`\\mathbf{H}`.

    .. note::

        The mathmatical notation and calculation procedure is based on the `inversion theory`_.

        .. _inversion theory: ../user/theory/inversion.ipynb

    Parameters
    ----------
    T : (M, N) array_like
        Matrix for a linear equation :math:`\\mathbf{T}\\in\\mathbb{R}^{M\\times N}`.
    H : (N, N) array_like
        Regularization matrix :math:`\\mathbf{H} \\in \\mathbb{R}^{N\\times N}` which must be a
        symmetric positive semi-definite matrix.
    Q : (M, M) array_like, optional
        Weighted matrix for the residual norm :math:`\\mathbf{Q}\\in\\mathbb{R}^{M\\times M}`,
        by default None (meaning :math:`\\mathbf{Q} = \\mathbf{I}`).
        This matrix must be a symmetric positive semi-definite matrix.
    use_gpu : bool, optional
        Whether to use GPU or not, by default False.
        If True, the :obj:`cupy` functionalities is used instead of `numpy` and `scipy` ones when
        calculating the inverse of a sparse matrix, singular value decomposition,
        inverted solution basis :math:`\\tilde{\\mathbf{V}}`, etc.
        Please ensure :obj:`cupy` is installed before using this option,
        otherwise an `ModuleNotFoundError` will be raised.
    dtype : str or numpy dtype, optional
        Data type of the matrix elements, by default numpy.float64.
        In case of using GPU, the data type numpy.float32 is a little bit faster and saves memory.
    sp : `.Spinner` or `.DummySpinner`, optional
        Spinner object to show the progress of calculation, by default `.DummySpinner`.

    Returns
    -------
    s : { (r, ), (r-1, ) } array
        Vector of singular values like
        :math:`\\begin{bmatrix}\\sigma_1&\\cdots&\\sigma_r\\end{bmatrix}^\\mathsf{T}\\in\\mathbb{R}^r`.
        If one set :math:`\\mathbf{T}` as a sparse matrix, :math:`r` is reduced by 1
        (i.e. :math:`r \\to r-1`) because of the use of `scipy.sparse.linalg.eigsh` function to
        calculate the singular values.
    U : (M, r) array
        Left singular vectors like :math:`\\mathbf{U}\\in\\mathbb{R}^{M\\times r}`.
    basis : (N, r) array
        Inverted solution basis like :math:`\\tilde{\\mathbf{V}} \\in \\mathbb{R}^{N\\times r}`.
    B : (M, M) scipy.sparse.csr_matrix
        Matrix :math:`\\mathbf{B}` coming from :math:`\\mathbf{Q} = \\mathbf{B}^\\mathsf{T}\\mathbf{B}`.
        Only returned when :math:`\\mathbf{Q}` is given.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> s, U, basis = compute_svd(T, H)

        >>> s, U, basis, B = compute_svd(T, H, Q, dtype=np.float32, use_gpu=True)
    """
    # === Validation of input parameters ===========================================================
    # import modules
    if use_gpu:
        from cupy import asarray, eye, get_default_memory_pool, get_default_pinned_memory_pool, sqrt
        from cupy.linalg import svd
        from cupyx.scipy.sparse import csc_matrix, csr_matrix, diags
        from cupyx.scipy.sparse.linalg import spsolve_triangular
        from scipy.sparse.linalg import eigsh  # NOTE: cupy eigsh has a bug

        mempool = get_default_memory_pool()
        pinned_mempool = get_default_pinned_memory_pool()
        _cupy_available = True
    else:
        from numpy import asarray, eye, sqrt
        from scipy.linalg import svd
        from scipy.sparse import csc_matrix, csr_matrix, diags
        from scipy.sparse.linalg import eigsh, spsolve_triangular

        _cupy_available = False

    # Set data type
    if dtype is None:
        dtype = float64
    else:
        dtype = np_dtype(dtype)

    # check spinner instance
    if sp is None:
        sp = DummySpinner()
    elif not isinstance(sp, (Spinner, DummySpinner)):
        raise TypeError("sp must be a Spinner or DummySpinner instance.")

    # check T, H matrix dimension
    if hasattr(T, "ndim"):
        if T.ndim != 2 or H.ndim != 2:
            raise ValueError("T and H must be 2-dimensional arrays.")
    else:
        raise AttributeError("T and H must have the attribute 'ndim'.")

    # check T, H matrix shape
    if hasattr(T, "shape"):
        if T.shape[1] != H.shape[0]:
            raise ValueError(
                "the number of columns of T must be same as that of H "
                f"({T.shape[1]=} != {H.shape[0]=})"
            )
        if H.shape[0] != H.shape[1]:
            raise ValueError(f"H must be a square matrix. ({H.shape=})")
        else:
            H = sp_csc_matrix(H, dtype=dtype)
    else:
        raise AttributeError("T and H must have the attribute 'shape'.")

    # check Q matrix
    if Q is not None:
        if hasattr(Q, "ndim"):
            if Q.ndim != 2:
                raise ValueError("Q must be a 2-dimensional array.")
        else:
            raise AttributeError("Q must have the attribute 'ndim'.")

        # check Q matrix shape
        if Q.shape[0] != T.shape[0] or Q.shape[1] != T.shape[0]:
            raise ValueError(
                "Q must be a square matrix with the same number of rows as T. "
                f"({Q.shape[0]=} != {T.shape[0]=}) or ({Q.shape[1]=} != {T.shape[0]=})"
            )
        else:
            Q = sp_csc_matrix(Q, dtype=dtype)

    # Define the base text for the spinner
    _base_text = sp.text + " "
    _use_gpu_text = " by GPU" if _cupy_available else ""

    # === Cholesky decomposition of Q and H matrices ===============================================
    sp.text = _base_text + "(executing cholesekey decomposition)"

    # For Q
    if Q is not None:
        _L_Q_t, _P_Q = _cholesky(Q)
        B = _L_Q_t.tocsr() @ _P_Q.tocsc()
    else:
        B = None

    # For H
    _L_H_t, _P_H = _cholesky(H)

    # === Compute C^-1 matrix ======================================================================
    sp.text = _base_text + f"(computing C^-1 using triangular solver{_use_gpu_text})"

    # Compute L_H^T^-1
    _L_H_t_inv = spsolve_triangular(
        csr_matrix(_L_H_t, dtype=dtype),
        eye(_L_H_t.shape[0], dtype=dtype),
        lower=False,
        overwrite_b=True,
    )
    # Compute C^-1 = P_H^T L_H^T^-1
    C_inv = csc_matrix(_P_H.T, dtype=dtype) @ _L_H_t_inv

    # convert to numpy array from cupy array
    if _cupy_available:
        C_inv = C_inv.get()

        # free GPU memory pools
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    # === Compute SVD components ===================================================================
    if issparse(T):
        # compute A = B @ T @ C^-1
        if B is not None:
            sp.text = _base_text + "(computing A = B @ T @ C^-1)"
            A = (
                csr_matrix(B, dtype=dtype)
                @ csr_matrix(T, dtype=dtype)
                @ csr_matrix(C_inv, dtype=dtype)
            )
        else:
            sp.text = _base_text + "(computing A = T @ C^-1)"
            A = csr_matrix(T, dtype=dtype) @ csr_matrix(C_inv, dtype=dtype)

        # compute AA^T
        sp.text = _base_text + "(computing AA^T)"
        At = A.T
        AAt = A @ At

        # compute eigenvalues and eigenvectors of AA^T
        sp.text = _base_text + f"(computing eigenvalues and vectors of AA^T{_use_gpu_text})"
        # NOTE: cupy eigsh has a bug (https://github.com/cupy/cupy/issues/6446) so
        # scipy.sparse.linalg.eigsh is used instead
        eigvals, u_vecs = eigsh(AAt, k=AAt.shape[0] - 1, which="LM", return_eigenvectors=True)
        # eigvals, u_vecs = eigsh(
        #     csr_matrix(AAt), k=AAt.shape[0] - 1, which="LM", return_eigenvectors=True
        # )

        # compute singular values and left vectors
        sp.text = _base_text + f"(computing singular values and left vectors{_use_gpu_text})"
        singular, u_vecs = _compute_su(
            asarray(eigvals, dtype=dtype), asarray(u_vecs, dtype=dtype), sqrt
        )

        # compute right singular vectors
        sp.text = _base_text + f"(computing right singular vectors{_use_gpu_text})"
        v_mat = (
            asarray(At.toarray(), dtype=dtype)  # type: ignore
            @ asarray(u_vecs, dtype=dtype)
            @ diags(1 / singular, dtype=dtype)
        )

        # compute inverted solution basis
        sp.text = _base_text + f"(computing inverted solution basis{_use_gpu_text})"
        basis = asarray(C_inv, dtype=dtype) @ v_mat

    else:
        # if T is a dense matrix, use SVD solver
        # compute A = B @ T @ C^-1
        if B is not None:
            sp.text = _base_text + "(computing A = B @ T @ C^-1)"
            A = (
                asarray(B.toarray(), dtype=dtype)  # type: ignore
                @ asarray(T, dtype=dtype)
                @ asarray(C_inv, dtype=dtype)
            )
        else:
            sp.text = _base_text + "(computing A = T @ C^-1)"
            A = asarray(T, dtype=dtype) @ asarray(C_inv, dtype=dtype)

        # compute SVD components
        sp.text = _base_text + f"(computing singular value decomposition{_use_gpu_text})"
        kwargs = dict(overwrite_a=True) if not _cupy_available else {}
        u_vecs, singular, vh = svd(A, full_matrices=False, **kwargs)  # type: ignore

        # compute inverted solution basis
        sp.text = _base_text + f"(computing inverted solution basis{_use_gpu_text})"
        basis = asarray(C_inv, dtype=dtype) @ asarray(vh.T, dtype=dtype)

    if _cupy_available:
        singular = singular.get()  # type: ignore
        u_vecs = u_vecs.get()  # type: ignore
        basis = basis.get()  # type: ignore

        # free GPU memory pools
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    # reset spinner text
    sp.text = _base_text

    if B is not None:
        return singular, u_vecs, basis, B

    return singular, u_vecs, basis


def _cholesky(mat: sp_csc_matrix) -> tuple[sp_csr_matrix, sp_csc_matrix]:
    """Cholesky decomposition of a symmetric positive semi-definite matrix.

    Parameters
    ----------
    mat : scipy.sparse.csc_matrix
        Symmetric positive semi-definite matrix.

    Returns
    -------
    L_mat_t : scipy.sparse.csr_matrix
        Cholesky factor :math:`\\mathbf{L}^\\mathsf{T}`.
    P_mat : scipy.sparse.csc_matrix
        Permutation matrix :math:`\\mathbf{P}`.
    """
    # cholesky decomposition of a symmetric positive semi-definite matrix
    factor = cholesky(mat)
    L_mat_t = factor.L().T.tocsr()

    # compute the fill-reducing permutation matrix P
    P_vec = factor.P()
    rows = arange(P_vec.size)
    data = ones_like(rows)
    P_mat = sp_csc_matrix((data, (rows, P_vec)), dtype=float)

    return L_mat_t, P_mat


def _compute_su(eigvals, eigvecs, sqrt: Callable) -> tuple:
    # sort eigenvalues and eigenvectors in descending order
    decend_index = eigvals.argsort()[::-1]
    eigvals = eigvals[decend_index]
    eigvecs = eigvecs[:, decend_index]

    # calculate singular values and left vectors (w/o zero eigenvalues)
    singular = sqrt(eigvals[eigvals > 0])
    u_vecs = eigvecs[:, eigvals > 0]
    return singular, u_vecs
