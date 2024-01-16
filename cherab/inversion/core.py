"""Module to offer the Core functionalities for the ill-posed inversion calculation.

This module includes the usefull functions or base classes for the ill-posed inversion calculation
based on Singular Value Decomposition (SVD) method.

The implementation is based on the `inversion theory`_.

.. _inversion theory: ../user/theory/inversion.ipynb
"""
from __future__ import annotations

from collections.abc import Callable

from numpy import arange, asarray, ndarray, ones_like, sqrt
from scipy.optimize import basinhopping
from scipy.sparse import csc_matrix as sp_csc_matrix
from scipy.sparse import csr_matrix as sp_csr_matrix
from scipy.sparse import issparse
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
    s : vector_like
        singular values like :math:`\\mathbf{s} = (\\sigma_1, \\sigma_2, ...) \\in \\mathbb{R}^r`
    u : array_like
        left singular vectors like :math:`\\mathbf{U}\\in\\mathbb{R}^{M\\times r}`
    basis : array_like
        inverted solution basis like :math:`\\tilde{\\mathbf{V}} \\in \\mathbb{R}^{N\\times r}`.
    data : vector_like
        given data as a vector in :math:`\\mathbb{R}^M`, by default None.
    """

    def __init__(self, s, u, basis, data=None):
        # validate SVD components
        s = asarray(s, dtype=float)
        if s.ndim != 1:
            raise ValueError("s must be a vector.")

        u = asarray(u, dtype=float)
        if u.ndim != 2:
            raise ValueError("u must have two dimensions.")
        if s.size != u.shape[1]:
            raise ValueError(
                "the number of columns of u must be same as that of singular values.\n"
                + f"({u.shape[1]=} != {s.size=})"
            )

        # set SVD components
        self._s = s
        self._u = u

        # set inverted solution basis
        self.basis = basis

        # set data values
        if data is not None:
            self.data = data
        else:
            self._data = None

        # set initial regularization parameter
        self._beta = 0.0

        # set initial optimal regularization parameter
        self._lambda_opt: float | None = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(s:{self._s.shape}, u:{self._u.shape}, basis:{self._basis.shape})"
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
        :math:`\\mathbf{s} = (\\sigma_1, \\sigma_2,...)\\in\\mathbb{R}^r`
        """
        return self._s

    @property
    def u(self) -> ndarray:
        """Left singular vectors :math:`\\mathbf{U}`.

        Left singular vactors form a matrix containing column vectors like
        :math:`\\mathbf{U}\\in\\mathbb{R}^{M\\times r}`
        """
        return self._u

    @property
    def basis(self) -> ndarray:
        """Inverted solution basis :math:`\\tilde{\\mathbf{V}}`.

        The inverted solution basis is a matrix containing column vectors like
        :math:`\\tilde{\\mathbf{V}} \\in \\mathbb{R}^{n\\times r}`.
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
    def data(self) -> ndarray:
        """Given data for inversion calculation :math:`\\mathbf{b}`.

        The given data is a vector array like :math:`\\mathbf{b} \\in \\mathbb{R}^M`.
        """
        return self._data

    @data.setter
    def data(self, value):
        data = asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._u.shape[0]:
            raise ValueError(
                "data size must be the same as the number of rows of U matrix.\n"
                + f"({data.size=} != {self._u.shape[0]=})"
            )
        self._data = data
        self._ub = self._u.T @ data

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
        beta
            regularization parameter

        Returns
        -------
        numpy.ndarray (N, )
            1-D array containing filter factors, the length of which is the same as the number of
            singular values.
        """
        return 1.0 / (1.0 + beta / self._s**2.0)

    def rho(self, beta: float) -> float:
        """Calculate squared residual norm :math:`\\rho`.

        :math:`\\rho` can be expressed with SVD components as follows:

        .. math::

            \\rho &= \\| \\mathbf{T}\\mathbf{x}_\\lambda - \\mathbf{b} \\|_2^2\\\\
                  &= \\| (\\mathbf{F}_\\lambda - \\mathbf{I}_r)\\mathbf{U}^\\mathsf{T}\\mathbf{b} \\|_2^2.

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        float
            squared residual norm :math:`\\rho`
        """
        factor = (self.filter(beta) - 1.0) ** 2.0
        return self._ub.dot(factor * self._ub)

    def eta(self, beta: float) -> float:
        """Calculate squared regularization norm :math:`\\eta`.

        :math:`\\eta` can be expressed with SVD components as follows:

        .. math::

            \\eta &= \\mathbf{x}_\\lambda^\\mathsf{T}\\mathbf{H}\\mathbf{x}_\\lambda\\\\
                  &= \\|\\mathbf{F}_\\lambda\\mathbf{S}^{-1}\\mathbf{U}^\\mathsf{T}\\mathbf{b}\\|_2^2

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        float
            squared regularization norm :math:`\\eta`
        """
        factor = (self.filter(beta) / self._s) ** 2.0
        return self._ub.dot(factor * self._ub)

    def eta_diff(self, beta: float) -> float:
        """Calculate differential of :math:`\\eta` with respect to regularization parameter
        :math:`\\lambda`.

        :math:`\\eta` can be calculated with SVD components as follows:

        .. math::

            \\eta' =
                \\frac{2}{\\lambda}
                (\\mathbf{U}^\\mathsf{T}\\mathbf{b})^\\mathsf{T}
                (\\mathbf{F}_\\lambda - \\mathbf{I}_r)
                \\mathbf{F}_\\lambda^2\\mathbf{S}^{-2}\\
                \\mathbf{U}^\\mathsf{T}\\mathbf{b}.

        Parameters
        ----------
        beta
            regularization parameter :math:`\\lambda`

        Returns
        -------
        float
            differential of :math:`\\eta` with respect to :math:`\\lambda`
        """
        filters = self.filter(beta)
        factor = (filters - 1.0) * (filters / self._s) ** 2.0
        return 2.0 * self._ub.dot(factor * self._ub) / beta

    def residual_norm(self, beta: float) -> float:
        """Return the residual norm:
        :math:`\\sqrt{\\rho} = \\|\\mathbf{T}\\mathbf{x}_\\lambda - \\mathbf{b}\\|_2`

        Parameters
        ----------
        beta
            reguralization parameter

        Returns
        -------
        float
            residual norm :math:`\\sqrt{\\rho}`
        """
        return sqrt(self.rho(beta))

    def regularization_norm(self, beta: float) -> float:
        """Return the regularization norm:
        :math:`\\sqrt{\\eta} = \\sqrt{\\mathbf{x}_\\lambda^\\mathsf{T}\\mathbf{H}\\mathbf{x}_\\lambda}`

        Parameters
        ----------
        beta
            reguralization parameter

        Returns
        -------
        float
            regularization norm :math:`\\sqrt{\\eta}`
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
            \\mathbf{U}^\\mathsf{T}\\mathbf{b}.

        Parameters
        ----------
        beta
            regularization parameter

        Returns
        -------
        numpy.ndarray (N, )
            solution vector :math:`\\mathbf{x}_\\lambda`
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
        bounds: tuple[float, float] = (-20.0, 2.0),
        stepsize: float = 10,
        **kwargs,
    ) -> tuple[ndarray, dict]:
        """Solve the ill-posed inversion equation.

        This method is used to seek the optimal regularization parameter finding the global minimum
        of an objective function using the :obj:`~scipy.optimize.basinhopping` function.

        An objective function `_objective_function` must be defined in the subclass.

        Parameters
        ----------
        bounds
            bounds of log10 of regularization parameter, by default (-20.0, 2.0).
        stepsize
            stepsize of optimization, by default 10.
        **kwargs
            keyword arguments for :obj:`~scipy.optimize.basinhopping` function.

        Returns
        -------
        tuple of :obj:`~numpy.ndarray` and :obj:`~scipy.optimize.OptimizeResult`
            (`sol`, `res`), where `sol` is the 1-D array of the solution vector
            and `res` is the :obj:`~scipy.optimize.OptimizeResult` object returned by
            :obj:`~scipy.optimize.basinhopping` function.
        """
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
        self._lambda_opt = 10 ** res.x[0]

        # optmized solution
        sol = self.solution(beta=self._lambda_opt)

        return sol, res

    def _objective_function(self, logbeta: float) -> float:
        raise NotImplementedError("To be defined in subclass.")


def compute_svd(
    gmat,
    hmat,
    use_gpu=False,
    sp: Spinner | DummySpinner | None = None,
) -> tuple[ndarray, ndarray, ndarray]:
    """Computes singular value decomposition (SVD) components of the geometry matrix
    :math:`\\mathbf{T}` and regularization matrix :math:`\\mathbf{H}`.

    .. note::

        The calculation procedure is based on the `inversion theory`_.

        .. _inversion theory: ../user/theory/inversion.ipynb

    Parameters
    ----------
    gmat : numpy.ndarray | scipy.sparse.spmatrix
        matrix for a linear equation which is called geometry matrix in tomography field
        spesifically, :math:`\\mathbf{T}\\in\\mathbb{R}^{M\\times N}`
    hmat : scipy.sparse.spmatrix
        regularization matrix :math:`\\mathbf{H} \\in \\mathbb{R}^{N\\times N}`.
        :math:`\\mathbf{H}` must be a positive definite matrix.
    use_gpu : bool, optional
        whether to use GPU or not, by default False.
        If True, the :obj:`cupy` functionalities is used instead of numpy and scipy ones when
        calculating the inverse of a sparse matrix, singular value decomposition,
        inverted solution basis :math:`\\tilde{\\mathbf{V}}`, etc.
        Please ensure :obj:`cupy` is installed before using this option,
        otherwise an ModuleNotFoundError will be raised.
    sp : Spinner or DummySpinner, optional
        spinner object to show the progress of calculation, by default DummySpinner()

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        singular value vectors :math:`\\mathbf{s}\\in\\mathbb{R}^r`, left singular vectors
        :math:`\\mathbf{U}\\in\\mathbb{R}^{M\\times r}` and inverted solution basis
        :math:`\\tilde{\\mathbf{V}}`

    Examples
    --------
    .. prompt:: python >>> auto

        >>> s, u, basis = compute_svd(gmat, hmat, use_gpu=True)
    """
    # === Validation of input parameters ===========================================================
    # import modules
    if use_gpu:
        from cupy import asarray, eye, get_default_memory_pool, get_default_pinned_memory_pool, sqrt
        from cupy.linalg import svd
        from cupyx.scipy.sparse import csr_matrix, diags
        from cupyx.scipy.sparse.linalg import spsolve_triangular
        from scipy.sparse.linalg import eigsh  # NOTE: cupy eigsh has a bug

        mempool = get_default_memory_pool()
        pinned_mempool = get_default_pinned_memory_pool()
        _cupy_available = True
    else:
        from numpy import asarray, eye, sqrt
        from scipy.linalg import svd
        from scipy.sparse import csr_matrix, diags
        from scipy.sparse.linalg import eigsh, spsolve_triangular

        _cupy_available = False

    # check if hmat is a sparse matrix
    if not issparse(hmat):
        raise TypeError("hmat must be a scipy.sparse.spmatrix.")
    else:
        hmat = sp_csc_matrix(hmat)

    # check matrix dimension
    if hasattr(gmat, "ndim"):
        if gmat.ndim != 2 or hmat.ndim != 2:
            raise ValueError("gmat and hmat must be 2-dimensional arrays.")
    else:
        raise AttributeError("gmat and hmat must have the attribute 'ndim'.")

    # check matrix shape
    if hasattr(gmat, "shape"):
        if gmat.shape[1] != hmat.shape[0]:
            raise ValueError("the number of columns of gmat must be same as that of hmat")
        if hmat.shape[0] != hmat.shape[1]:
            raise ValueError("hmat must be a square matrix.")
    else:
        raise AttributeError("gmat and hmat must have the attribute 'shape'.")

    # check spinner instance
    if sp is None:
        sp = DummySpinner()
    elif not isinstance(sp, (Spinner, DummySpinner)):
        raise TypeError("sp must be a Spinner or DummySpinner instance.")

    _base_text = sp.text + " "
    _use_gpu_text = " by GPU" if _cupy_available else ""
    # ==============================================================================================

    # compute L and P^T using cholesekey decomposition
    sp.text = _base_text + "(computing L and P^T using cholesekey decomposition)"
    L_mat, Pt = _compute_L_Pt(hmat)

    # compute L^{-T} using triangular solver
    sp.text = _base_text + f"(computing L^-T using triangular solver{_use_gpu_text})"
    Lt_inv = spsolve_triangular(
        csr_matrix(L_mat), eye(L_mat.shape[0]), lower=True, overwrite_b=True
    ).T

    # convert to numpy array from cupy array
    if _cupy_available:
        Lt_inv = Lt_inv.get()

        # free GPU memory pools
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    # compute Pt @ Lt^{-1}
    # This calculation is performed in CPU because the performance of cupy is worse than numpy or
    # scipy in this calculation.
    sp.text = _base_text + "(computing Pt @ Lt^-1)"
    Pt_Lt_inv: sp_csr_matrix = Pt @ sp_csr_matrix(Lt_inv)

    if issparse(gmat):
        # compute A = gmat @ Pt @ Lt^{-1}
        sp.text = _base_text + "(computing A = gmat @ Pt @ L^-T)"
        A_mat: sp_csr_matrix = gmat.tocsc() @ Pt_Lt_inv

        # compute AA^T
        sp.text = _base_text + "(computing AA^T)"
        At = A_mat.T
        AAt = A_mat @ At

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
        singular, u_vecs = _compute_su(asarray(eigvals), asarray(u_vecs), sqrt)

        # compute right singular vectors
        sp.text = _base_text + f"(computing right singular vectors{_use_gpu_text})"
        v_mat = asarray(At.A) @ asarray(u_vecs) @ diags(1 / singular)

        # compute inverted solution basis
        sp.text = _base_text + f"(computing inverted solution basis{_use_gpu_text})"
        basis = asarray(Pt_Lt_inv.A) @ v_mat

    else:
        # if gmat is a dense matrix, use SVD solver
        # compute A = gmat @ Pt @ Lt^{-1}
        sp.text = _base_text + "(computing A = gmat @ Pt @ L^-T)"
        A_mat: ndarray = gmat @ Pt_Lt_inv.A

        # compute SVD components
        sp.text = _base_text + f"(computing SVD components directory{_use_gpu_text})"
        kwargs = dict(overwrite_a=True) if not _cupy_available else {}
        u_vecs, singular, vh = svd(asarray(A_mat), full_matrices=False, **kwargs)

        # compute inverted solution basis
        sp.text = _base_text + f"(computing inverted solution basis{_use_gpu_text})"
        basis = asarray(Pt_Lt_inv.A) @ asarray(vh.T)

    if _cupy_available:
        singular = singular.get()
        u_vecs = u_vecs.get()
        basis = basis.get()

        # free GPU memory pools
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    # reset spinner text
    sp.text = _base_text

    return singular, u_vecs, basis


def _compute_L_Pt(hmat: sp_csc_matrix) -> tuple[sp_csr_matrix, sp_csr_matrix]:
    # cholesky decomposition of H
    factor = cholesky(hmat)
    L_mat = factor.L().tocsr()

    # compute the fill-reducing permutation matrix P
    P_vec = factor.P()
    rows = arange(P_vec.size)
    data = ones_like(rows)
    P_mat = sp_csc_matrix((data, (rows, P_vec)), dtype=float)

    return L_mat, P_mat.T


def _compute_su(eigvals, eigvecs, sqrt: Callable):
    # sort eigenvalues and eigenvectors in descending order
    decend_index = eigvals.argsort()[::-1]
    eigvals = eigvals[decend_index]
    eigvecs = eigvecs[:, decend_index]

    # calculate singular values and left vectors (w/o zero eigenvalues)
    singular = sqrt(eigvals[eigvals > 0])
    u_vecs = eigvecs[:, eigvals > 0]
    return singular, u_vecs
