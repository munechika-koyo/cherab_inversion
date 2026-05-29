"""Utility functions for non-iterative inversion methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

from numpy import arange, asarray, float64, ones_like
from numpy import dtype as np_dtype
from numpy.typing import DTypeLike, NDArray
from rich.progress import Progress, TaskID
from scipy.sparse import csc_array as sp_csc_array
from scipy.sparse import csr_array as sp_csr_array
from scipy.sparse import issparse
from scipy.sparse._base import _spbase
from sksparse.cholmod import cholesky  # type: ignore[import]

__all__ = ["compute_svd"]


@overload
def compute_svd(
    T: NDArray[float64] | _spbase[float64, tuple[int, int]],
    H: NDArray[float64] | _spbase[float64, tuple[int, int]],
    Q: None = None,
    use_gpu: bool = False,
    dtype: DTypeLike | None = None,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]: ...


@overload
def compute_svd(
    T: NDArray[float64] | _spbase[float64, tuple[int, int]],
    H: NDArray[float64] | _spbase[float64, tuple[int, int]],
    Q: NDArray[float64] | _spbase[float64, tuple[int, int]],
    use_gpu: bool = False,
    dtype: DTypeLike | None = None,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> tuple[
    NDArray[float64], NDArray[float64], NDArray[float64], sp_csr_array[float64, tuple[int, int]]
]: ...


def compute_svd(
    T: NDArray[float64] | _spbase[float64, tuple[int, int]],
    H: NDArray[float64] | _spbase[float64, tuple[int, int]],
    Q: NDArray[float64] | _spbase[float64, tuple[int, int]] | None = None,
    use_gpu: bool = False,
    dtype: DTypeLike | None = None,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> (
    tuple[
        NDArray[float64], NDArray[float64], NDArray[float64], sp_csr_array[float64, tuple[int, int]]
    ]
    | tuple[NDArray[float64], NDArray[float64], NDArray[float64]]
):
    r"""Compute singular value decomposition (SVD) components of the generalized Tikhonov regularization problem.

    This function returns the :math:`\mathbf{s}`, :math:`\mathbf{U}`, :math:`\tilde{\mathbf{V}}`
    and :math:`\mathbf{B}` from the given matrix :math:`\mathbf{T}`, :math:`\mathbf{Q}`, and
    :math:`\mathbf{H}`.

    .. note::

        The mathematical notation and calculation procedure is based on the
        :doc:`inversion theory </theory/inversion>`.

    Parameters
    ----------
    T : (M, N) array_like
        Matrix for a linear equation :math:`\mathbf{T}\in\mathbb{R}^{M\times N}`.
    H : (N, N) array_like
        Regularization matrix :math:`\mathbf{H} \in \mathbb{R}^{N\times N}` which must be a
        symmetric positive semi-definite matrix.
    Q : (M, M) array_like, optional
        Weighted matrix for the residual norm :math:`\mathbf{Q}\in\mathbb{R}^{M\times M}`,
        by default None (meaning :math:`\mathbf{Q} = \mathbf{I}`).
        This matrix must be a symmetric positive semi-definite matrix.
    use_gpu
        Whether to use GPU or not, by default False.
        If True, the `cupy` functionalities is used instead of `numpy` and `scipy` ones when
        calculating the inverse of a sparse matrix, singular value decomposition,
        inverted solution basis :math:`\tilde{\mathbf{V}}`, etc.
        Please ensure `cupy` is installed before using this option,
        otherwise an `ModuleNotFoundError` will be raised.
    dtype
        Data type of the matrix elements, by default same as the input matrix `.T`.
        In case of using GPU, the data type numpy.float32 is a little bit faster and saves memory.
    progress
        `rich.progress.Progress` instance for displaying computation status, by default None.
    task_id
        Task ID within *progress* to update with sub-step descriptions, by default None.

    Returns
    -------
    s : { (r, ), (r-1, ) } ndarray
        Vector of singular values like
        :math:`\begin{bmatrix}\sigma_1&\cdots&\sigma_r\end{bmatrix}^\mathsf{T}\in\mathbb{R}^r`.
        If one set :math:`\mathbf{T}` as a sparse matrix, :math:`r` is reduced by 1
        (i.e. :math:`r \to r-1`) because of the use of `scipy.sparse.linalg.eigsh` function to
        calculate the singular values.
    U : (M, r) ndarray
        Left singular vectors like :math:`\mathbf{U}\in\mathbb{R}^{M\times r}`.
    basis : (N, r) ndarray
        Inverted solution basis like :math:`\tilde{\mathbf{V}} \in \mathbb{R}^{N\times r}`.
    B : (M, M) scipy.sparse.csr_array
        Matrix :math:`\mathbf{B}` coming from :math:`\mathbf{Q} = \mathbf{B}^\mathsf{T}\mathbf{B}`.
        Only returned when :math:`\mathbf{Q}` is given.

    Raises
    ------
    AttributeError
        If `T` or `H` (or `Q` if given) do not have the attributes `ndim` or `shape`.
    ValueError
        If the dimensions or shapes of `T`, `H`, or `Q` (if given) are not appropriate.

    Examples
    --------
    >>> s, U, basis = compute_svd(T, H)

    >>> s, U, basis, B = compute_svd(T, H, Q, dtype=np.float32, use_gpu=True)
    """
    # === Validation of input parameters ===========================================================
    # import modules
    if use_gpu:
        from cupy import (  # type: ignore[import-untyped]
            asarray,
            eye,
            get_default_memory_pool,
            get_default_pinned_memory_pool,
            sqrt,
        )
        from cupy.linalg import svd  # type: ignore[import-untyped]
        from cupyx.scipy.sparse import diags  # type: ignore[import-untyped]
        from cupyx.scipy.sparse.linalg import spsolve_triangular  # type: ignore[import-untyped]
        from scipy.sparse.linalg import eigsh  # NOTE: cupy eigsh has a bug

        mempool = get_default_memory_pool()
        pinned_mempool = get_default_pinned_memory_pool()
        _cupy_available = True
    else:
        from numpy import asarray, eye, sqrt
        from scipy.linalg import svd
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh, spsolve_triangular

        mempool = None
        pinned_mempool = None
        _cupy_available = False

    # Set data type
    if dtype is None:
        dtype = T.dtype if hasattr(T, "dtype") else float64
    else:
        dtype = np_dtype(dtype)

    # helper to update progress task description
    def _update(desc: str) -> None:
        if progress is not None and task_id is not None:
            progress.update(task_id, description=_base_desc + desc)

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
            H = sp_csc_array(H, dtype=dtype)  # type: ignore[no-matching-overload]
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
        Q = sp_csc_array(Q, dtype=dtype)  # type: ignore[no-matching-overload]

    # Capture base description for progress updates
    _base_desc = (
        progress.tasks[task_id].description + " "
        if progress is not None and task_id is not None
        else ""
    )
    _use_gpu_text = " by GPU" if _cupy_available else ""

    # === Cholesky decomposition of Q and H matrices ===============================================
    _update("(executing cholesky decomposition)")

    # For Q
    if Q is not None:
        assert isinstance(Q, sp_csc_array)
        _L_Q_t, _P_Q = _cholesky(Q)
        B = _L_Q_t.tocsr() @ _P_Q.tocsc()
    else:
        B = None

    # For H
    assert isinstance(H, sp_csc_array)
    _L_H_t, _P_H = _cholesky(H)

    # === Compute C^-1 matrix ======================================================================
    _update(f"(computing C^-1 using triangular solver{_use_gpu_text})")

    # Compute L_H^T^-1
    _L_H_t_inv = spsolve_triangular(
        _L_H_t.astype(dtype),
        eye(_L_H_t.shape[0], dtype=dtype),
        lower=False,
        overwrite_b=True,
    )
    # Compute C^-1 = P_H^T L_H^T^-1
    C_inv = sp_csc_array(_P_H.T).astype(dtype) @ _L_H_t_inv

    # convert to numpy array from cupy array
    if _cupy_available:
        C_inv = C_inv.get()  # type: ignore[union-attr]

        # free GPU memory pools
        if mempool is not None:
            mempool.free_all_blocks()
        if pinned_mempool is not None:
            pinned_mempool.free_all_blocks()

    # === Compute SVD components ===================================================================
    if issparse(T):
        # compute A = B @ T @ C^-1
        if B is not None:
            _update("(computing A = B @ T @ C^-1)")
            A = (
                sp_csr_array(B).astype(dtype)
                @ sp_csr_array(T).astype(dtype)  # type: ignore[call-overload]
                @ sp_csr_array(C_inv).astype(dtype)
            )
        else:
            _update("(computing A = T @ C^-1)")
            A = sp_csr_array(T).astype(dtype) @ sp_csr_array(C_inv).astype(dtype)  # type: ignore[call-overload]

        # compute AA^T
        _update("(computing AA^T)")
        At = A.T
        AAt = A @ At

        # compute eigenvalues and eigenvectors of AA^T
        _update(f"(computing eigenvalues and vectors of AA^T{_use_gpu_text})")
        # NOTE: cupy eigsh has a bug (https://github.com/cupy/cupy/issues/6446) so
        # scipy.sparse.linalg.eigsh is used instead
        eigvals, u_vecs = eigsh(AAt, k=AAt.shape[0] - 1, which="LM", return_eigenvectors=True)  # type: ignore[call-overload]
        # eigvals, u_vecs = eigsh(
        #     csr_matrix(AAt), k=AAt.shape[0] - 1, which="LM", return_eigenvectors=True
        # )

        # compute singular values and left vectors
        _update(f"(computing singular values and left vectors{_use_gpu_text})")
        singular, u_vecs = _compute_su(
            asarray(eigvals, dtype=dtype), asarray(u_vecs, dtype=dtype), sqrt
        )

        # compute right singular vectors
        _update(f"(computing right singular vectors{_use_gpu_text})")
        v_mat = (
            asarray(At.toarray(), dtype=dtype)  # type: ignore
            @ asarray(u_vecs, dtype=dtype)
            @ diags(1 / singular, dtype=dtype)  # type: ignore[call-overload]
        )

        # compute inverted solution basis
        _update(f"(computing inverted solution basis{_use_gpu_text})")
        basis = asarray(C_inv, dtype=dtype) @ v_mat

    else:
        # if T is a dense matrix, use SVD solver
        # compute A = B @ T @ C^-1
        if B is not None:
            _update("(computing A = B @ T @ C^-1)")
            A = (
                asarray(B.toarray(), dtype=dtype)  # type: ignore
                @ asarray(T, dtype=dtype)
                @ asarray(C_inv, dtype=dtype)
            )
        else:
            _update("(computing A = T @ C^-1)")
            A = asarray(T, dtype=dtype) @ asarray(C_inv, dtype=dtype)

        # compute SVD components
        _update(f"(computing singular value decomposition{_use_gpu_text})")
        kwargs = dict(overwrite_a=True) if not _cupy_available else {}
        u_vecs, singular, vh = svd(A, full_matrices=False, **kwargs)  # type: ignore

        # compute inverted solution basis
        _update(f"(computing inverted solution basis{_use_gpu_text})")
        basis = asarray(C_inv, dtype=dtype) @ asarray(vh.T, dtype=dtype)

    if _cupy_available:
        singular = singular.get()  # type: ignore
        u_vecs = u_vecs.get()  # type: ignore
        basis = basis.get()  # type: ignore

        # free GPU memory pools
        if mempool is not None:
            mempool.free_all_blocks()
        if pinned_mempool is not None:
            pinned_mempool.free_all_blocks()

    if B is not None:
        return singular, u_vecs, basis, B

    return singular, u_vecs, basis


def _cholesky(mat: sp_csc_array) -> tuple[sp_csr_array, sp_csc_array]:
    r"""Cholesky decomposition of a symmetric positive semi-definite matrix.

    Parameters
    ----------
    mat
        Symmetric positive semi-definite matrix.

    Returns
    -------
    L_mat_t : scipy.sparse.csr_array
        Cholesky factor :math:`\mathbf{L}^\\mathsf{T}`.
    P_mat : scipy.sparse.csc_array
        Permutation matrix :math:`\mathbf{P}`.
    """
    # cholesky decomposition of a symmetric positive semi-definite matrix
    factor = cholesky(mat)
    L_mat_t = factor.L().T.tocsr()

    # compute the fill-reducing permutation matrix P
    P_vec = asarray(factor.P())
    rows = arange(P_vec.size)
    data = ones_like(rows, dtype=float64)
    P_mat = sp_csc_array((data, (rows, P_vec)))

    return L_mat_t, P_mat


def _compute_su(eigvals, eigvecs, sqrt: Callable) -> tuple:
    # sort eigenvalues and eigenvectors in descending order
    descend_index = eigvals.argsort()[::-1]
    eigvals = eigvals[descend_index]
    eigvecs = eigvecs[:, descend_index]

    # calculate singular values and left vectors (w/o zero eigenvalues)
    singular = sqrt(eigvals[eigvals > 0])
    u_vecs = eigvecs[:, eigvals > 0]
    return singular, u_vecs
