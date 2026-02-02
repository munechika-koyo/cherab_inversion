import numpy as np
import pytest
from scipy.linalg import svd
from scipy.sparse import csc_matrix, csr_matrix

from cherab.inversion import _SVDBase, compute_svd


@pytest.mark.parametrize(
    ("Q", "use_gpu", "dtype"),
    [
        pytest.param(None, False, None, id="default"),
        pytest.param(None, False, np.float32, id="float32"),
        pytest.param(np.eye(64), False, None, id="Q=I"),
    ],
)
def test_compute_svd(test_data, Q, use_gpu, dtype):
    # Retrieve the test data matrix
    T = test_data.matrix

    # compute SVD
    returns = compute_svd(
        T,
        np.eye(T.shape[1]),
        Q=Q,
        dtype=dtype,
        use_gpu=use_gpu,
    )

    # check return values
    if Q is None:
        assert len(returns) == 3
        s, U, V = returns
    else:
        assert len(returns) == 4
        s, U, V, B = returns

    # compute SVD by numpy
    dtype = T.dtype if dtype is None else dtype
    U_true, s_true, Vh_true = svd(T.astype(dtype), full_matrices=False, overwrite_a=True)

    # check singular values in the range of matrix rank
    rank = np.linalg.matrix_rank(T)
    np.testing.assert_allclose(s[:rank], s_true.astype(dtype)[:rank], rtol=0, atol=1.0e-10)

    # check if U and V are orthogonal matrices
    np.testing.assert_allclose(U_true[:, :rank].T @ U[:, :rank], np.eye(rank), rtol=0, atol=1.0e-6)
    np.testing.assert_allclose(Vh_true[:rank, :] @ V[:, :rank], np.eye(rank), rtol=0, atol=1.0e-6)

    # check Q = B.T @ B
    if Q is not None:
        np.testing.assert_allclose(Q, (B.T @ B).toarray(), rtol=0, atol=1.0e-10)


@pytest.mark.parametrize(
    ("Q", "use_gpu", "dtype"),
    [
        pytest.param(None, False, None, id="sparse"),
        pytest.param(None, False, np.float32, id="sparse_float32"),
        pytest.param(np.eye(48), False, None, id="sparse_Q=I"),
    ],
)
def test_compute_svd_sparse(test_tomography_data, Q, use_gpu, dtype):
    T = test_tomography_data.matrix
    H = csc_matrix(np.eye(T.shape[1]))
    returns = compute_svd(csr_matrix(T), H, Q=Q, use_gpu=use_gpu, dtype=dtype)

    # check return values
    if Q is None:
        assert len(returns) == 3
        s, U, V = returns
    else:
        assert len(returns) == 4
        s, U, V, B = returns

    # compute svd by numpy
    U_true, s_true, Vh_true = svd(T, full_matrices=False, overwrite_a=True)

    # check singular values in the range of matrix rank - 1
    rank = np.linalg.matrix_rank(T)
    np.testing.assert_allclose(s[:rank], s_true[: rank - 1], rtol=0, atol=1.0e-10)

    # check if U and V are orthogonal matrices
    np.testing.assert_allclose(
        np.abs(U_true[:, : rank - 1].T @ U), np.eye(U.shape[1], dtype=dtype), rtol=0, atol=1.0e-4
    )
    np.testing.assert_allclose(
        np.abs(Vh_true[: rank - 1, :] @ V), np.eye(V.shape[1], dtype=dtype), rtol=0, atol=1.0e-4
    )


@pytest.fixture
def svdbase(test_data):
    U_true, s_true, Vh_true = svd(test_data.matrix, full_matrices=False, overwrite_a=True)
    return _SVDBase(s_true, U_true, Vh_true.T, data=test_data.b)


@pytest.fixture
def lambdas():
    return np.logspace(-40, 2, num=500)


class TestSVDBase:
    @pytest.mark.parametrize(
        ("B", "has_b"),
        [
            pytest.param(None, True, id="default"),
            pytest.param(None, False, id="default_no_b"),
            pytest.param(csr_matrix(np.eye(64)), True, id="B=I"),
            pytest.param(csr_matrix(np.eye(64)), False, id="B=I_no_b"),
            pytest.param(np.eye(64), True, id="B=I_dense"),
        ],
    )
    def test__init(self, test_data, computed_svd, B, has_b):
        u, s, vh = computed_svd
        b = test_data.b if has_b else None
        svd_base = _SVDBase(s, u, vh.T, B=B, data=b)

        if has_b:
            assert svd_base._ub.shape == (s.size,)
        else:
            assert svd_base._ub is None

    def test_filter(self, svdbase, lambdas):
        for beta in lambdas:
            filters = svdbase.filter(beta)
            assert isinstance(filters, np.ndarray)
            assert filters.shape == svdbase._s.shape

    def test_rho(self, svdbase, lambdas):
        for beta in lambdas:
            rho = svdbase.rho(beta)
            assert isinstance(rho, float)

    def test_eta(self, svdbase, lambdas):
        for beta in lambdas:
            eta = svdbase.eta(beta)
            assert isinstance(eta, float)

    def test_eta_diff(self, svdbase, lambdas):
        for beta in lambdas:
            eta_diff = svdbase.eta_diff(beta)
            assert isinstance(eta_diff, float)

    def test_residual_norm(self, svdbase, lambdas):
        for beta in lambdas:
            res_norm = svdbase.residual_norm(beta)
            assert isinstance(res_norm, float)

    def test_regularization_norm(self, svdbase, lambdas):
        for beta in lambdas:
            reg_norm = svdbase.regularization_norm(beta)
            assert isinstance(reg_norm, float)

    def test_solution(self, svdbase, lambdas):
        for beta in lambdas:
            sol = svdbase.solution(beta)
            assert isinstance(sol, np.ndarray)
            assert sol.ndim == 1
            assert sol.size == svdbase._basis.shape[0]

    def test__objective_function(self, svdbase):
        with pytest.raises(NotImplementedError):
            svdbase._objective_function(1.0)
