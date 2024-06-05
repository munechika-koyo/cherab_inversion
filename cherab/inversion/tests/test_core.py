import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix

from cherab.inversion import _SVDBase, compute_svd


@pytest.mark.parametrize("use_gpu", [False])
def test_compute_svd(test_data, computed_svd, use_gpu):
    hmat = csc_matrix(np.eye(test_data.matrix.shape[1]))
    s, u, v = compute_svd(test_data.matrix, hmat, use_gpu=use_gpu)

    # compute svd by numpy
    u_np, s_np, vh_np = computed_svd

    # check singular values in the range of matrix rank
    rank = np.linalg.matrix_rank(test_data.matrix)
    np.testing.assert_allclose(s[:rank], s_np[:rank], rtol=0, atol=1.0e-10)

    # TODO: check u and v


def test_compute_svd_sparse(test_tomography_data):
    matrix = csr_matrix(test_tomography_data.matrix)
    hmat = csc_matrix(np.eye(matrix.shape[1]))
    s, u, v = compute_svd(matrix, hmat, use_gpu=False)

    # compute svd by numpy
    u_np, s_np, vh_np = np.linalg.svd(test_tomography_data.matrix, full_matrices=False)

    # check singular values in the range of matrix rank - 1
    rank = np.linalg.matrix_rank(test_tomography_data.matrix)
    np.testing.assert_allclose(s[:rank], s_np[: rank - 1], rtol=0, atol=1.0e-10)

    # TODO: check u and v


@pytest.fixture
def svdbase(test_data, computed_svd):
    u, s, vh = computed_svd
    return _SVDBase(s, u, vh.T, data=test_data.b)


@pytest.fixture
def lambdas():
    return np.logspace(-40, 2, num=500)


class TestSVDBase:
    def test__init(self, test_data, computed_svd):
        u, s, vh = computed_svd
        _SVDBase(s, u, vh.T, data=test_data.b)

    def test__ub(self, svdbase):
        assert isinstance(svdbase._ub, np.ndarray)
        assert svdbase._ub.ndim == 1
        assert svdbase._ub.size == svdbase._s.size

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
