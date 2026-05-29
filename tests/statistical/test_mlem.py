# ruff: noqa: N802
import numpy as np
import pytest
from scipy.sparse import csr_array

from cherab.inversion.statistical import MLEM


class TestMLEMInit:
    def test_valid_dense(self, test_tomography_data):
        mlem = MLEM(test_tomography_data.matrix)
        assert mlem.T is test_tomography_data.matrix

    def test_valid_sparse(self, test_tomography_data):
        T = csr_array(test_tomography_data.matrix)
        mlem = MLEM(T)
        assert mlem.T is T

    def test_invalid_not_ndarray(self):
        with pytest.raises(TypeError, match="T must be an ndarray"):
            MLEM([[1, 2], [3, 4]])

    def test_invalid_1d(self):
        with pytest.raises(ValueError, match="T must be a 2D array"):
            MLEM(np.ones(4))

    def test_valid_with_data(self, test_tomography_data):
        T = test_tomography_data.matrix
        mlem = MLEM(T, data=test_tomography_data.b)
        assert mlem.data.shape[0] == T.shape[0]


class TestMLEMData:
    def test_set_1d_data(self, test_tomography_data):
        T = test_tomography_data.matrix
        mlem = MLEM(T)
        mlem.data = test_tomography_data.b
        assert mlem.data.ndim == 1

    def test_set_2d_data(self, test_tomography_data):
        T = test_tomography_data.matrix
        mlem = MLEM(T)
        b = np.column_stack([test_tomography_data.b, test_tomography_data.b])
        mlem.data = b
        assert mlem.data.ndim == 2
        assert mlem.data.shape[0] == T.shape[0]

    def test_invalid_data_wrong_size(self, test_tomography_data):
        T = test_tomography_data.matrix
        mlem = MLEM(T)
        with pytest.raises(ValueError, match="data size must be the same"):
            mlem.data = np.ones(T.shape[0] + 1)

    def test_invalid_data_3d(self, test_tomography_data):
        T = test_tomography_data.matrix
        mlem = MLEM(T)
        with pytest.raises(ValueError, match="data must be a vector or a matrix"):
            mlem.data = np.ones((T.shape[0], 2, 2))


class TestMLEMSolve:
    @pytest.fixture
    def mlem(self, test_tomography_data):
        T = test_tomography_data.matrix
        return MLEM(T, data=test_tomography_data.b)

    def test_solve_returns_correct_shape(self, mlem, test_tomography_data):
        sol, status = mlem.solve(max_iter=5, quiet=True)
        assert isinstance(sol, np.ndarray)
        assert sol.shape == (test_tomography_data.matrix.shape[1],)

    def test_solve_status_keys(self, mlem):
        _, status = mlem.solve(max_iter=5, quiet=True)
        assert "niter" in status
        assert "converged" in status
        assert "diffs" in status
        assert "tol" in status

    def test_solve_non_negative(self, mlem):
        sol, _ = mlem.solve(max_iter=10, quiet=True)
        # Voxels with zero detector coverage may yield NaN; all observed voxels must be >= 0.
        assert np.all(np.isnan(sol) | (sol >= 0))

    def test_solve_store_temp(self, mlem):
        sol, status = mlem.solve(max_iter=5, quiet=True, store_temp=True)
        assert len(status["x_temp"]) == status["niter"]

    def test_solve_raises_without_data(self, test_tomography_data):
        mlem = MLEM(test_tomography_data.matrix)
        mlem._data = None  # explicitly trigger the None check
        with pytest.raises(ValueError, match="data must be set"):
            mlem.solve(quiet=True)

    def test_solve_2d_x0(self, test_tomography_data):
        """Test solve with explicitly provided 2D x0 (covers x0.ndim==2 branch)."""
        T = test_tomography_data.matrix
        b = test_tomography_data.b
        b2d = np.column_stack([b, b])
        mlem = MLEM(T, data=b2d)
        N = T.shape[1]
        x0_2d = np.ones((N, 2))
        sol, _ = mlem.solve(x0=x0_2d, max_iter=5, quiet=True)
        assert sol.shape == (N, 2)

    def test_solve_with_custom_x0(self, mlem, test_tomography_data):
        x0 = np.ones(test_tomography_data.matrix.shape[1])
        sol, status = mlem.solve(x0=x0, max_iter=5, quiet=True)
        assert sol.shape == x0.shape

    def test_solve_x0_ndim_gt2_raises_ValueError(self, mlem, test_tomography_data):
        N = test_tomography_data.matrix.shape[1]
        x0 = np.ones((N, 2, 2))  # 3D array
        with pytest.raises(ValueError, match="x0 must be a vector or a matrix"):
            mlem.solve(x0=x0, quiet=True)

    def test_solve_x0_wrong_size_raises_ValueError(self, mlem, test_tomography_data):
        N = test_tomography_data.matrix.shape[1]
        x0 = np.ones(N // 2)  # wrong size
        with pytest.raises(ValueError, match="x0 must have the same size"):
            mlem.solve(x0=x0, quiet=True)

    def test_solve_2d_data(self, test_tomography_data):
        """Test solve with 2D data (multiple time slices)."""
        T = test_tomography_data.matrix
        b = test_tomography_data.b
        b2d = np.column_stack([b, b])  # shape (M, 2)
        mlem = MLEM(T, data=b2d)
        sol, status = mlem.solve(max_iter=5, quiet=True)
        assert sol.shape == (T.shape[1], 2)

    def test_solve_sparse_matrix(self, test_tomography_data):
        T = csr_array(test_tomography_data.matrix)
        mlem = MLEM(T, data=test_tomography_data.b)
        sol, status = mlem.solve(max_iter=5, quiet=True)
        assert isinstance(sol, np.ndarray)
        assert sol.shape == (T.shape[1],)
