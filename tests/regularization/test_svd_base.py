# ruff: noqa: N802
import pickle

import numpy as np
import pytest
from scipy.linalg import svd
from scipy.sparse import csc_array, csr_array

from cherab.inversion import _SVDBase, compute_svd
from cherab.inversion.regularization import GCV, SVD


class TestComputeSvdValidation:
    """Tests for compute_svd input validation (missing coverage lines)."""

    def test_T_no_ndim_raises_AttributeError(self, test_data):
        T = test_data.matrix
        H = np.eye(T.shape[1])

        class NoNdim:
            dtype = np.float64

        with pytest.raises(AttributeError, match="ndim"):
            compute_svd(NoNdim(), H)

    def test_T_not_2d_raises_ValueError(self, test_data):
        T = test_data.matrix
        H = np.eye(T.shape[1])
        with pytest.raises(ValueError, match="2-dimensional"):
            compute_svd(T.ravel(), H)

    def test_H_not_2d_raises_ValueError(self, test_data):
        T = test_data.matrix
        with pytest.raises(ValueError, match="2-dimensional"):
            compute_svd(T, np.eye(T.shape[1]).ravel())

    def test_T_no_shape_raises_AttributeError(self, test_data):
        """T without 'shape' attribute raises AttributeError."""

        class NoShape:
            ndim = 2
            dtype = np.float64

        with pytest.raises(AttributeError, match="shape"):
            compute_svd(NoShape(), np.eye(4))

    def test_T_H_col_row_mismatch_raises_ValueError(self, test_data):
        T = test_data.matrix
        H = np.eye(T.shape[1] + 1)  # wrong size
        with pytest.raises(ValueError, match="number of columns of T"):
            compute_svd(T, H)

    def test_H_not_square_raises_ValueError(self, test_data):
        T = test_data.matrix
        N = T.shape[1]
        H = np.ones((N, N + 1))  # non-square
        with pytest.raises(ValueError, match="square"):
            compute_svd(T, H)

    def test_Q_no_ndim_raises_AttributeError(self, test_data):
        T = test_data.matrix
        H = np.eye(T.shape[1])

        class NoNdim:
            shape = (T.shape[0], T.shape[0])

        with pytest.raises(AttributeError, match="ndim"):
            compute_svd(T, H, Q=NoNdim())

    def test_Q_not_2d_raises_ValueError(self, test_data):
        T = test_data.matrix
        H = np.eye(T.shape[1])
        with pytest.raises(ValueError, match="2-dimensional"):
            compute_svd(T, H, Q=np.ones(T.shape[0]))

    def test_Q_shape_mismatch_raises_ValueError(self, test_data):
        T = test_data.matrix
        H = np.eye(T.shape[1])
        Q = np.eye(T.shape[0] + 1)  # wrong size
        with pytest.raises(ValueError, match="square matrix with the same number of rows"):
            compute_svd(T, H, Q=Q)


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
    H = csc_array(np.eye(T.shape[1]))
    returns = compute_svd(csr_array(T), H, Q=Q, use_gpu=use_gpu, dtype=dtype)

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
            pytest.param(csr_array(np.eye(64)), True, id="B=I"),
            pytest.param(csr_array(np.eye(64)), False, id="B=I_no_b"),
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
            np.testing.assert_array_equal(svd_base._ub, np.zeros(s.size))

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


class TestSVDBaseInit:
    """Tests for _SVDBase constructor edge cases."""

    def test_matrix_mode(self, test_data):
        """Single positional arg (2D matrix T) triggers matrix mode (lines 110-125)."""
        T = test_data.matrix
        svd_base = _SVDBase(T)
        assert svd_base._s.ndim == 1
        assert svd_base._U.ndim == 2
        assert svd_base._B is None

    def test_matrix_mode_with_Q(self, test_data):
        """Matrix mode with Q specified yields B component (lines 115-120)."""
        T = test_data.matrix
        Q = np.eye(T.shape[0])
        svd_base = _SVDBase(T, Q=Q)
        assert svd_base._B is not None

    def test_4th_positional_arg(self, computed_svd, test_data):
        """4th positional arg is treated as B (line 131)."""
        u, s, vh = computed_svd
        B = csr_array(np.eye(u.shape[0]))
        svd_base = _SVDBase(s, u, vh.T, B)
        assert svd_base._B is B

    def test_init_too_few_args_raises_TypeError(self, computed_svd):
        """Passing only 2 positional args raises TypeError (line 133)."""
        u, s, vh = computed_svd
        with pytest.raises(TypeError, match="Either provide SVD components"):
            _SVDBase(s, u)

    def test_init_s_not_1d_raises_ValueError(self, computed_svd):
        """Non-1D s raises ValueError (line 141)."""
        u, s, vh = computed_svd
        with pytest.raises(ValueError, match="s must be a vector"):
            _SVDBase(s.reshape(1, -1), u, vh.T)

    def test_init_U_not_2d_raises_ValueError(self, computed_svd):
        """Non-2D U raises ValueError (line 145)."""
        u, s, vh = computed_svd
        with pytest.raises(ValueError, match="U must have two dimensions"):
            _SVDBase(s, u.ravel(), vh.T)

    def test_init_U_cols_mismatch_raises_ValueError(self, computed_svd):
        """U column count != s size raises ValueError (line 147)."""
        u, s, vh = computed_svd
        with pytest.raises(ValueError, match="number of columns of U"):
            _SVDBase(s, u[:, :-1], vh.T)  # U with one fewer column


class TestSVDBaseProperties:
    """Tests for _SVDBase property getters and setters."""

    def test_repr(self, svdbase):
        """__repr__ returns formatted string (line 169)."""
        r = repr(svdbase)
        assert svdbase.__class__.__name__ in r

    def test_setstate(self, svdbase):
        """Pickling/unpickling calls __setstate__ (line 178)."""
        restored = pickle.loads(pickle.dumps(svdbase))
        np.testing.assert_array_equal(restored._s, svdbase._s)

    def test_basis_property(self, svdbase):
        """Accessing .basis property returns _basis (line 212)."""
        b = svdbase.basis
        assert b is svdbase._basis

    def test_basis_setter_non_ndarray(self, svdbase):
        """Setting basis with a list triggers asarray conversion (line 217)."""
        new_basis = svdbase._basis.tolist()
        svdbase.basis = new_basis
        assert isinstance(svdbase._basis, np.ndarray)

    def test_basis_setter_wrong_cols_raises_ValueError(self, svdbase):
        """Setting basis with wrong number of columns raises ValueError (line 219)."""
        M, N = svdbase._basis.shape
        with pytest.raises(ValueError, match="number of columns of inverted solution basis"):
            svdbase.basis = np.ones((M, N + 1))

    def test_B_setter_no_shape_raises_AttributeError(self, svdbase):
        """Setting B to an object without 'shape' raises AttributeError (line 237)."""
        with pytest.raises(AttributeError, match="shape"):
            svdbase.B = 42

    def test_B_setter_non_square_raises_ValueError(self, svdbase):
        """Setting non-square B raises ValueError (line 239)."""
        M = svdbase._U.shape[0]
        with pytest.raises(ValueError, match="square"):
            svdbase.B = np.ones((M, M + 1))

    def test_B_setter_size_mismatch_raises_ValueError(self, svdbase):
        """Setting B with wrong size raises ValueError (line 241)."""
        wrong_size = svdbase._U.shape[0] // 2
        with pytest.raises(ValueError, match="number of rows of B"):
            svdbase.B = np.eye(wrong_size)

    def test_B_setter_updates_ub_when_data_set(self, svdbase):
        """Setting B after data is set updates _ub (line 247)."""
        assert svdbase._data is not None  # fixture has data set
        M = svdbase._U.shape[0]
        B = csr_array(np.eye(M))
        svdbase.B = B
        assert svdbase._ub is not None
        np.testing.assert_array_equal(svdbase._ub, svdbase._U.T @ svdbase._B @ svdbase._data)

    def test_data_setter_not_1d_raises_ValueError(self, svdbase):
        """Setting 2D data raises ValueError (line 261)."""
        M = svdbase._U.shape[0]
        with pytest.raises(ValueError, match="data must be a vector"):
            svdbase.data = np.ones((M, 2))

    def test_data_setter_wrong_size_raises_ValueError(self, svdbase):
        """Setting data with wrong size raises ValueError (line 263)."""
        with pytest.raises(ValueError, match="data size must be the same"):
            svdbase.data = np.ones(svdbase._U.shape[0] // 2)


class TestSVDBaseGenerateBounds:
    """Tests for _SVDBase._generate_bounds via SVD.solve()."""

    @pytest.fixture
    def svd_solver(self, computed_svd, test_data):
        u, s, vh = computed_svd
        return SVD(s, u, vh.T, data=test_data.b)

    def test_bounds_wrong_len_raises_ValueError(self, svd_solver):
        """bounds with length != 2 raises ValueError (line 491)."""
        with pytest.raises(ValueError, match="bounds must contain two elements"):
            svd_solver.solve(GCV(), bounds=(-10.0,))

    def test_bounds_lower_none_uses_default(self, svd_solver):
        """bounds lower=None uses default lower bound (line 495)."""
        sol, _ = svd_solver.solve(GCV(), bounds=(None, 2.0))
        assert sol is not None

    def test_bounds_upper_none_uses_default(self, svd_solver):
        """bounds upper=None uses default upper bound (line 497)."""
        sol, _ = svd_solver.solve(GCV(), bounds=(-40.0, None))
        assert sol is not None

    def test_bounds_lower_ge_upper_raises_ValueError(self, svd_solver):
        """bounds where lower >= upper raises ValueError (line 500)."""
        with pytest.raises(ValueError, match="first element of bounds must be smaller"):
            svd_solver.solve(GCV(), bounds=(2.0, 1.0))


def test_compute_svd_with_progress(test_data):
    """Passing progress and task_id to compute_svd covers utility.py line 149."""
    from rich.progress import Progress

    T = test_data.matrix
    H = np.eye(T.shape[1])
    with Progress() as progress:
        task_id = progress.add_task("test svd", total=None)
        returns = compute_svd(T, H, progress=progress, task_id=task_id)
    assert len(returns) == 3
