# ruff: noqa: N802
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from scipy.sparse import csc_array, csr_array

from cherab.inversion.regularization import GCV, MFR, Lcurve
from cherab.inversion.tools._derivative import derivative_matrix


@pytest.fixture
def dmat_pair(test_tomography_data):
    vmap = test_tomography_data.voxel_map
    mask = test_tomography_data.mask
    dmat_r = derivative_matrix(vmap.shape, axis=0, scheme="backward", mask=mask)
    dmat_z = derivative_matrix(vmap.shape, axis=1, scheme="backward", mask=mask)
    return [(dmat_r, dmat_r), (dmat_z, dmat_z)]


@pytest.fixture
def mfr(test_tomography_data, dmat_pair):
    gmat = csr_array(test_tomography_data.matrix)
    return MFR(gmat, dmat_pair, data=test_tomography_data.b)


class TestMfrInit:
    """Tests for MFR.__init__ validation."""

    def test_T_no_ndim_raises_TypeError(self, dmat_pair):
        with pytest.raises(TypeError, match="T must be an array-like object"):
            MFR(42, dmat_pair)

    def test_T_1d_raises_ValueError(self, dmat_pair, test_tomography_data):
        T = test_tomography_data.matrix
        with pytest.raises(ValueError, match="T must be a 2D array"):
            MFR(T.ravel(), dmat_pair)

    def test_dmats_not_collection_raises_TypeError(self, test_tomography_data):
        T = csr_array(test_tomography_data.matrix)
        with pytest.raises(TypeError, match="dmats must be a collection"):
            MFR(T, 42)

    def test_dmat1_not_sparse_raises_TypeError(self, test_tomography_data, dmat_pair):
        T = csr_array(test_tomography_data.matrix)
        N = T.shape[1]
        bad_pair = [(np.eye(N), dmat_pair[0][1])]
        with pytest.raises(TypeError, match="not a scipy sparse array"):
            MFR(T, bad_pair)

    def test_dmat2_not_sparse_raises_TypeError(self, test_tomography_data, dmat_pair):
        T = csr_array(test_tomography_data.matrix)
        N = T.shape[1]
        bad_pair = [(dmat_pair[0][0], np.eye(N))]
        with pytest.raises(TypeError, match="not a scipy sparse array"):
            MFR(T, bad_pair)

    def test_dmat_shape_mismatch_raises_ValueError(self, test_tomography_data, dmat_pair):
        T = csr_array(test_tomography_data.matrix)
        N = T.shape[1]
        d1 = csc_array(np.eye(N))
        d2 = csc_array(np.eye(N + 1))
        bad_pair = [(d1, d2)]
        with pytest.raises(ValueError, match="same shape"):
            MFR(T, bad_pair)

    def test_dmat_not_square_raises_ValueError(self, test_tomography_data):
        T = csr_array(test_tomography_data.matrix)
        M, N = T.shape
        d = csc_array(np.ones((M, N)))
        with pytest.raises(ValueError, match="square"):
            MFR(T, [(d, d)])


class TestMfrProperties:
    """Tests for MFR property getters and data setter."""

    def test_T_property(self, mfr, test_tomography_data):
        assert mfr.T is not None
        assert mfr.T.shape == test_tomography_data.matrix.shape

    def test_dmats_property(self, mfr, dmat_pair):
        assert mfr.dmats is dmat_pair

    def test_Q_property(self, mfr):
        assert mfr.Q is None

    def test_data_property(self, mfr, test_tomography_data):
        np.testing.assert_array_equal(mfr.data, test_tomography_data.b)

    def test_data_setter_not_1d_raises_ValueError(self, mfr):
        M = mfr.T.shape[0]
        with pytest.raises(ValueError, match="data must be a vector"):
            mfr.data = np.ones((M, 2))

    def test_data_setter_wrong_size_raises_ValueError(self, mfr):
        with pytest.raises(ValueError, match="data size must be"):
            mfr.data = np.ones(mfr.T.shape[0] // 2)


class TestMfrSolveValidation:
    """Tests for MFR.solve() validation errors."""

    def test_invalid_regularizer_raises_TypeError(self, mfr):
        with pytest.raises(TypeError, match="regularizer must be a subclass of _SVDBase"):
            mfr.solve(miter=1, regularizer=object, spinner=False)

    def test_invalid_criterion_raises_TypeError(self, mfr):
        with pytest.raises(TypeError, match="criterion must be a Criterion instance"):
            mfr.solve(miter=1, criterion="invalid", spinner=False)

    def test_no_data_raises_ValueError(self, test_tomography_data, dmat_pair):
        T = csr_array(test_tomography_data.matrix)
        mfr_no_data = MFR(T, dmat_pair)
        mfr_no_data._data = None  # explicitly trigger the None check
        with pytest.raises(ValueError, match="data attribute is not set"):
            mfr_no_data.solve(miter=1, spinner=False)

    def test_x0_ndim_ne_1_raises_ValueError(self, mfr):
        N = mfr.T.shape[1]
        with pytest.raises(ValueError, match="Initial solution must be a 1D array"):
            mfr.solve(miter=1, x0=np.ones((N, 2)), spinner=False)

    def test_x0_wrong_shape_raises_ValueError(self, mfr):
        N = mfr.T.shape[1]
        with pytest.raises(ValueError, match="same size as the columns"):
            mfr.solve(miter=1, x0=np.ones(N // 2), spinner=False)

    def test_x0_not_ndarray_raises_TypeError(self, mfr):
        N = mfr.T.shape[1]
        with pytest.raises(TypeError, match="Initial solution must be a numpy array"):
            mfr.solve(miter=1, x0=[1.0] * N, spinner=False)

    def test_store_regularizers_path_none_uses_cwd(self, mfr, tmp_path, monkeypatch):
        """When store_regularizers=True and path=None, uses Path.cwd() (line 228)."""
        monkeypatch.chdir(tmp_path)
        sol, status = mfr.solve(
            miter=1,
            criterion=GCV(),
            store_regularizers=True,
            path=None,
            spinner=False,
        )
        assert len(list(tmp_path.glob("*.pickle"))) == status["niter"]


class TestMfr:
    @pytest.mark.parametrize(
        ["kwargs", "expectation"],
        [
            pytest.param({}, does_not_raise(), id="valid (default)"),
            pytest.param(dict(eps=-1.0), pytest.raises(ValueError), id="invalid (negative eps)"),
            pytest.param(
                dict(derivative_weights=[1]),
                pytest.raises(ValueError),
                id="invalid (less length of derivative weights)",
            ),
            pytest.param(
                dict(derivative_weights=[1.0, 1.0]),
                does_not_raise(),
                id="valid (correct derivative_weights length)",
            ),
        ],
    )
    def test_regularization_matrix(self, mfr, kwargs, expectation):
        x0 = np.ones(mfr.T.shape[1])
        with expectation:
            mfr.regularization_matrix(x0, **kwargs)

    @pytest.mark.parametrize(
        ("criterion", "store_regularizers"),
        [
            pytest.param(Lcurve(), False, id="lcurve"),
            pytest.param(GCV(), False, id="gcv"),
            pytest.param(Lcurve(), True, id="lcurve_store"),
            pytest.param(GCV(), True, id="gcv_store"),
        ],
    )
    def test_solve(self, mfr, tmp_path, criterion, store_regularizers):
        # directory where to store the regularizers
        if store_regularizers:
            regularizers_dir = tmp_path / "regularizers"
            regularizers_dir.mkdir()
        else:
            regularizers_dir = None

        # set the number of iterations to 4
        num_iter = 4

        # solve the MFR problem
        sol, status = mfr.solve(
            miter=num_iter,
            criterion=criterion,
            store_regularizers=store_regularizers,
            path=regularizers_dir,
            spinner=False,
        )

        if regularizers_dir is not None:
            assert len(list(regularizers_dir.glob("*.pickle"))) == status["niter"]

    def test_solve_with_valid_x0(self, mfr):
        """Passing valid 1D x0 covers the x0-shape False-branch (line 220->226)."""
        N = mfr.T.shape[1]
        x0 = np.ones(N)
        sol, status = mfr.solve(miter=1, x0=x0, spinner=False)
        assert sol.shape == (N,)

    def test_solve_spinner_true(self, mfr):
        """spinner=True creates a Progress object, covering lines 276 and 296."""
        sol, status = mfr.solve(miter=1, spinner=True)
        assert "niter" in status
