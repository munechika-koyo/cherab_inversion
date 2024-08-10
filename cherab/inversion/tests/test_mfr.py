from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cherab.inversion.derivative import derivative_matrix
from cherab.inversion.gcv import GCV
from cherab.inversion.lcurve import Lcurve
from cherab.inversion.mfr import Mfr


@pytest.fixture
def mfr(test_tomography_data):
    gmat = csr_matrix(test_tomography_data.matrix)

    # compute derivative matrices
    vmap = test_tomography_data.voxel_map
    mask = test_tomography_data.mask
    dmat_r = derivative_matrix(vmap.shape, axis=0, scheme="backward", mask=mask)
    dmat_z = derivative_matrix(vmap.shape, axis=1, scheme="backward", mask=mask)
    dmat_pair = [(dmat_r, dmat_r), (dmat_z, dmat_z)]

    return Mfr(gmat, dmat_pair, data=test_tomography_data.b)


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
        ],
    )
    def test_regularization_matrix(self, mfr, kwargs, expectation):
        x0 = np.ones(mfr.T.shape[1])
        with expectation:
            mfr.regularization_matrix(x0, **kwargs)

    @pytest.mark.parametrize(
        ("regularizer", "store_regularizers"),
        [
            pytest.param(Lcurve, False, id="lcurve"),
            pytest.param(GCV, False, id="gcv"),
            pytest.param(Lcurve, True, id="lcurve_store"),
            pytest.param(GCV, True, id="gcv_store"),
        ],
    )
    def test_solve(self, mfr, tmp_path, regularizer, store_regularizers):
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
            regularizer=regularizer,
            store_regularizers=store_regularizers,
            path=regularizers_dir,
        )

        if regularizers_dir is not None:
            assert len(list(regularizers_dir.glob("*.pickle"))) == status["niter"]
