from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cherab.inversion.derivative import compute_dmat
from cherab.inversion.gcv import GCV
from cherab.inversion.lcurve import Lcurve
from cherab.inversion.mfr import Mfr

BASE = Path(__file__).parent


@pytest.fixture
def mfr(test_tomography_data):
    gmat = csr_matrix(test_tomography_data.matrix)

    # compute derivative matrices
    voxel_map = test_tomography_data.voxel_map
    dmat_r = compute_dmat(voxel_map, kernel_type="r")
    dmat_z = compute_dmat(voxel_map, kernel_type="z")
    dmat_pair = [(dmat_r, dmat_r), (dmat_z, dmat_z)]

    return Mfr(gmat, dmat_pair, test_tomography_data.b)


class TestMfr:
    def test_regularization_matrix(self, mfr):
        x0 = np.ones(mfr.gmat.shape[1])

        mfr.regularization_matrix(x0)

        with pytest.raises(ValueError):
            mfr.regularization_matrix(x0, derivative_weights=[1.0])

        with pytest.raises(ValueError):
            mfr.regularization_matrix(x0, eps=-1.0)

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
