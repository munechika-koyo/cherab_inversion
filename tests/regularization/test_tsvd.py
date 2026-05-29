# ruff: noqa: N802
import numpy as np

from cherab.inversion import TSVD, Lcurve
from cherab.inversion.regularization import GCV


class TestTSVD:
    def test_lcurve_discrete_optimize(self, test_data, computed_svd):
        u, sigma, vh = computed_svd
        tsvd = TSVD(sigma, u, vh.T, data=test_data.b)
        lcurve = Lcurve()

        sol, result = tsvd.solve(lcurve)

        assert isinstance(sol, np.ndarray)
        assert sol.ndim == 1
        assert "k" in result
        assert "curvatures" in result
        assert 1 <= result["k"] <= sigma.size

        # Discrete TSVD with L-curve chooses one of beta = sigma^2.
        np.testing.assert_allclose(tsvd.lambda_opt, sigma[result["k"] - 1] ** 2)

        # The curvatures array must contain at least one finite value.
        # (Old buggy code returned all -inf because eta_diff==0 for TSVD,
        # which caused argmax to always pick k=1.)
        curvatures = result["curvatures"]
        assert np.any(np.isfinite(curvatures)), (
            "All curvature values are non-finite; eta_diff==0 bug may have recurred."
        )

        # The chosen k must not be trivially 1 for this well-conditioned test problem.
        assert result["k"] > 1, (
            "L-curve selected k=1 (only one singular value kept), which suggests "
            "the curvature was all -inf and argmax returned index 0."
        )

    def test_solve_criterion_none_keeps_all_components(self, test_data, computed_svd):
        """criterion=None keeps all SVD components (lines 91-92)."""
        u, sigma, vh = computed_svd
        tsvd = TSVD(sigma, u, vh.T, data=test_data.b)

        sol, result = tsvd.solve(criterion=None)

        assert isinstance(sol, np.ndarray)
        assert result == {}
        # All singular values should be used: lambda_opt set just below sigma[-1]^2
        assert tsvd.lambda_opt < sigma[-1] ** 2

    def test_solve_continuous_criterion(self, test_data, computed_svd):
        """Continuous criterion without optimize_discrete uses super().solve() (line 99)."""
        u, sigma, vh = computed_svd
        tsvd = TSVD(sigma, u, vh.T, data=test_data.b)
        gcv = GCV()

        sol, result = tsvd.solve(gcv, bounds=(-20.0, 2.0), stepsize=10)

        assert isinstance(sol, np.ndarray)
        assert sol.ndim == 1
