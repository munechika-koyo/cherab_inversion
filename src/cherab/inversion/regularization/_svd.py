"""Tikhonov regularization via SVD."""

from ._base import _SVDBase

__all__ = ["SVD"]


class SVD(_SVDBase):
    r"""Tikhonov regularization solver based on Singular Value Decomposition.

    This class inherits all methods from `~._SVDBase` and uses the Tikhonov filter:

    .. math::

        f_{\lambda,i} = \left(1 + \frac{\lambda}{\sigma_i^2}\right)^{-1}.

    Pass a `~.Criterion` object (e.g. `~.GCV` or `~.Lcurve`) to :meth:`~._SVDBase.solve`
    to determine the optimal regularization parameter.

    Parameters
    ----------
    *args, **kwargs
        Same as `~._SVDBase`.

    Examples
    --------
    >>> from cherab.inversion import GCV, SVD, compute_svd

    Prepare forward model and data:

    >>> A = np.array([[1, 0], [0, 1], [1, 1]])
    >>> b = np.array([1, 1, 2])

    Then directly input them to `~.SVD`:

    >>> svd = SVD(A, data=b)

    or add data later:

    >>> svd.data = b

    Alternatively, compute SVD components first and then pass them to `~.SVD`:

    >>> s, U, basis = compute_svd(A, np.eye(A.shape[1]))
    >>> svd = SVD(s, U, basis, data=b)

    When the SVD solver is ready, pass a criterion to :meth:`~.SVD.solve` to find the
    optimal regularization parameter and the corresponding solution:

    >>> gcv = GCV()
    >>> sol, status = svd.solve(gcv)
    """
