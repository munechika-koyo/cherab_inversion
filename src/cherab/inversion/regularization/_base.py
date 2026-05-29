from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, overload

from numpy import asarray, eye, float64, log10, ndarray, sqrt, zeros
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.sparse._base import _spbase

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

    from .criteria._base import Criterion


class _SVDBase:
    r"""Base class for inversion calculations based on the Singular Value Decomposition (SVD) method.

    This class provides the common mathematical operations for SVD-based inversion solvers.
    The Tikhonov filter is used by default; subclasses may override :meth:`filter` to provide
    alternative filter strategies (e.g. `~.TSVD`).

    To obtain the optimal regularization parameter, pass a `~.Criterion` object to :meth:`solve`.

    Can be constructed in two ways:

    **SVD-component mode**
        Construct directly from precomputed SVD components.

    **Matrix mode**
        Construct from a forward matrix. SVD components are computed internally.

    Parameters
    ----------
    s
        Singular values, :math:`\mathbf{s} = (\sigma_1, \sigma_2, ...) \in \mathbb{R}^r`.
    U
        Left singular vectors, :math:`\mathbf{U}\in\mathbb{R}^{M\times r}`.
    basis
        Inverted solution basis, :math:`\tilde{\mathbf{V}} \in \mathbb{R}^{N\times r}`.
    B
        Matrix :math:`\mathbf{B}` from :math:`\mathbf{Q} = \mathbf{B}^\mathsf{T}\mathbf{B}`.
        Default is None, i.e. :math:`\mathbf{B} = \mathbf{I}`.
    data
        Given data vector, :math:`\mathbf{b}\in\mathbb{R}^M`, by default None.
    T
        Forward matrix :math:`\mathbf{T}\in\mathbb{R}^{M\times N}` (2-D dense or sparse).
        When the first positional argument is 2-D, SVD components are computed internally
        via `~.compute_svd`.
    H
        Regularization matrix :math:`\mathbf{H}\in\mathbb{R}^{N\times N}`, by default
        ``numpy.eye(N)`` (i.e. standard L2 regularization).
    Q
        Weighted matrix :math:`\mathbf{Q}\in\mathbb{R}^{M\times M}`, by default None.
    data
        Given data vector, :math:`\mathbf{b}\in\mathbb{R}^M`, by default None.
    use_gpu
        Whether to use GPU acceleration via `cupy`, by default False.
    dtype
        Data type passed to `~.compute_svd`, by default None (inherits from `.T`).
    sp
        `rich.progress.Progress` instance for displaying computation status, by default None.
    task_id
        Task ID within `~.sp` to update with sub-step descriptions, by default None.
    """

    @overload
    def __init__(
        self,
        s: ArrayLike,
        U: ArrayLike,
        basis: ArrayLike,
        /,
        B: NDArray[float64] | _spbase[float64, tuple[int, int]] | None = None,
        *,
        data: ArrayLike | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        T: NDArray[float64] | _spbase[float64, tuple[int, int]],
        /,
        *,
        H: NDArray[float64] | _spbase[float64, tuple[int, int]] | None = None,
        Q: NDArray[float64] | _spbase[float64, tuple[int, int]] | None = None,
        data: ArrayLike | None = None,
        use_gpu: bool = False,
        dtype: DTypeLike | None = None,
        sp: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> None: ...

    def __init__(
        self,
        *args: Any,
        B: NDArray[float64] | _spbase[float64, tuple[int, int]] | None = None,
        data: ArrayLike | None = None,
        H: NDArray[float64] | _spbase[float64, tuple[int, int]] | None = None,
        Q: NDArray[float64] | _spbase[float64, tuple[int, int]] | None = None,
        use_gpu: bool = False,
        dtype: DTypeLike | None = None,
        sp: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> None:
        n = len(args)
        if n == 1:
            # Matrix mode: single positional arg is the 2-D forward matrix T.
            from .utility import compute_svd

            _T: NDArray[float64] | _spbase[float64, tuple[int, int]] = args[0]
            _H: NDArray[float64] | _spbase[float64, tuple[int, int]] = (
                H if H is not None else eye(_T.shape[1])
            )
            s: Any
            U: Any
            basis: Any
            if Q is not None:
                s, U, basis, B = compute_svd(
                    _T, _H, Q, use_gpu=use_gpu, dtype=dtype, progress=sp, task_id=task_id
                )
            else:
                s, U, basis = compute_svd(
                    _T, _H, None, use_gpu=use_gpu, dtype=dtype, progress=sp, task_id=task_id
                )
                B = None
        elif n >= 3:
            # SVD-component mode: (s, U, basis[, B]).
            s, U, basis = args[0], args[1], args[2]
            # 4th positional arg takes priority over keyword B
            if n >= 4:
                B = args[3]
        else:
            raise TypeError(
                "Either provide SVD components (s, U, basis) as positional arguments, "
                "or a 2-D forward matrix T as the first positional argument."
            )

        # validate SVD components
        s = asarray(s, dtype=float)
        if s.ndim != 1:
            raise ValueError("s must be a vector.")

        U = asarray(U, dtype=float)
        if U.ndim != 2:
            raise ValueError("U must have two dimensions.")
        if s.size != U.shape[1]:
            raise ValueError(
                "the number of columns of U must be same as that of singular values.\n"
                + f"({U.shape[1]=} != {s.size=})"
            )

        self._s = s
        self._U = U
        self.basis = basis

        self._B: NDArray[float64] | _spbase[float64, tuple[int, int]] | None = None
        self._data: NDArray[float64] | None = None
        self._ub: NDArray[float64] = zeros(s.size)

        if B is not None:
            self.B = B
        if data is not None:
            self.data = data

        self._beta = 0.0
        self._lambda_opt: float | None = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(s:{self._s.shape}, U:{self._U.shape}, basis:{self._basis.shape})"
        )

    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __reduce__(self):
        return self.__new__, (self.__class__,), self.__getstate__()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def s(self) -> NDArray[float64]:
        r"""Singular values :math:`\mathbf{s}`.

        Singular values form a vector array like
        :math:`\mathbf{s} = \begin{bmatrix}\sigma_1 & \cdots & \sigma_r\end{bmatrix}^\mathsf{T} \in \mathbb{R}^r`.
        """
        return self._s

    @property
    def U(self) -> NDArray[float64]:
        r"""Left singular vectors :math:`\mathbf{U}`.

        Left singular vectors form a matrix containing column vectors like
        :math:`\mathbf{U}=\begin{bmatrix}\mathbf{u}_1 & \cdots &\mathbf{u}_r\end{bmatrix}\in\mathbb{R}^{M\times r}`.
        """
        return self._U

    @property
    def basis(self) -> NDArray[float64]:
        r"""Inverted solution basis :math:`\tilde{\mathbf{V}}`.

        The inverted solution basis is a matrix containing column vectors like
        :math:`\tilde{\mathbf{V}}=\begin{bmatrix}\tilde{\mathbf{v}}_1&\cdots&\tilde{\mathbf{v}}_r\end{bmatrix}\in\mathbb{R}^{N\times r}`.
        """
        return self._basis

    @basis.setter
    def basis(self, mat: ArrayLike) -> None:
        if not isinstance(mat, ndarray):
            mat = asarray(mat, dtype=float)
        if mat.shape[1] != self._s.size:
            raise ValueError(
                "the number of columns of inverted solution basis "
                "must be same as that of singular values.\n"
                f"({mat.shape[1]=} != {self._s.size=})"
            )
        self._basis = mat

    @property
    def B(self) -> NDArray[float64] | _spbase[float64, tuple[int, int]] | None:
        r"""Matrix :math:`\mathbf{B}` from :math:`\mathbf{Q} = \mathbf{B}^\mathsf{T}\mathbf{B}`.

        If users do not specify the matrix :math:`\mathbf{B}`, this property is None.
        """
        return self._B

    @B.setter
    def B(self, mat: NDArray[float64] | _spbase[float64, tuple[int, int]]) -> None:
        if not hasattr(mat, "shape"):
            raise AttributeError("B must have the attribute 'shape'.")
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("B must be a square matrix.")
        if mat.shape[0] != self._U.shape[0]:
            raise ValueError(
                "the number of rows of B must be same as that of U matrix.\n"
                f"({mat.shape[0]=} != {self._U.shape[0]=})"
            )
        self._B = mat
        if self._data is not None:
            self._ub = self._U.T @ self._B @ self._data

    @property
    def data(self) -> NDArray[float64] | None:
        r"""Given data for inversion calculation :math:`\mathbf{b}`.

        The given data is a vector array like :math:`\mathbf{b} \in \mathbb{R}^M`.
        """
        return self._data

    @data.setter
    def data(self, value: ArrayLike) -> None:
        data = asarray(value, dtype=float)
        if data.ndim != 1:
            raise ValueError("data must be a vector.")
        if data.size != self._U.shape[0]:
            raise ValueError(
                "data size must be the same as the number of rows of U matrix.\n"
                f"({data.size=} != {self._U.shape[0]=})"
            )
        self._data = data
        if self._B is not None:
            self._ub = self._U.T @ self._B @ data
        else:
            self._ub = self._U.T @ data

    @property
    def bounds(self) -> tuple[float, float]:
        r"""Default bounds of :math:`\log_{10}\lambda`: :math:`(\log_{10}\sigma_r^2, \log_{10}\sigma_1^2)`."""
        return (2.0 * log10(self._s[-1]), 2.0 * log10(self._s[0]))

    @property
    def lambda_opt(self) -> float | None:
        """Optimal regularization parameter, set after :meth:`solve` is executed."""
        return self._lambda_opt

    # ------------------------------------------------------------------
    # Mathematical operations
    # ------------------------------------------------------------------

    def filter(self, beta: float) -> NDArray[float64]:
        r"""Tikhonov filter factors :math:`f_{\lambda,i}`.

        The filter factors are diagonal elements of the filter matrix :math:`\mathbf{F}_\lambda`,
        and can be expressed with SVD components as follows:

        .. math::

            f_{\lambda, i} = \left( 1 + \frac{\lambda}{\sigma_i^2} \right)^{-1}.

        Parameters
        ----------
        beta
            Regularization parameter :math:`\lambda`.

        Returns
        -------
        (r,) ndarray
            1-D array containing filter factors, the length of which is the same as the number of
            singular values.
        """
        return 1.0 / (1.0 + beta / self._s**2.0)

    def rho(self, beta: float) -> float:
        r"""Squared residual norm :math:`\rho`.

        :math:`\rho` can be expressed with SVD components as follows:

        .. math::

            \rho &= \| \mathbf{T}\mathbf{x}_\lambda - \mathbf{b} \|_\mathbf{Q}^2\\
                 &= \|
                        (\mathbf{F}_\lambda - \mathbf{I}_r)
                        \mathbf{U}^\mathsf{T}\mathbf{B}\mathbf{b}
                    \|^2.

        Parameters
        ----------
        beta
            Regularization parameter.

        Returns
        -------
        float
            Squared residual norm :math:`\rho`.
        """
        factor = (self.filter(beta) - 1.0) ** 2.0
        return float(self._ub.dot(factor * self._ub))

    def eta(self, beta: float) -> float:
        r"""Squared regularization norm :math:`\eta`.

        :math:`\eta` can be expressed with SVD components as follows:

        .. math::

            \eta &= \|\mathbf{x}_\lambda\|_\mathbf{H}^2\\
                 &= \|
                        \mathbf{F}_\lambda\mathbf{S}^{-1}
                        \mathbf{U}^\mathsf{T}\mathbf{B}\mathbf{b}
                    \|^2

        Parameters
        ----------
        beta
            Regularization parameter.

        Returns
        -------
        float
            Squared regularization norm :math:`\eta`.
        """
        factor: NDArray[float64] = (self.filter(beta) / self._s) ** 2.0
        return float(self._ub.dot(factor * self._ub))

    def eta_diff(self, beta: float) -> float:
        r"""Differential of :math:`\eta`.

        :math:`\eta'\equiv\frac{\partial\eta}{\partial\lambda}` can be calculated with SVD
        components as follows:

        .. math::

            \eta' =
                \frac{2}{\lambda}
                (\mathbf{U}^\mathsf{T}\mathbf{B}\mathbf{b})^\mathsf{T}
                (\mathbf{F}_\lambda - \mathbf{I}_r)
                \mathbf{F}_\lambda^2\mathbf{S}^{-2}\
                \mathbf{U}^\mathsf{T}\mathbf{B}\mathbf{b}.

        Parameters
        ----------
        beta
            Regularization parameter :math:`\lambda`.

        Returns
        -------
        float
            Differential of :math:`\eta` with respect to :math:`\lambda`.
        """
        filters = self.filter(beta)
        factor: NDArray[float64] = (filters - 1.0) * (filters / self._s) ** 2.0
        return 2.0 * float(self._ub.dot(factor * self._ub)) / beta

    def residual_norm(self, beta: float) -> float:
        r"""Residual norm: :math:`\sqrt{\rho} = \|\mathbf{T}\mathbf{x}_\lambda - \mathbf{b}\|_{\mathbf{Q}}`.

        Parameters
        ----------
        beta
            Regularization parameter.

        Returns
        -------
        float
            Residual norm :math:`\sqrt{\rho}`.
        """
        return sqrt(self.rho(beta))

    def regularization_norm(self, beta: float) -> float:
        r"""Regularization norm: :math:`\sqrt{\eta} = \|\mathbf{x}_\lambda\|_\mathbf{H}`.

        Parameters
        ----------
        beta
            Regularization parameter.

        Returns
        -------
        float
            Regularization norm :math:`\sqrt{\eta}`.
        """
        return sqrt(self.eta(beta))

    def solution(self, beta: float) -> ndarray:
        r"""Solution vector :math:`\mathbf{x}_\lambda`.

        The solution vector :math:`\mathbf{x}_\lambda` can be expressed with SVD components as

        .. math::

            \mathbf{x}_\lambda
            =
            \tilde{\mathbf{V}}
            \mathbf{F}_\lambda
            \mathbf{S}^{-1}
            \mathbf{U}^\mathsf{T}\mathbf{B}\mathbf{b}.

        Parameters
        ----------
        beta
            Regularization parameter.

        Returns
        -------
        (N, ) array
            Solution vector :math:`\mathbf{x}_\lambda`.
        """
        return self._basis @ ((self.filter(beta) / self._s) * self._ub)

    # ------------------------------------------------------------------
    # Optimization / solving
    # ------------------------------------------------------------------

    def solve(
        self,
        criterion: Criterion,
        bounds: Collection[float | None] | None = None,
        stepsize: float = 10,
        **kwargs: Any,
    ) -> tuple[ndarray, Any]:
        r"""Solve the ill-posed inversion problem using *criterion* to select :math:`\lambda`.

        Parameters
        ----------
        criterion
            A `~.Criterion` instance (e.g. `~.GCV`, `~.Lcurve`, etc.).
        bounds
            Boundary pair ``(log10_lower, log10_upper)`` for the optimization, by default
            :attr:`bounds`.  Either element may be ``None`` to use the default limit.
        stepsize
            Step-size for `~scipy.optimize.basinhopping`, by default 10.
        **kwargs
            Additional keyword arguments forwarded to the criterion's ``optimize`` method.

        Returns
        -------
        sol : (N,) ndarray
            Optimal solution vector.
        result : :obj:`~scipy.optimize.OptimizeResult`
            Object returned by :obj:`~scipy.optimize.basinhopping` function.
        """
        bounds_resolved = self._generate_bounds(bounds)
        lambda_opt, result = criterion.optimize(self, bounds_resolved, stepsize, **kwargs)
        self._lambda_opt = lambda_opt
        return self.solution(lambda_opt), result

    def _generate_bounds(self, bounds: Collection[float | None] | None) -> tuple[float, float]:
        default_lower, default_upper = self.bounds

        if bounds is None:
            return (default_lower, default_upper)

        if len(bounds) != 2:
            raise ValueError("bounds must contain two elements.")

        lower, upper = bounds
        if lower is None:
            lower = default_lower
        if upper is None:
            upper = default_upper

        if lower >= upper:
            raise ValueError(
                "the first element of bounds must be smaller than the second one. "
                f"({lower} >= {upper})"
            )
        return (lower, upper)
