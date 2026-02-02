"""Module to offer the function to generate a derivative matrix."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy import ndarray
from scipy.sparse import csc_array, csr_array, dia_array, diags_array, lil_array

__all__ = ["diag_dict", "derivative_matrix", "laplacian_matrix", "Derivative"]


def diag_dict(grid_shape: tuple[int, int]) -> dict[str, dia_array]:
    """Return a dictionary of diagonal matrices.

    The key of the dictionary corresponds to the position of grid points.
    `b`, `c`, and `f` mean backward, center, and forward along the given axis respectively.
    e.g. `bf` means the backward and forward grid points along the axis 0 and 1 respectively.

    The following figure shows how the each grid point is connected.

    .. code-block:: none

        bb ─── bc ─── bf    --> axis 1
        │       │       │
        │       │       │
        cb ─── cc ─── cf
        │       │       │
        │       │       │
        fb ─── fc ─── ff

        |
        V
        axis 0

    A grid point is regarded to be flattened along the axis 1, so if the index of ``cc`` is ``i``,
    then the index of ``bc`` is ``i - N1``, ``fc`` is ``i + N1``, etc. where ``N1`` is the number
    of grid points along the axis 1.
    If the index is out of the grid, then the corresponding element is set to zero (dirichlet
    boundary condition).

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Shape of the grid (N0, N1), where N0 and N1 are the number of grid points along the axis 0
        and 1 respectively.

    Returns
    -------
    dict[str, `scipy.sparse.dia_array`]
        Dictionary of diagonal matrices, the keys of which are ``"bb"``, ``"bc"``, ``"bf"``,
        ``"cb"``, ``"cc"``, ``"cf"``, ``"fb"``, ``"fc"``, and ``"ff"``.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> diag = diag_dict((3, 3))
        >>> diag["cc"].toarray()
        array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        >>> diag["bf"].toarray()
        array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    n0, n1 = grid_shape
    bins = n0 * n1

    # list of each grid position index including the dirichlet boundary condition
    cb = cf = np.tile([1] * (n1 - 1) + [0], n0)[: bins - 1]
    fb = bf = np.tile([0] + [1] * (n1 - 1), n0)[: bins - n1 + 1]
    ff = bb = np.tile([1] * (n1 - 1) + [0], n0)[: bins - n1 - 1]

    return {
        "bb": diags_array(bb, offsets=-n1 - 1, shape=(bins, bins)),
        "bc": diags_array([1], offsets=-n1, shape=(bins, bins)),
        "bf": diags_array(bf, offsets=-n1 + 1, shape=(bins, bins)),
        "cb": diags_array(cb, offsets=-1, shape=(bins, bins)),
        "cc": diags_array([1], offsets=0, shape=(bins, bins)),
        "cf": diags_array(cf, offsets=1, shape=(bins, bins)),
        "fb": diags_array(fb, offsets=n1 - 1, shape=(bins, bins)),
        "fc": diags_array([1], offsets=n1, shape=(bins, bins)),
        "ff": diags_array(ff, offsets=n1 + 1, shape=(bins, bins)),
    }


def derivative_matrix(
    grid_shape: tuple[int, int],
    grid_step: float = 1.0,
    axis: int = 0,
    scheme: Literal["forward", "backward", "central"] = "forward",
    mask: ndarray | None = None,
) -> csc_array:
    """Generate derivative matrix.

    This function computes the derivative matrix for a regular orthogonal coordinate grid.
    The grid points must be equally spaced along the given axis.
    The numerical scheme is based on the finite difference method of forward, backward, or
    central difference.
    The dirichlet boundary condition is applied to the edge of the grid.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Shape of the grid (N0, N1), where N0 and N1 are the number of grid points along the axis 0
        and 1 respectively.
    grid_step : float, optional
        Grid step size along the user-specified axis, by default 1.0.
    axis : int, optional
        Axis along which the derivative is taken. Default is 0.
        Choose from 0 or 1.
    scheme : {"forward", "backward", "central"}, optional
        Scheme of the derivative. Default is "forward".
        Choose from "forward", "backward", or "central".
    mask : ndarray, optional
        Mask array. Default is None.
        If masking a certain grid point, the corresponding row and column is set to `False` in the
        mask array.

    Returns
    -------
    (N, N) :obj:`scipy.sparse.csc_array`
        Derivative Compressed Sparse Column matrix, where N is the number of grid points
        same as ``grid_shape[0] * grid_shape[1]``.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> dmat = derivative_matrix((3, 3), 1, axis=0, scheme="forward")
        >>> dmat.toarray()
        array([[-1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
               [ 0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
               [ 0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.],
               [ 0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.],
               [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.]])

    Notes
    -----
    The detailed explanation of the derivative matrix can be found in the
    `theory of the derivative matrix`_.

    .. _theory of the derivative matrix: ../user/theory/derivative.ipynb
    """
    # validation
    if mask is None:
        pass
    elif isinstance(mask, ndarray):
        if mask.shape != grid_shape:
            raise ValueError("mask shape must be the same as grid shape")
    else:
        raise TypeError("mask must be None or numpy.ndarray")

    n0, n1 = grid_shape

    if n0 < 1 or n1 < 1:
        raise ValueError("element of grid_shape must be positive integer")
    if grid_step <= 0:
        raise ValueError("grid_step must be positive float")

    # Compute derivative matrix
    diag = diag_dict(grid_shape)

    if axis == 0:
        if scheme == "forward":
            dmat = diag["fc"] - diag["cc"]
        elif scheme == "backward":
            dmat = diag["cc"] - diag["bc"]
        elif scheme == "central":
            dmat = (diag["fc"] - diag["bc"]) * 0.5
        else:
            raise ValueError(
                f"Invalid scheme: {scheme}. Choose from 'forward', 'backward', or 'central'."
            )
        dmat /= grid_step

    elif axis == 1:
        if scheme == "forward":
            dmat = diag["cf"] - diag["cc"]
        elif scheme == "backward":
            dmat = diag["cc"] - diag["cb"]
        elif scheme == "central":
            dmat = (diag["cf"] - diag["cb"]) * 0.5
        else:
            raise ValueError(
                f"Invalid scheme: {scheme}. Choose from 'forward', 'backward', or 'central'."
            )
        dmat /= grid_step

    else:
        raise ValueError("axis must be 0 or 1")

    # masking
    if mask is not None:
        mask = mask.flatten()
        dmat = dmat[mask, :][:, mask]

    return dmat.tocsc()


def laplacian_matrix(
    grid_shape: tuple[int, int],
    grid_steps: tuple[float, float] = (1.0, 1.0),
    diagonal: bool = True,
    mask: ndarray | None = None,
) -> csc_array:
    """Generate laplacian matrix.

    This function computes the laplacian matrix for a regular orthogonal coordinate grid.
    The grid points must be equally spaced along the given axis.
    The numerical scheme is based on the finite difference method.
    The dirichlet boundary condition is applied to the edge of the grid.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        Shape of the grid (N0, N1), where N0 and N1 are the number of grid points along the axis 0
        and 1 respectively.
    grid_steps : tuple[double, double], optional
        Step size of the grid (h0, h1), where h0 and h1 are the step size along the axis 0 and 1
        respectively, by default (1.0, 1.0).
    diagonal : bool, optional
        Whether to include the diagonal term or not. Default is True.
    mask : ndarray, optional
        Mask array. Default is None.
        If masking a certain grid point, the corresponding row and column is set to `False` in the
        mask array.

    Returns
    -------
    (N, N) :obj:`scipy.sparse.csc_array`
        Laplacian Compressed Sparse Column matrix, where N is the number of grid points
        same as ``grid_shape[0] * grid_shape[1]``.

    Examples
    --------
    .. prompt:: python >>> auto

        >>> lmat = laplacian_matrix((3, 3), (1, 1), diagonal=False)
        >>> lmat.toarray()
        array([[-4.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
               [ 1., -4.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
               [ 0.,  1., -4.,  0.,  0.,  1.,  0.,  0.,  0.],
               [ 1.,  0.,  0., -4.,  1.,  0.,  1.,  0.,  0.],
               [ 0.,  1.,  0.,  1., -4.,  1.,  0.,  1.,  0.],
               [ 0.,  0.,  1.,  0.,  1., -4.,  0.,  0.,  1.],
               [ 0.,  0.,  0.,  1.,  0.,  0., -4.,  1.,  0.],
               [ 0.,  0.,  0.,  0.,  1.,  0.,  1., -4.,  1.],
               [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  1., -4.]])

        >>> lmat2 = laplacian_matrix((3, 3), (1, 1), diagonal=True)
        >>> lmat2.toarray()
        array([[-6. ,  1. ,  0. ,  1. ,  0.5,  0. ,  0. ,  0. ,  0. ],
               [ 1. , -6. ,  1. ,  0.5,  1. ,  0.5,  0. ,  0. ,  0. ],
               [ 0. ,  1. , -6. ,  0. ,  0.5,  1. ,  0. ,  0. ,  0. ],
               [ 1. ,  0.5,  0. , -6. ,  1. ,  0. ,  1. ,  0.5,  0. ],
               [ 0.5,  1. ,  0.5,  1. , -6. ,  1. ,  0.5,  1. ,  0.5],
               [ 0. ,  0.5,  1. ,  0. ,  1. , -6. ,  0. ,  0.5,  1. ],
               [ 0. ,  0. ,  0. ,  1. ,  0.5,  0. , -6. ,  1. ,  0. ],
               [ 0. ,  0. ,  0. ,  0.5,  1. ,  0.5,  1. , -6. ,  1. ],
               [ 0. ,  0. ,  0. ,  0. ,  0.5,  1. ,  0. ,  1. , -6. ]])

    Notes
    -----
    The detailed explanation of the laplacian matrix can be found in the
    `theory of the laplacian matrix`_.

    .. _theory of the laplacian matrix: ../user/theory/derivative.ipynb
    """
    # validation
    if mask is None:
        pass
    elif isinstance(mask, ndarray):
        if mask.shape != grid_shape:
            raise ValueError("mask shape must be the same as grid shape")
    else:
        raise TypeError("mask must be None or numpy.ndarray")

    n0, n1 = grid_shape
    h0, h1 = grid_steps

    if n0 < 1 or n1 < 1:
        raise ValueError("element of grid_shape must be positive integer")
    if h0 <= 0 or h1 <= 0:
        raise ValueError("element of grid_steps must be positive float")

    # Compute laplacian matrix
    diag = diag_dict(grid_shape)

    lmat = (diag["fc"] - 2 * diag["cc"] + diag["bc"]) / (h0**2) + (
        diag["cf"] - 2 * diag["cc"] + diag["cb"]
    ) / (h1**2)

    if diagonal:
        step = h0**2 + h1**2
        lmat += (diag["ff"] - 2 * diag["cc"] + diag["bb"]) / step + (
            diag["fb"] - 2 * diag["cc"] + diag["bf"]
        ) / step

    # masking
    if mask is not None:
        mask = mask.flatten()
        lmat = lmat[mask, :][:, mask]

    return lmat.tocsc()


class Derivative:
    """Class for derivative matrices.

    This class is used to generate various derivative matrices using grid coordinates information.

    Derivative matrices are applied to spatial profiles defined on the grid coordinates (cartesian,
    cylindrical, etc.) and are created with their grid coordinates.

    Parameters
    ----------
    grid : (..., L, M, ndim) or (N, ), array_like
        The grid coordinates.
        The shape of the array means the resolution of the grid. For example, in 3-D spatial grid,
        the shape is (L, M, N, 3), where 3 is the dimension of the grid.
        If the grid is 1-D like (N,), it is automatically reshaped to (N, 1).
    grid_map : (..., L, M) array_like, optional
        The map of the grid index with `int` type.
        The shape of the array must be the same as the grid shape except for the last dimension.
        The masked area is represented by negative values.
        If None, the grid_map is automatically created by the order of the grid array.

    Examples
    --------
    >>> import numpy as np
    >>> from cherab.inversion import Derivative
    >>> x = np.linspace(-10, 10, 21)
    >>> derivative = Derivative(x)
    >>> derivative
    Derivative(grid=array(21, 1), grid_map=array(21,))
    """

    def __init__(
        self,
        grid,
        grid_map=None,
    ) -> None:
        # set properties
        self._grid_setter(grid)
        self._grid_map_setter(grid_map)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(grid=array{self.grid.shape}, grid_map=array{self.grid_map.shape})"
        )

    @property
    def grid(self) -> ndarray:
        """The grid coordinates."""
        return self._grid

    def _grid_setter(self, array) -> None:
        array = np.asarray_chkfinite(array)
        if array.ndim < 1:
            raise ValueError("grid must be at least 1-D.")
        if array.ndim < 2:
            array = array[:, None]  # add the last dimension
        self._grid = array

    @property
    def grid_map(self) -> ndarray:
        """The map of the grid index."""
        return self._grid_map

    def _grid_map_setter(self, array) -> None:
        if array is None:
            array = np.arange(np.prod(self._grid.shape[:-1]), dtype=np.int32).reshape(
                self._grid.shape[:-1]
            )

        array = np.asarray_chkfinite(array)

        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"grid_map must be integer type, not {array.dtype}.")
        if array.ndim != self._grid.ndim - 1:
            raise ValueError(f"grid_map must be {self._grid.ndim - 1}D, not {array.ndim}.")
        if array.shape != self._grid.shape[:-1]:
            raise ValueError(f"grid_map shape must be {self._grid.shape[:-1]}, not {array.shape}.")
        self._grid_map = array

    def matrix_along_axis(
        self,
        axis: int,
        diff_type: Literal["forward", "backward"] = "forward",
        boundary: Literal["dirichlet", "neumann", "periodic"] = "dirichlet",
    ) -> csr_array:
        """Derivative matrix along the specified axis.

        .. note::

            If the grid has masked area, edges before the masked area take either the dirichlet
            (forward) or neumann (backward) condition.
            The periodic boundary condition is only available for the last and first points
            along the axis.

        Parameters
        ----------
        axis : int
            The axis to calculate the derivative matrix.
            The axis must be in the grid dimensions.
        diff_type : {"forward", "backward"}, optional
            The type of the difference method, by default "forward".
        boundary : {"dirichlet", "neumann", "periodic"}, optional
            The boundary condition of the derivative matrix, by default "dirichlet".
            If the boundary condition is "dirichlet", the outer boundary is set to zero.
            If the boundary condition is "neumann", the difference type is set to the opposite
            direction at the edge of the grid.
            If the boundary condition is "periodic", the first and last points are connected.

        Returns
        -------
        csr_array
            The derivative matrix along the specified axis.

        Examples
        --------
        Firstly, let us operate the derivative matrix to a simple 1-D profile and compare it with
        gradient values calculated by the `numpy.gradient` function.

        .. plot::
            :include-source:

            import numpy as np
            import matplotlib.pyplot as plt
            from cherab.inversion import Derivative

            # Create a sample 1-D sine profile
            x = np.linspace(0, 2 * np.pi, 100, endpoint=False)
            y = np.sin(x)

            # Calculate the derivative matrix
            derivative = Derivative(x)
            dmat = derivative.matrix_along_axis(0, boundary="dirichlet")

            # Compare gradient values
            plt.plot(x, y, label="y = sin(x)")
            plt.plot(x, dmat @ y, label="y' = D @ y")
            plt.plot(x, np.gradient(y, x), label="y' = np.gradient(y, x)", linestyle="--")
            plt.legend()
            plt.show()

        Next, we create a sample 2-D profile in cylindrical coordinates and apply the derivative
        along the R and Phi axis to the profile.

        .. plot:: ../scripts/derivative_cylindrical.py
            :include-source:
        """
        if axis < 0 or axis >= self._grid_map.ndim:
            raise ValueError(f"Invalid axis: {axis}.")

        if diff_type == "forward":
            sign = 1.0
            pair = (0, 1)
            edge = -1
        elif diff_type == "backward":
            sign = -1.0
            pair = (1, 0)
            edge = 0
        else:
            raise ValueError(f"Invalid diff_type: {diff_type}.")

        if boundary not in {"dirichlet", "neumann", "periodic"}:
            raise ValueError(f"Invalid boundary: {boundary}.")

        bins = self._grid_map.max() + 1
        mat = lil_array((bins, bins), dtype=np.float64)

        for fixed_indices in np.ndindex(
            *[dim for i, dim in enumerate(self._grid_map.shape) if i != axis]
        ):
            # slices for the fixed indices
            slc1: list[int | slice] = list(fixed_indices)
            slc2: list[int | slice] = list(fixed_indices)

            slc1.insert(axis, slice(1, None))
            slc2.insert(axis, slice(None, -1))

            # calculate each segment length along the axis
            lengths = np.linalg.norm(self._grid[tuple(slc1)] - self._grid[tuple(slc2)], axis=-1)

            # extract the indices of the fixed axis
            slc: list[int | slice] = list(fixed_indices)
            slc.insert(axis, slice(0, None))
            indices = self._grid_map[tuple(slc)]

            for k, (i, j) in enumerate(zip(indices[pair[0] :], indices[pair[1] :])):  # noqa: B905
                # skip the masked area
                if i < 0:
                    continue

                mat[i, i] = -sign / lengths[k]
                mat[i, j] = sign / lengths[k]

            # Boundary condition
            i = indices[edge]

            # skip the masked area
            if i < 0:
                continue

            elif boundary == "dirichlet":
                mat[i, i] = -sign / lengths[edge]

            elif boundary == "neumann":
                j = indices[edge - int(sign)]
                mat[i, i] = sign / lengths[edge]
                mat[i, j] = -sign / lengths[edge]

            elif boundary == "periodic":
                j = indices[edge + int(sign)]
                length = np.linalg.norm(
                    self._grid[tuple(slc)][0, :] - self._grid[tuple(slc)][-1, :]
                )
                mat[i, i] = -sign / length
                mat[i, j] = sign / length

        return mat.tocsr()

    def matrix_gradient(
        self, func: Callable[[float, float], float], diagonal: bool = False
    ) -> tuple[csr_array, csr_array]:
        """Derivative matrices based on the gradient of the given function.

        This function returns derivative matrices along the parallel and perpendicular directions
        to the gradient of the given function.
        The gradient is calculated by the finite central difference using `numpy.gradient` at
        each grid point.

        .. warning::
            Currently, this method can only be used for a 2-D function, which uses the first and
            second axis of the grid coordinates, i.e. the grid shape is only allowed to be
            `(L, M, ndim)` or `(L, M, N, ndim)`. and the function use the `L` and `M` as the
            first and second axis respectively.

        Parameters
        ----------
        func : Callable[[float, float], float]
            The scalar function to calculate the gradient.
            The function uses the first and second coordinates of the grid as the input.
        diagonal : bool, optional
            Whether to include the diagonal difference along to the gradient, by default False.

        Returns
        -------
        dmat_parallel : csr_array
            The derivative matrix along the parallel direction to the gradient.
        dmat_perpendicular : csr_array
            The derivative matrix along the perpendicular direction to the gradient.

        Examples
        --------
        Let us create a sample 2-D profile and apply the derivative matrices along the parallel
        and perpendicular directions to the gradient of the concentrically monotonically increasing
        function.

        .. plot:: ../scripts/derivative_gradient.py
            :include-source:
        """
        # Check the grid shape and convert to (L, M, N) array
        num_axis = self._grid_map.ndim
        if num_axis < 2:
            raise ValueError(f"The grid must be at least 2-D ({num_axis - 1}-D).")
        elif num_axis == 2:
            grid_map = self._grid_map[:, :, np.newaxis]
        elif num_axis == 3:
            grid_map = self._grid_map
        else:
            raise NotImplementedError("The grid over 4-D is not supported yet.")

        # Check the grid coordinate dimension and convert to (L, M, N, 2) array
        grid_dim = self._grid.shape[-1]
        if grid_dim < 2:
            raise ValueError(f"The grid coordinate must be at least 2-D ({grid_dim}-D).")
        if num_axis == 2:
            grid = self._grid[..., np.newaxis, :2]
        elif num_axis == 3:
            grid = self._grid[..., :2]
        else:
            raise NotImplementedError("The grid coordinate over 4-D is not supported yet.")

        # Calculate the gradient of the function along axis 0 and 1
        func_values = np.zeros((grid.shape[0], grid.shape[1]))
        for i, j in np.ndindex(*grid.shape[:2]):
            func_values[i, j] = func(grid[i, j, 0, 0], grid[i, j, 0, 1])

        # Calculate the gradient
        grads_0, grads_1 = np.gradient(func_values, grid[:, 0, 0, 0], grid[0, :, 0, 1])

        # Prepare the derivative matrices
        bins = grid_map.max() + 1
        dmat_para = lil_array((bins, bins), dtype=np.float64)
        dmat_perp = lil_array((bins, bins), dtype=np.float64)

        L = grid_map.shape[0]
        M = grid_map.shape[1]

        for indices in np.ndindex(grid_map.shape):
            index = grid_map[indices]

            # skip the masked area
            if index < 0:
                continue

            # Extract axes indices
            i, j, k = indices

            # Calculate the gradient values
            g0, g1 = grads_0[i, j], grads_1[i, j]
            g0_abs, g1_abs = np.abs(g0), np.abs(g1)
            g0_sign, g1_sign = int(np.sign(g0)), int(np.sign(g1))

            # Parallel direction
            forward_para_index0 = i + g0_sign
            forward_para_index1 = j + g1_sign

            dmat_para[index, index] += -g0_abs - g1_abs
            if forward_para_index0 > -1 and forward_para_index0 < L:
                dmat_para[index, grid_map[forward_para_index0, j, k]] += g0_abs

            if forward_para_index1 > -1 and forward_para_index1 < M:
                dmat_para[index, grid_map[i, forward_para_index1, k]] += g1_abs

            # Perpendicular direction
            forward_perp_index0 = i - g1_sign
            forward_perp_index1 = j + g0_sign

            dmat_perp[index, index] += -g1_abs - g0_abs
            if forward_perp_index0 > -1 and forward_perp_index0 < L:
                dmat_perp[index, grid_map[forward_perp_index0, j, k]] += g1_abs

            if forward_perp_index1 > -1 and forward_perp_index1 < M:
                dmat_perp[index, grid_map[i, forward_perp_index1, k]] += g0_abs

            # Add the diagonal term
            if diagonal:
                value = g0_abs * g1_abs / np.hypot(g0, g1)

                dmat_para[index, index] += -value
                dmat_perp[index, index] += -value
                if (
                    forward_para_index0 > -1
                    and forward_para_index0 < L
                    and forward_para_index1 > -1
                    and forward_para_index1 < M
                ):
                    diag_para_index = grid_map[forward_para_index0, forward_para_index1, k]
                    dmat_para[index, diag_para_index] += value
                if (
                    forward_perp_index0 > -1
                    and forward_perp_index0 < L
                    and forward_perp_index1 > -1
                    and forward_perp_index1 < M
                ):
                    diag_perp_index = grid_map[forward_perp_index0, forward_perp_index1, k]
                    dmat_perp[index, diag_perp_index] += value

        return dmat_para.tocsr(), dmat_perp.tocsr()
