"""Module to offer the function to generate a derivative matrix."""

from typing import Literal

import numpy as np
from numpy import ndarray
from scipy.sparse import csc_array, dia_array, diags_array

__all__ = ["diag_dict", "derivative_matrix", "laplacian_matrix"]


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
