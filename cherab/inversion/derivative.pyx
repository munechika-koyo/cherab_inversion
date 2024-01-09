"""Module to offer the function to generate a derivative matrix."""
import warnings

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, diags

cimport cython
from numpy cimport import_array, ndarray

__all__ = ["compute_dmat", "diag_dict", "derivative_matrix", "laplacian_matrix"]


import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef object compute_dmat(
    ndarray voxel_map,
    str kernel_type="laplacian4",
    ndarray kernel=None,
):
    """Generate derivative sparse matrix.

    .. warning::

        This function will be deprecated in the future.
        Please use `.derivative_matrix` or `.laplacian_matrix` instead.


    Parameters
    ----------
    voxel_map : numpy.ndarray
        (N, M) voxel map matrix (negative value must be input into masked voxels)
        If the additional dimension size of the matrix is 1, then it is squeezed to a 2-D matrix.
    kernel_type : {"x", "y", "r", "z", "laplacian4", "laplacian8", "custom"}, optional
        Derivative kernel type. Default is "laplacian8".
        `"custom"` is available only when `.kernel` is specified.
        "r" and "z" are the same as "y" and "x", respectively.
    kernel : numpy.ndarray, optional
        (3, 3) custom kernel matrix. Default is None.

    Returns
    -------
    :obj:`scipy.sparse.csc_matrix`
        (N, N) derivative Compressed Sparse Column matrix (if N > M)

    Notes
    -----
    The derivative matrix is generated by the kernel convolution method.
    The kernel is a 3x3 matrix, and the convolution is performed as follows:

    .. math::

        I_{x, y}' = \\sum_{i=-1}^{1}\\sum_{j=-1}^{1} K_{i,j} \\times I_{x + i, y + j},

    where :math:`I_{x, y}` is the 2-D image at the point :math:`(x, y)` and :math:`K_{i,j}` is the
    kernel matrix.
    Using derivative kernel like a laplacian filter, the derivative matrix defined as follows is
    generated:

    .. math::

        \\mathbf{I}' = \\mathbf{K} \\cdot \\mathbf{I},

    where :math:`\\mathbf{I}` is the vecotrized image and :math:`\\mathbf{K}` is the derivative
    matrix.

    The implemented derivative kernels are as follows:

    - First derivative in x-direction (`kernel_type="x"` or `"z"`):
      :math:`\\mathbf{K} = \\begin{bmatrix} 0 & 0 & 0 \\\\ -1 & 1 & 0 \\\\ 0 & 0 & 0 \\end{bmatrix}`
    - First derivative in y-direction (`kernel_type="y"` or `"r"`):
      :math:`\\mathbf{K} = \\begin{bmatrix} 0 & -1 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 0 \\end{bmatrix}`
    - Laplacian-4 (`kernel_type="laplacian4"`):
      :math:`\\mathbf{K} = \\begin{bmatrix} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{bmatrix}`
    - Laplacian-8 (`kernel_type="laplacian8"`):
      :math:`\\mathbf{K} = \\begin{bmatrix} 1 & 1 & 1 \\\\ 1 & -8 & 1 \\\\ 1 & 1 & 1 \\end{bmatrix}`


    Examples
    --------
    .. prompt:: python >>> auto

        >>> from raysect.optical import World
        >>> from cherab.phix.tools.raytransfer import import_phix_rtc
        >>> from cherab.inversion.derivative import compute_dmat
        >>>
        >>> world = World()
        >>> rtc = import_phix_rtc(world)
        >>>
        >>> laplacian = compute_dmat(rtc.voxel_map)
        >>> laplacian.toarray()
        array([[-8.,  1.,  0., ...,  0.,  0.,  0.],
               [ 1., -8.,  1., ...,  0.,  0.,  0.],
               [ 0.,  1., -8., ...,  0.,  0.,  0.],
               ...,
               [ 0.,  0.,  0., ..., -8.,  1.,  0.],
               [ 0.,  0.,  0., ...,  1., -8.,  1.],
               [ 0.,  0.,  0., ...,  0.,  1., -8.]])
    """
    cdef:
        int i, j, x, y, row, col, map_mat_max
        double[3][3] kernel_carray
        ndarray map_matrix
        object dmatrix
        double[:, ::] kernel_mv
        int[:, ::] map_matrix_mv

    warnings.warn(
        "This function will be deprecated in the future. "
        "Please use `.derivative_matrix` or `.laplacian_matrix` instead.",
        DeprecationWarning,
    )

    # define derivative kernel
    if kernel_type in {"z", "x"}:
        kernel_carray[0][:] = [0, 0, 0]
        kernel_carray[1][:] = [-1, 1, 0]
        kernel_carray[2][:] = [0, 0, 0]
        kernel_mv = kernel_carray

    elif kernel_type in {"r", "y"}:
        kernel_carray[0][:] = [0, -1, 0]
        kernel_carray[1][:] = [0, 1, 0]
        kernel_carray[2][:] = [0, 0, 0]
        kernel_mv = kernel_carray

    elif kernel_type == "laplacian4":
        kernel_carray[0][:] = [0, 1, 0]
        kernel_carray[1][:] = [1, -4, 1]
        kernel_carray[2][:] = [0, 1, 0]
        kernel_mv = kernel_carray

    elif kernel_type == "laplacian8":
        kernel_carray[0][:] = [1, 1, 1]
        kernel_carray[1][:] = [1, -8, 1]
        kernel_carray[2][:] = [1, 1, 1]
        kernel_mv = kernel_carray

    elif kernel_type == "custom":
        if kernel is None:
            raise ValueError("kernel must be specified when kernel_type is 'custom'")
        else:
            if kernel.ndim != 2:
                raise ValueError("kernel must be 2-D matrix")

            elif kernel.shape[0] != 3 or kernel.shape[1] != 3:
                raise ValueError("kernel must be 3x3 matrix")

            else:
                kernel_mv = kernel.astype(float)

    else:
        raise ValueError(
            "kernel must be 'x', 'y', 'r', 'z', 'laplacian4', 'laplacian8' or 'custom'"
        )

    # padding voxel map boundary by -1
    voxel_map = np.squeeze(voxel_map)
    if voxel_map.ndim == 2:
        map_matrix = np.pad(np.squeeze(voxel_map), pad_width=1, constant_values=-1)
        map_mat_max = map_matrix.max()
    else:
        raise ValueError("voxel_map must be 2-D matrix")

    # define derivative matrix as a sparse matrix
    dmatrix = lil_matrix((map_mat_max + 1, map_mat_max + 1), dtype=float)

    # define memoryview
    map_matrix_mv = map_matrix.astype(np.intc)

    # generate derivative matrix
    for row in range(map_mat_max + 1):
        (x,), (y,) = np.where(map_matrix == row)  # TODO: replace to cythonic codes
        for i in range(-1, 1 + 1):
            for j in range(-1, 1 + 1):
                col = map_matrix_mv[x + i, y + j]
                if col > -1:
                    dmatrix[row, col] = kernel_mv[i + 1, j + 1]
                else:
                    pass

    return dmatrix.tocsc()


cpdef dict diag_dict((int, int) grid_shape):
    """Return a dictionary of diagonal matrices.

    The key of the dictionary corresponds to the position of grid points.
    `b`, `c`, and `f` mean backward, center, and forward along the given axis respectively.
    e.g. `bf` means the backward and forward grid points along the axis 0 and 1 respectively.

    The following figure shows how the each grid point is connected.

    .. code-block:: none

        bb ─── bc ─── bf    --> axis 1
        │       │        │
        │       │        │
        cb ─── cc ─── cf
        │       │        │
        │       │        │
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
        shape of the grid (N0, N1), where N0 and N1 are the number of grid points along the axis 0
        and 1 respectively.

    Returns
    -------
    dict[str, `scipy.sparse.csc_matrix`]
        dictionary of diagonal matrices, the keys of which are ``"bb"``, ``"bc"``, ``"bf"``,
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
    cdef:
        int n0, n1, bins
        object bb, bf, cb, cf, fb, ff

    n0, n1 = grid_shape
    bins = n0 * n1

    # list of each grid position index including the dirichlet boundary condition
    cb = cf = np.tile([1] * (n1 - 1) + [0], n0)[:bins - 1]
    fb = bf = np.tile([0] + [1] * (n1 - 1), n0)[:bins - n1 + 1]
    ff = bb = np.tile([1] * (n1 - 1) + [0], n0)[:bins - n1 - 1]

    return {
        "bb": diags(bb, -n1 - 1, shape=(bins, bins), format="csc"),
        "bc": diags([1], -n1, shape=(bins, bins), format="csc"),
        "bf": diags(bf, -n1 + 1, shape=(bins, bins), format="csc"),
        "cb": diags(cb, -1, shape=(bins, bins), format="csc"),
        "cc": diags([1], 0, shape=(bins, bins), format="csc"),
        "cf": diags(cf, 1, shape=(bins, bins), format="csc"),
        "fb": diags(fb, n1 - 1, shape=(bins, bins), format="csc"),
        "fc": diags([1], n1, shape=(bins, bins), format="csc"),
        "ff": diags(ff, n1 + 1, shape=(bins, bins), format="csc"),
    }

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef object derivative_matrix(
    (int, int) grid_shape,
    float grid_step = 1.0,
    int axis = 0,
    str scheme = "forward",
    object mask = None,
):
    """Generate derivative matrix.

    This compute the derivative matrix for a regular orthogonal coordinate grid.
    The numerical scheme is based on the finite difference method of forward, backward, or
    central difference.
    The dirichlet boundary condition is applied to the edge of the grid.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        shape of the grid (N0, N1), where N0 and N1 are the number of grid points along the axis 0
        and 1 respectively.
    grid_step : float, optional
        grid step size along the user-specified axis, by default 1.0.
    axis : int, optional
        axis along which the derivative is taken. Default is 0.
        Choose from 0 or 1.
    scheme : {"forward", "backward", "central"}, optional
        scheme of the derivative. Default is "forward".
        Choose from "forward", "backward", or "central".
    mask : numpy.ndarray, optional
        mask array. Default is None.
        If masking a certain grid point, the corresponding row and column is set to False in the
        mask array.

    Returns
    -------
    :obj:`scipy.sparse.csc_matrix`
        (N, N) derivative Compressed Sparse Column matrix, where N is the number of grid points
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
    """

    cdef:
        int n0, n1
        dict[str, csc_matrix] diag
        object dmat

    # validation
    if mask is None:
        pass
    elif isinstance(mask, np.ndarray):
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

    return dmat

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef object laplacian_matrix(
    (int, int) grid_shape,
    (double, double) grid_step = (1.0, 1.0),
    bint diagonal = True,
    object mask = None,
):
    """Generate laplacian matrix.

    This compute the laplacian matrix for a regular orthogonal coordinate grid.
    The numerical scheme is based on the finite difference method.
    The dirichlet boundary condition is applied to the edge of the grid.

    Parameters
    ----------
    grid_shape : tuple[int, int]
        shape of the grid (N0, N1), where N0 and N1 are the number of grid points along the axis 0
        and 1 respectively.
    grid_step : tuple[double, double], optional
        step size of the grid (h0, h1), where h0 and h1 are the step size along the axis 0 and 1
        respectively, by default (1.0, 1.0)
    diagonal : bool, optional
        whether to include the diagonal term or not. Default is True.
    mask : numpy.ndarray, optional
        mask array. Default is None.
        If masking a certain grid point, the corresponding row and column is set to False in the
        mask array.

    Returns
    -------
    :obj:`scipy.sparse.csc_matrix`
        (N, N) laplacian Compressed Sparse Column matrix, where N is the number of grid points
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
    """

    cdef:
        int n0, n1
        double h0, h1, step
        dict[str, csc_matrix] diag
        object lmat

    # validation
    if mask is None:
        pass
    elif isinstance(mask, np.ndarray):
        if mask.shape != grid_shape:
            raise ValueError("mask shape must be the same as grid shape")
    else:
        raise TypeError("mask must be None or numpy.ndarray")

    n0, n1 = grid_shape
    h0, h1 = grid_step

    if n0 < 1 or n1 < 1:
        raise ValueError("element of grid_shape must be positive integer")
    if h0 <= 0 or h1 <= 0:
        raise ValueError("element of grid_step must be positive float")

    # Compute laplacian matrix
    diag = diag_dict(grid_shape)

    lmat = (diag["fc"] - 2 * diag["cc"] + diag["bc"]) / (h0 ** 2) \
         + (diag["cf"] - 2 * diag["cc"] + diag["cb"]) / (h1 ** 2)

    if diagonal:
        step = h0 ** 2 + h1 ** 2
        lmat += (diag["ff"] - 2 * diag["cc"] + diag["bb"]) / step \
              + (diag["fb"] - 2 * diag["cc"] + diag["bf"]) / step

    # masking
    if mask is not None:
        mask = mask.flatten()
        lmat = lmat[mask, :][:, mask]

    return lmat
