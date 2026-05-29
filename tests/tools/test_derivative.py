# ruff: noqa: N802
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from cherab.inversion.tools._derivative import (
    Derivative,
    derivative_matrix,
    diag_dict,
    laplacian_matrix,
)

####################################################################################################
# Test for diag_dict function
# ---------------------------
DIAG_CASE = {
    "2x2 grid": (
        (2, 2),
        np.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        ),
        does_not_raise(),
    ),
    "3x3 grid": (
        (3, 3),
        np.array(
            [
                [1, 1, 0, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 0, 1, 1],
            ]
        ),
        does_not_raise(),
    ),
}


@pytest.mark.parametrize(
    ["grid_shape", "expected", "expectation"],
    [pytest.param(*case, id=key) for key, case in DIAG_CASE.items()],
)
def test_diag_dict(grid_shape, expected, expectation):
    with expectation:
        diag = diag_dict(grid_shape)
        diag_mat = sum(diag.values())
        np.testing.assert_array_equal(diag_mat.toarray(), expected)


####################################################################################################
# Test for derivative_matrix function
# -----------------------------------

DMAT_CASE = {
    "2x2 grid, 1 steps, 0 axis, forward scheme, no mask": (
        (2, 2),
        1,
        0,
        "forward",
        None,
        np.array(
            [
                [-1, 0, 1, 0],
                [0, -1, 0, 1],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]
        ),
        does_not_raise(),
    ),
    "2x2 grid, 1 steps, 0 axis, backward scheme, no mask": (
        (2, 2),
        1,
        0,
        "backward",
        None,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [-1, 0, 1, 0],
                [0, -1, 0, 1],
            ]
        ),
        does_not_raise(),
    ),
    "3x3 grid, 1 steps, 0 axis, central scheme, no mask": (
        (3, 3),
        1,
        0,
        "central",
        None,
        np.array(
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, -1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, -1, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0, 0],
            ]
        )
        * 0.5,
        does_not_raise(),
    ),
    "2x2 grid, 1 steps, 1 axis, forward scheme, no mask": (
        (2, 2),
        1,
        1,
        "forward",
        None,
        np.array(
            [
                [-1, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, -1],
            ]
        ),
        does_not_raise(),
    ),
    "2x2 grid, 1 steps, 1 axis, backward scheme, no mask": (
        (2, 2),
        1,
        1,
        "backward",
        None,
        np.array(
            [
                [1, 0, 0, 0],
                [-1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 1],
            ]
        ),
        does_not_raise(),
    ),
    "3x3 grid, 1 steps, 1 axis, central scheme, no mask": (
        (3, 3),
        1,
        1,
        "central",
        None,
        np.array(
            [
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, -1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, -1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, -1, 0],
            ]
        )
        * 0.5,
        does_not_raise(),
    ),
    "2x2 grid, 0.5 steps, 1 axis, forward scheme, no mask": (
        (2, 2),
        0.5,
        1,
        "forward",
        None,
        np.array(
            [
                [-1, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, -1],
            ]
        )
        * 2,
        does_not_raise(),
    ),
    "3x3 grid, 1 steps, 1 axis, forward scheme, (2, 0) and (2, 2) mask": (
        (3, 3),
        1,
        1,
        "forward",
        np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [-1, 1, 0, 0, 0, 0, 0],
                [0, -1, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0],
                [0, 0, 0, -1, 1, 0, 0],
                [0, 0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 0, 0, -1],
            ]
        ),
        does_not_raise(),
    ),
    "invalid grid shape": (
        (2, -1),
        1,
        0,
        "forward",
        None,
        None,
        pytest.raises(ValueError),
    ),
    "invalid grid step": (
        (2, 2),
        -1,
        0,
        "forward",
        None,
        None,
        pytest.raises(ValueError),
    ),
    "invalid axis": (
        (2, 2),
        1,
        2,
        "forward",
        None,
        None,
        pytest.raises(ValueError),
    ),
    "invalid scheme": (
        (2, 2),
        1,
        0,
        "invalid",
        None,
        None,
        pytest.raises(ValueError),
    ),
    "invalid mask shape": (
        (2, 2),
        1,
        0,
        "forward",
        np.zeros((3, 3), dtype=bool),
        None,
        pytest.raises(ValueError),
    ),
    "invalid mask type": (
        (2, 2),
        1,
        0,
        "forward",
        "invalid",
        None,
        pytest.raises(TypeError),
    ),
    "axis 1, invalid scheme": (
        (2, 2),
        1,
        1,
        "invalid",
        None,
        None,
        pytest.raises(ValueError),
    ),
}


@pytest.mark.parametrize(
    ["grid_shape", "grid_step", "axis", "scheme", "mask", "expected_dmat", "expectation"],
    [pytest.param(*case, id=key) for key, case in DMAT_CASE.items()],
)
def test_derivative_matrix(grid_shape, grid_step, axis, scheme, mask, expected_dmat, expectation):
    with expectation:
        dmat = derivative_matrix(grid_shape, grid_step, axis, scheme, mask)
        np.testing.assert_array_equal(dmat.toarray(), expected_dmat)


####################################################################################################
# Test for laplacian_matrix function
# ----------------------------------

LMAT_CASE = {
    "2x2 grid, 1x1 steps, no diagonal, no mask": (
        (2, 2),
        (1, 1),
        False,
        None,
        np.array([[-4, 1, 1, 0], [1, -4, 0, 1], [1, 0, -4, 1], [0, 1, 1, -4]]),
        does_not_raise(),
    ),
    "3x3 grid, 1x1 steps, no diagonal, no mask": (
        (3, 3),
        (1, 1),
        False,
        None,
        np.array(
            [
                [-4, 1, 0, 1, 0, 0, 0, 0, 0],
                [1, -4, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, -4, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, -4, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, -4, 1, 0, 1, 0],
                [0, 0, 1, 0, 1, -4, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, -4, 1, 0],
                [0, 0, 0, 0, 1, 0, 1, -4, 1],
                [0, 0, 0, 0, 0, 1, 0, 1, -4],
            ]
        ),
        does_not_raise(),
    ),
    "3x3 grid, 1x1 steps, diagonal, no mask": (
        (3, 3),
        (1, 1),
        True,
        None,
        np.array(
            [
                [-6, 1, 0, 1, 0.5, 0, 0, 0, 0],
                [1, -6, 1, 0.5, 1, 0.5, 0, 0, 0],
                [0, 1, -6, 0, 0.5, 1, 0, 0, 0],
                [1, 0.5, 0, -6, 1, 0, 1, 0.5, 0],
                [0.5, 1, 0.5, 1, -6, 1, 0.5, 1, 0.5],
                [0, 0.5, 1, 0, 1, -6, 0, 0.5, 1],
                [0, 0, 0, 1, 0.5, 0, -6, 1, 0],
                [0, 0, 0, 0.5, 1, 0.5, 1, -6, 1],
                [0, 0, 0, 0, 0.5, 1, 0, 1, -6],
            ]
        ),
        does_not_raise(),
    ),
    "3x3 grid, 0.5x0.25 steps, diagonal, no mask": (
        (3, 3),
        (0.5, 0.25),
        True,
        None,
        np.array(
            [
                [-52.8, 16, 0, 4, 3.2, 0, 0, 0, 0],
                [16, -52.8, 16, 3.2, 4, 3.2, 0, 0, 0],
                [0, 16, -52.8, 0, 3.2, 4, 0, 0, 0],
                [4, 3.2, 0, -52.8, 16, 0, 4, 3.2, 0],
                [3.2, 4, 3.2, 16, -52.8, 16, 3.2, 4, 3.2],
                [0, 3.2, 4, 0, 16, -52.8, 0, 3.2, 4],
                [0, 0, 0, 4, 3.2, 0, -52.8, 16, 0],
                [0, 0, 0, 3.2, 4, 3.2, 16, -52.8, 16],
                [0, 0, 0, 0, 3.2, 4, 0, 16, -52.8],
            ]
        ),
        does_not_raise(),
    ),
    "3x3 grid, 1x1 steps, no diagonal, (2, 0) and (2, 2) mask": (
        (3, 3),
        (1, 1),
        False,
        np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [-4, 1, 0, 1, 0, 0, 0],
                [1, -4, 1, 0, 1, 0, 0],
                [0, 1, -4, 0, 0, 1, 0],
                [1, 0, 0, -4, 1, 0, 0],
                [0, 1, 0, 1, -4, 1, 1],
                [0, 0, 1, 0, 1, -4, 0],
                [0, 0, 0, 0, 1, 0, -4],
            ]
        ),
        does_not_raise(),
    ),
    "3x3 grid, 1x1 steps, diagonal, (2, 0) and (2, 2) mask": (
        (3, 3),
        (1, 1),
        True,
        np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [-6, 1, 0, 1, 0.5, 0, 0],
                [1, -6, 1, 0.5, 1, 0.5, 0],
                [0, 1, -6, 0, 0.5, 1, 0],
                [1, 0.5, 0, -6, 1, 0, 0.5],
                [0.5, 1, 0.5, 1, -6, 1, 1],
                [0, 0.5, 1, 0, 1, -6, 0.5],
                [0, 0, 0, 0.5, 1, 0.5, -6],
            ]
        ),
        does_not_raise(),
    ),
    "invalid grid shape": (
        (2, -1),
        (1, 1),
        False,
        None,
        None,
        pytest.raises(ValueError),
    ),
    "invalid grid steps": (
        (2, 2),
        (1, -1),
        False,
        None,
        None,
        pytest.raises(ValueError),
    ),
    "invalid mask shape": (
        (2, 2),
        (1, 1),
        False,
        np.zeros((3, 3), dtype=bool),
        None,
        pytest.raises(ValueError),
    ),
    "invalid mask type": (
        (2, 2),
        (1, 1),
        False,
        "invalid",
        None,
        pytest.raises(TypeError),
    ),
}


@pytest.mark.parametrize(
    ["grid_shape", "grid_steps", "diagonal", "mask", "expected_lmat", "expectation"],
    [pytest.param(*case, id=key) for key, case in LMAT_CASE.items()],
)
def test_laplacian_matrix(grid_shape, grid_steps, diagonal, mask, expected_lmat, expectation):
    with expectation:
        lmat = laplacian_matrix(grid_shape, grid_steps, diagonal, mask)
        np.testing.assert_array_equal(lmat.toarray(), expected_lmat)


####################################################################################################
# Test for Derivative class
# -------------------------
# Test for __init__ method to change grid
@pytest.mark.parametrize(
    ["setting_grid", "expected_grid", "expected_grid_map", "expectation"],
    [
        pytest.param(
            [1, 2, 3],
            np.array([1, 2, 3])[:, None],
            np.array([0, 1, 2], dtype=np.int32),
            does_not_raise(),
            id="valid 1-D grid setting",
        ),
        pytest.param(
            [[0, 0], [1, 1], [2, 2]],
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([0, 1, 2], dtype=np.int32),
            does_not_raise(),
            id="valid 2-D grid setting",
        ),
        pytest.param(
            [[0, 0], [1, 1], [2, 2]],
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([0, 1, 2], dtype=np.int32),
            does_not_raise(),
            id="valid 2-D grid setting",
        ),
        pytest.param(
            "invalid grid",
            None,
            None,
            pytest.raises(ValueError),
            id="invalid grid setting",
        ),
    ],
)
def test_derivative_set_grid(setting_grid, expected_grid, expected_grid_map, expectation):
    with expectation:
        derivative_instance = Derivative(setting_grid)
        np.testing.assert_array_equal(derivative_instance.grid, expected_grid)
        np.testing.assert_array_equal(derivative_instance.grid_map, expected_grid_map)


# Test for __init__ method to change grid_map
@pytest.mark.parametrize(
    ["grid", "setting_grid_map", "expected_grid_map", "expectation"],
    [
        pytest.param(
            [1, 2, 3],
            [0, 1, 2],
            np.array([0, 1, 2], dtype=np.int32),
            does_not_raise(),
            id="valid 1-D grid mapping",
        ),
        pytest.param(
            [[[0, 0], [1, 0]], [[0, 1], [1, 1]]],
            [[0, 1], [2, 3]],
            np.arange(4, dtype=np.int32).reshape(2, 2),
            does_not_raise(),
            id="valid 2-D grid mapping",
        ),
        pytest.param(
            [[[0, 0], [1, 0]], [[0, 1], [1, 1]]],
            [0, 1, 2, 3],
            None,
            pytest.raises(ValueError),
            id="invalid 2-D grid mapping (invalid shape)",
        ),
        pytest.param(
            [1, 2, 3],
            np.array([0, 1, 2], dtype=float),
            None,
            pytest.raises(TypeError),
            id="invalid 1-D grid mapping (invalid dtype)",
        ),
    ],
)
def test_derivative_set_grid_map(grid, setting_grid_map, expected_grid_map, expectation):
    with expectation:
        derivative_instance = Derivative(grid, setting_grid_map)
        np.testing.assert_array_equal(derivative_instance.grid_map, expected_grid_map)


# Create a fixture for Derivative class
@pytest.fixture
def derivative_instance():
    return Derivative([[[0, 0], [0, 1]], [[1, 0], [1, 1]]], [[0, 2], [1, 3]])


# Test for matrix_along_axis method
MATRIX_ALONG_AXIS_CASES = {
    "axis: 0, default params": (
        0,
        "dirichlet",
        "forward",
        np.array(
            [
                [-1, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, 0, -1],
            ]
        ),
        does_not_raise(),
    ),
    "axis: 0, boundary: 'neumann'": (
        0,
        "neumann",
        "forward",
        np.array(
            [
                [-1, 1, 0, 0],
                [-1, 1, 0, 0],
                [0, 0, -1, 1],
                [0, 0, -1, 1],
            ]
        ),
        does_not_raise(),
    ),
    "axis: 0, boundary: dirichlet, diff_type: backward": (
        0,
        "dirichlet",
        "backward",
        np.array(
            [
                [1, 0, 0, 0],
                [-1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 1],
            ]
        ),
        does_not_raise(),
    ),
    "axis: 1, boundary: 'periodic'": (
        1,
        "periodic",
        "forward",
        np.array(
            [
                [-1, 0, 1, 0],
                [0, -1, 0, 1],
                [1, 0, -1, 0],
                [0, 1, 0, -1],
            ]
        ),
        does_not_raise(),
    ),
    "axis: 1, boundary: 'dirichlet', diff_type: 'forward'": (
        1,
        "dirichlet",
        "forward",
        np.array(
            [
                [-1, 0, 1, 0],
                [0, -1, 0, 1],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]
        ),
        does_not_raise(),
    ),
    "invalid axis": (2, "dirichlet", "forward", None, pytest.raises(ValueError)),
    "invalid boundary": (0, "invalid boundary", "forward", None, pytest.raises(ValueError)),
    "invalid invalid diff_type": (
        0,
        "dirichlet",
        "invalid diff_type",
        None,
        pytest.raises(ValueError),
    ),
}


@pytest.mark.parametrize(
    ["axis", "boundary", "diff_type", "expected_matrix", "expectation"],
    [pytest.param(*case, id=key) for key, case in MATRIX_ALONG_AXIS_CASES.items()],
)
def test_derivative_matrix_along_axis(
    derivative_instance, axis, boundary, diff_type, expected_matrix, expectation
):
    with expectation:
        matrix = derivative_instance.matrix_along_axis(axis, boundary=boundary, diff_type=diff_type)
        np.testing.assert_array_equal(matrix.toarray(), expected_matrix)


# Test for matrix_gradient method
DIAG_VALUE = 0.7071067811865475
MATRIX_GRADIENT = {
    "f(x, y) = x": (
        lambda x, y: x,
        np.array([[-1, 1, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, -1]]),
        np.array([[-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0], [0, 0, 0, -1]]),
        False,
        does_not_raise(),
    ),
    "f(x, y) = y": (
        lambda x, y: y,
        np.array([[-1, 0, 1, 0], [0, -1, 0, 1], [0, 0, -1, 0], [0, 0, 0, -1]]),
        np.array([[-1, 0, 0, 0], [1, -1, 0, 0], [0, 0, -1, 0], [0, 0, 1, -1]]),
        False,
        does_not_raise(),
    ),
    "f(x, y) = x + y": (
        lambda x, y: x + y,
        np.array([[-2, 1, 1, 0], [0, -2, 0, 1], [0, 0, -2, 1], [0, 0, 0, -2]]),
        np.array([[-2, 0, 1, 0], [1, -2, 0, 1], [0, 0, -2, 0], [0, 0, 1, -2]]),
        False,
        does_not_raise(),
    ),
    "f(x, y) = x + y (w/ diagonal)": (
        lambda x, y: x + y,
        np.array(
            [
                [-2.0 - DIAG_VALUE, 1, 1, DIAG_VALUE],
                [0, -2.0 - DIAG_VALUE, 0, 1],
                [0, 0, -2.0 - DIAG_VALUE, 1],
                [0, 0, 0, -2.0 - DIAG_VALUE],
            ]
        ),
        np.array(
            [
                [-2.0 - DIAG_VALUE, 0, 1, 0],
                [1, -2.0 - DIAG_VALUE, DIAG_VALUE, 1],
                [0, 0, -2.0 - DIAG_VALUE, 0],
                [0, 0, 1, -2.0 - DIAG_VALUE],
            ]
        ),
        True,
        does_not_raise(),
    ),
    "invalid scalar function": (
        lambda x: x,
        None,
        None,
        None,
        pytest.raises(TypeError),
    ),
}


@pytest.mark.parametrize(
    ["function", "expected_matrix_para", "expected_matrix_perp", "diagonal", "expectation"],
    [pytest.param(*case, id=key) for key, case in MATRIX_GRADIENT.items()],
)
def test_derivative_matrix_gradient(
    derivative_instance, function, expected_matrix_para, expected_matrix_perp, diagonal, expectation
):
    with expectation:
        mat_para, mat_perp = derivative_instance.matrix_gradient(function, diagonal=diagonal)
        np.testing.assert_array_equal(mat_para.toarray(), expected_matrix_para)
        np.testing.assert_array_equal(mat_perp.toarray(), expected_matrix_perp)


####################################################################################################
# Additional edge-case tests for improved coverage
# ------------------------------------------------


def test_derivative_repr(derivative_instance):
    """__repr__ should return a formatted string."""
    s = repr(derivative_instance)
    assert "Derivative" in s
    assert "grid" in s


def test_derivative_0d_grid_raises_ValueError():
    """Passing a 0-D array as grid should raise ValueError."""
    with pytest.raises(ValueError, match="1-D"):
        Derivative(np.array(42.0))


def test_derivative_set_grid_map_wrong_shape():
    """grid_map with correct ndim but wrong shape should raise ValueError."""
    # 2D grid (shape (2,3,2)), expects grid_map shape (2,3)
    grid = np.array([[[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]]])  # (2,3,2)
    with pytest.raises(ValueError, match="grid_map shape"):
        Derivative(grid, np.arange(4, dtype=np.int32).reshape(2, 2))


def test_derivative_matrix_along_axis_inner_masked(derivative_instance):
    """matrix_along_axis with a masked interior pixel skips masked rows."""
    # Build a Derivative with grid_map that has -1 for masked pixel at (0,0)
    grid = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])  # (2,2,2)
    masked_grid_map = np.array([[-1, 2], [1, 3]], dtype=np.int32)
    d = Derivative(grid, masked_grid_map)
    # Should run without error; masked pixel skipped in inner loop (line 533)
    mat = d.matrix_along_axis(0, boundary="dirichlet", diff_type="forward")
    assert mat.shape[0] == 4  # bins = max+1 = 4


def test_derivative_matrix_along_axis_boundary_masked():
    """matrix_along_axis with a masked boundary pixel skips boundary update."""
    # grid_map where boundary pixel (edge) is -1
    grid = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])  # (2,2,2)
    # axis=0, forward → edge=-1 (last row). Mask the last row of first column.
    masked_grid_map = np.array([[0, 2], [1, -1]], dtype=np.int32)
    d = Derivative(grid, masked_grid_map)
    # Should run without error; masked boundary pixel skipped (line 543)
    mat = d.matrix_along_axis(0, boundary="dirichlet", diff_type="forward")
    assert mat.shape[0] == 3  # bins = max+1 = 3


def test_derivative_matrix_gradient_1d_grid_map_raises_ValueError():
    """matrix_gradient on 1-D grid (grid_map.ndim == 1) should raise ValueError."""
    d = Derivative(np.array([0.0, 1.0, 2.0]))  # 1D grid → grid_map.ndim=1
    with pytest.raises(ValueError, match="2-D"):
        d.matrix_gradient(lambda x, y: x + y)


def test_derivative_matrix_gradient_1d_coord_raises_ValueError():
    """matrix_gradient on grid with 1-D coordinates should raise ValueError."""
    # Create a 2-D grid_map but with 1-D coordinates (grid.shape[-1] == 1)
    grid = np.array([[[0.0], [1.0]], [[2.0], [3.0]]])  # shape (2,2,1)
    grid_map = np.arange(4, dtype=np.int32).reshape(2, 2)
    d = Derivative(grid, grid_map)
    with pytest.raises(ValueError, match="2-D"):
        d.matrix_gradient(lambda x, y: x + y)


def test_derivative_matrix_gradient_3d_grid_map():
    """matrix_gradient on 3-D grid_map (num_axis==3) should not raise."""
    # shape (2,2,2,3): L=2, M=2, K=2 slices, ndim=3
    grid = np.zeros((2, 2, 2, 3))
    # x-coord varies along L, y-coord varies along M, z-coord varies along K
    grid[1, :, :, 0] = 1.0  # x=1 for L=1
    grid[:, 1, :, 1] = 1.0  # y=1 for M=1
    grid[:, :, 1, 2] = 1.0  # z=1 for K=1
    grid_map = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    d = Derivative(grid, grid_map)
    mat_para, mat_perp = d.matrix_gradient(lambda x, y: x + y)
    assert mat_para.shape[0] == 8


def test_derivative_matrix_gradient_masked_skips_negative_index():
    """matrix_gradient with negative grid_map values skips masked pixels."""
    grid = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])  # (2,2,2)
    grid_map = np.array([[-1, 0], [1, 2]], dtype=np.int32)
    d = Derivative(grid, grid_map)
    mat_para, mat_perp = d.matrix_gradient(lambda x, y: x + y)
    # bins = max+1 = 3; should run without error (line 653 continue)
    assert mat_para.shape[0] == 3


def test_derivative_matrix_gradient_4d_grid_map_raises_NotImplementedError():
    """4-D grid_map (num_axis > 3) raises NotImplementedError."""
    grid = np.zeros((2, 2, 2, 2, 3))  # shape (2,2,2,2,3) → grid_map.ndim = 4
    grid_map = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2)
    d = Derivative(grid, grid_map)
    with pytest.raises(NotImplementedError):
        d.matrix_gradient(lambda x, y: x + y)
