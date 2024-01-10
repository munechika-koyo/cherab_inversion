from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from cherab.inversion.derivative import compute_dmat, derivative_matrix, diag_dict, laplacian_matrix

# valid cases
CASES = [
    {
        "vmap": np.arange(6).reshape(2, 3),
        "kernel_type": "x",
        "expected": np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, 0],
                [0, -1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, -1, 1],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(2, 3),
        "kernel_type": "z",
        "expected": np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [-1, 1, 0, 0, 0, 0],
                [0, -1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, -1, 1, 0],
                [0, 0, 0, 0, -1, 1],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(3, 2),
        "kernel_type": "y",
        "expected": np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0],
                [0, -1, 0, 1, 0, 0],
                [0, 0, -1, 0, 1, 0],
                [0, 0, 0, -1, 0, 1],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(3, 2),
        "kernel_type": "r",
        "expected": np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0],
                [0, -1, 0, 1, 0, 0],
                [0, 0, -1, 0, 1, 0],
                [0, 0, 0, -1, 0, 1],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(2, 1, 3),
        "kernel_type": "laplacian4",
        "expected": np.array(
            [
                [-4, 1, 0, 1, 0, 0],
                [1, -4, 1, 0, 1, 0],
                [0, 1, -4, 0, 0, 1],
                [1, 0, 0, -4, 1, 0],
                [0, 1, 0, 1, -4, 1],
                [0, 0, 1, 0, 1, -4],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(3, 1, 2),
        "kernel_type": "laplacian8",
        "expected": np.array(
            [
                [-8, 1, 1, 1, 0, 0],
                [1, -8, 1, 1, 0, 0],
                [1, 1, -8, 1, 1, 1],
                [1, 1, 1, -8, 1, 1],
                [0, 0, 1, 1, -8, 1],
                [0, 0, 1, 1, 1, -8],
            ]
        ),
    },
    {
        "vmap": np.arange(6).reshape(3, 1, 2),
        "kernel_type": "custom",
        "kernel": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        "expected": np.array(
            [
                [-4, 1, 1, 0, 0, 0],
                [1, -4, 0, 1, 0, 0],
                [1, 0, -4, 1, 1, 0],
                [0, 1, 1, -4, 0, 1],
                [0, 0, 1, 0, -4, 1],
                [0, 0, 0, 1, 1, -4],
            ]
        ),
    },
]

# invalid cases
INVALID_CASES = [
    {
        "vmap": np.zeros((2, 2, 2)),  # invalid shape
        "kernel_type": "x",
        "error": ValueError,
    },
    {
        "vmap": np.zeros((4, 3)),
        "kernel_type": "_",  # invalid kernel type
        "error": ValueError,
    },
    {
        "vmap": np.zeros((3, 5)),
        "kernel_type": "custom",
        "kernel": np.zeros((2, 2, 2)),  # invalid kernel dimension
        "error": ValueError,
    },
    {
        "vmap": np.zeros((5, 5)),
        "kernel_type": "custom",
        "kernel": np.zeros((2, 2)),  # invalid kernel shape
        "error": ValueError,
    },
]


def test_compute_dmat():
    # valid tests
    for case in CASES:
        vmap = case["vmap"]
        kernel_type = case["kernel_type"]
        if kernel_type == "custom":
            kernel = case["kernel"]
            dmat = compute_dmat(vmap, kernel_type=kernel_type, kernel=kernel)
        else:
            dmat = compute_dmat(vmap, kernel_type=kernel_type, kernel=None)
        assert np.allclose(dmat.A, case["expected"])

    # invalid tests
    for case in INVALID_CASES:
        vmap = case["vmap"]
        kernel_type = case["kernel_type"]
        if kernel_type == "custom":
            kernel = case["kernel"]
            with pytest.raises(case["error"]):
                compute_dmat(vmap, kernel_type=kernel_type, kernel=kernel)
        else:
            with pytest.raises(case["error"]):
                compute_dmat(vmap, kernel_type=kernel_type, kernel=None)


####################################################################################################

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
        np.testing.assert_array_equal(diag_mat.A, expected)


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
        ) * 0.5,
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
        ) * 0.5,
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
        ) * 2,
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
            dtype=bool
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
}


@pytest.mark.parametrize(
    ["grid_shape", "grid_step", "axis", "scheme", "mask", "expected_dmat", "expectation"],
    [pytest.param(*case, id=key) for key, case in DMAT_CASE.items()],
)
def test_derivative_matrix(grid_shape, grid_step, axis, scheme, mask, expected_dmat, expectation):
    with expectation:
        dmat = derivative_matrix(grid_shape, grid_step, axis, scheme, mask)
        np.testing.assert_array_equal(dmat.A, expected_dmat)


LMAT_CASE = {
    "2x2 grid, 1x1 steps, no diagonal, no mask": (
        (2, 2),
        (1, 1),
        False,
        None,
        np.array([[-4, 1, 1, 0], [1, -4, 0, 1], [1, 0, -4, 1], [0, 1, 1, -4]]),
        does_not_raise(),
    ),
    "3x3 grid, 1x1 steps, no diagnonal, no mask": (
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
                [-6 , 1  , 0  , 1  , 0.5, 0  , 0  , 0  , 0  ],
                [1  , -6 , 1  , 0.5, 1  , 0.5, 0  , 0  , 0  ],
                [0  , 1  , -6 , 0  , 0.5, 1  , 0  , 0  , 0  ],
                [1  , 0.5, 0  , -6 , 1  , 0  , 1  , 0.5, 0  ],
                [0.5, 1  , 0.5, 1  , -6 , 1  , 0.5, 1  , 0.5],
                [0  , 0.5, 1  , 0  , 1  , -6 , 0  , 0.5, 1  ],
                [0  , 0  , 0  , 1  , 0.5, 0  , -6 , 1  , 0  ],
                [0  , 0  , 0  , 0.5, 1  , 0.5, 1  , -6 , 1  ],
                [0  , 0  , 0  , 0  , 0.5, 1  , 0  , 1  , -6 ],
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
                [-52.8 , 16  , 0  , 4  , 3.2, 0  , 0  , 0  , 0  ],
                [16  , -52.8 , 16  , 3.2, 4  , 3.2, 0  , 0  , 0  ],
                [0  , 16  , -52.8 , 0  , 3.2, 4  , 0  , 0  , 0  ],
                [4  , 3.2, 0  , -52.8 , 16  , 0  , 4  , 3.2, 0  ],
                [3.2, 4  , 3.2, 16  , -52.8 , 16  , 3.2, 4  , 3.2],
                [0  , 3.2, 4  , 0  , 16  , -52.8 , 0  , 3.2, 4  ],
                [0  , 0  , 0  , 4  , 3.2, 0  , -52.8 , 16  , 0  ],
                [0  , 0  , 0  , 3.2, 4  , 3.2, 16  , -52.8 , 16  ],
                [0  , 0  , 0  , 0  , 3.2, 4  , 0  , 16  , -52.8 ],
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
            dtype=bool
        ),
        np.array(
            [
                [-4, 1, 0, 1, 0, 0, 0],
                [1, -4, 1, 0, 1, 0, 0],
                [0, 1, -4, 0, 0 ,1, 0],
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
            dtype=bool
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
}


@pytest.mark.parametrize(
    ["grid_shape", "grid_steps", "diagonal", "mask", "expected_lmat", "expectation"],
    [pytest.param(*case, id=key) for key, case in LMAT_CASE.items()],
)
def test_laplacian_matrix(grid_shape, grid_steps, diagonal, mask, expected_lmat, expectation):
    with expectation:
        lmat = laplacian_matrix(grid_shape, grid_steps, diagonal, mask)
        np.testing.assert_array_equal(lmat.A, expected_lmat)
