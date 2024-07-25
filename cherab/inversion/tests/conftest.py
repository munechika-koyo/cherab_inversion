import numpy as np
import pytest

from cherab.inversion.data import get_sample_data


def true_func(x):
    """True function."""
    return 2.0 * np.exp(-6.0 * (x - 0.8) ** 2) + np.exp(-2.0 * (x + 0.5) ** 2)


def kernel(s, t):
    """Kernel function."""
    u = np.pi * (np.sin(s) + np.sin(t))
    if u == 0:
        return np.cos(s) + np.cos(t)
    else:
        return (np.cos(s) + np.cos(t)) * (np.sin(u) / u) ** 2


class TestData:
    """Test data class offering a simple ill-posed inverse problem.

    Attributes
    ----------
    matrix : numpy.ndarray(M, N)
        Operator matrix.
        M and N is the number of data and model, respectively.
    x_true : numpy.ndarray(N, )
        True model.
    b : numpy.ndarray(M, )
        Measured data with white noise.
    """

    def __init__(self):
        _t = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=64, endpoint=True)
        _s = np.linspace(-np.pi * 0.5, np.pi * 0.5, num=64, endpoint=True)

        # construct operator matrix
        matrix = np.array([[kernel(i, j) for j in _t] for i in _s])
        matrix[:, 0] *= 0.5
        matrix[:, -1] *= 0.5
        matrix *= np.abs(_t[1] - _t[0])

        # compute svd

        # set attributes
        self.matrix: np.ndarray = matrix
        self.x_true: np.ndarray = true_func(_t)

        # mesured exact unperturbed data and added white noise
        b_0 = matrix.dot(self.x_true)
        # rng = np.random.default_rng()
        # b_noise = rng.normal(0, 1.0e-4, b_0.size)
        self.b = b_0  # + b_noise


class TestTomographyData:
    """Test data class offering a simple tomography problem.

    Attributes
    ----------
    matrix : numpy.ndarray(M, N)
        Ray transfer matrix (geometry matrix).
        M and N is the number of detectors and voxels, respectively.
    phantom : numpy.ndarray(N, )
        Tomographic phantom data.
    phantom2d : numpy.ndarray(:math:`n_R`, :math:`n_Z`)
        Tomographic phantom data as a 2D array.
        :math:`n_R` and :math:`n_Z` is the number of :math:`R` and :math:`Z` grid points.
    b : numpy.ndarray(M, )
        Measured data with white noise.
    mask : numpy.ndarray(:math:`n_R`, :math:`n_Z`)
        Mask of the voxels.
    voxel_map : numpy.ndarray(:math:`n_R`, :math:`n_Z`)
        Voxel map indicating which voxel each grid point belongs to.
    grid_centres : numpy.ndarray(:math:`n_R`, :math:`n_Z`, 2)
        Coordinates of the grid points. Each grid point is represented by a 2D vectorin the
        :math:`R-Z` plane.
    """

    def __init__(self):
        grid_data = get_sample_data("bolo.npz")
        self.matrix = grid_data["sensitivity_matrix"]
        self.mask = grid_data["mask"].squeeze()
        self.grid_centres = grid_data["grid_centres"]
        self.voxel_map = grid_data["voxel_map"].squeeze()

        self._create_phantom()

        b0 = self.matrix.dot(self.phantom)
        rng = np.random.default_rng()
        self.b = b0 + rng.normal(0, b0.max() * 1.0e-2, b0.size)

    def _create_phantom(self):
        phantom = np.zeros(self.matrix.shape[1])
        for i in range(phantom.shape[0]):
            (row,), (col,) = np.where(self.voxel_map == i)
            phantom[i] = self._emission_function_2d(self.grid_centres[row, col])

        # Create a 2D phantom
        phantom_2d = np.full(self.voxel_map.shape, np.nan)
        phantom_2d[self.mask] = phantom

        self.phantom = phantom
        self.phantom2d = phantom_2d

    def _emission_function_2d(self, rz_point: np.ndarray) -> float:
        """Emission function for a 2D phantom in the :math:`R-Z` plane.

        Parameters
        ----------
        rz_point : numpy.ndarray(2, )
            Coordinates of the point in the :math:`R-Z` plane.

        Returns
        -------
        float
            Emission value at the given point.
        """
        PLASMA_AXIS = np.array([1.5, 1.5])
        LCFS_RADIUS = 1.0

        RADIATION_PEAK = 1.0
        CENTRE_PEAK_WIDTH = 0.05
        RING_WIDTH = 0.025
        direction = rz_point - PLASMA_AXIS
        bearing = np.arctan2(direction[1], direction[0])

        # calculate radius of coordinate from magnetic axis
        radius_from_axis = np.hypot(*direction)
        closest_ring_point = PLASMA_AXIS + (0.5 * direction / radius_from_axis)
        radius_from_ring = np.hypot(*(rz_point - closest_ring_point))

        # evaluate pedestal -> core function
        if radius_from_axis <= LCFS_RADIUS:
            central_radiatior = RADIATION_PEAK * np.exp(-(radius_from_axis**2) / CENTRE_PEAK_WIDTH)

            ring_radiator = (
                RADIATION_PEAK * np.cos(bearing) * np.exp(-(radius_from_ring**2) / RING_WIDTH)
            )
            ring_radiator = max(0.0, ring_radiator)

            return central_radiatior + ring_radiator
        else:
            return 0.0


@pytest.fixture
def test_data():
    return TestData()


@pytest.fixture
def test_tomography_data():
    return TestTomographyData()


@pytest.fixture
def computed_svd(test_data):
    u, sigma, vh = np.linalg.svd(test_data.matrix, full_matrices=False)
    return u, sigma, vh
