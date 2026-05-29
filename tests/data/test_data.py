import numpy as np

from cherab.inversion.data import get_sample_data


def test_get_sample_data_returns_numpy_array():
    """Default call returns a numpy NpzFile for .npz files."""
    data = get_sample_data("bolo.npz")
    assert isinstance(data, np.lib.npyio.NpzFile)
    data.close()


def test_get_sample_data_asfileobj_false_returns_str():
    """asfileobj=False returns a string path."""
    path = get_sample_data("bolo.npz", asfileobj=False)
    assert isinstance(path, str)
    assert path.endswith("bolo.npz")


def test_get_sample_data_np_load_false_returns_fileobj():
    """np_load=False returns a file-like object instead of numpy array."""
    fobj = get_sample_data("bolo.npz", np_load=False)
    assert hasattr(fobj, "read")
    fobj.close()
