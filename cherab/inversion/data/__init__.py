"""Data module for the cherab-inversion package.

Here we provide a function to access sample data files stored in the ``cherab/inversion/data``
directory within the cherab-inversion package.
"""

import gzip
from pathlib import Path

import numpy as np

__all__ = ["get_sample_data"]


def get_sample_data(fname: str, asfileobj=True, *, np_load=True):
    """Retrieve a sample data file.

    Sample data files are stored in the ``cherab/inversion/data`` directory within the
    cherab-inversion package.

    If the filename ends in .gz, the file is implicitly ungzipped.  If the
    filename ends with .npy or .npz, and *asfileobj* is `True`, the file is
    loaded with `numpy.load`.

    Parameters
    ----------
    fname : str
        Path relative to the ``cherab/inversion/data`` directory.
    asfileobj : bool, optional
        If `True`, return a file object, otherwise return a file path.
    np_load : bool, optional
        If `True`, load .npy or .npz files with `numpy.load`.

    Returns
    -------
    str or file object
        Path to the file or file object.

    Examples
    --------
    >>> data = get_sample_data("bolo.npz")
    """
    path = Path(__file__).parent / fname
    if asfileobj:
        suffix = path.suffix.lower()
        if suffix == ".gz":
            return gzip.open(path)
        elif suffix in [".npy", ".npz"]:
            if np_load:
                return np.load(path)
            else:
                return path.open("rb")
        elif suffix in [".csv", ".xrc", ".txt"]:
            return path.open("r")
        else:
            return path.open("rb")
    else:
        return str(path)
