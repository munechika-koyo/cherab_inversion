"""Regularization-based inversion methods."""

from . import criteria
from ._base import _SVDBase
from ._mfr import MFR
from ._svd import SVD
from ._tsvd import TSVD
from .criteria import GCV, PRESS, Lcurve
from .utility import compute_svd

__all__ = [
    "criteria",
    "_SVDBase",
    "SVD",
    "TSVD",
    "MFR",
    "GCV",
    "Lcurve",
    "PRESS",
    "compute_svd",
]
