"""Subpackage for Inversion Problem."""
from .core import _SVDBase, compute_svd
from .gcv import GCV
from .lcurve import Lcurve
from .mfr import Mfr

__all__ = ["compute_svd", "_SVDBase", "Lcurve", "GCV", "Mfr"]
__version__ = "0.1.4"
