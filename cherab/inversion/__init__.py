"""Subpackage for Inversion Problem."""
# If a version with git hash was stored, use that instead
from . import version  # noqa: F401

# Import some features from the subpackages
from .core import _SVDBase, compute_svd
from .gcv import GCV
from .lcurve import Lcurve
from .mfr import Mfr
from .version import __version__  # noqa: F401

__all__ = ["compute_svd", "_SVDBase", "Lcurve", "GCV", "Mfr"]
