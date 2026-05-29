"""Regularization parameter selection criteria."""

from ._base import Criterion
from ._gcv import GCV
from ._lcurve import Lcurve
from ._press import PRESS

__all__ = ["Criterion", "GCV", "Lcurve", "PRESS"]
