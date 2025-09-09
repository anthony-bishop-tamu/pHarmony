
from ._version import version as __version__
from .PeakMatcher import MatchPeaks, getPeakPositionsFromFile, NoPeaksFoundError
from ._log import VERBOSE_LEVEL  # noqa: F401
import logging


__author__ = 'Anthony C Bishop, A. Joshua Wand; Texas A&M University'
__all__ = ["MatchPeaks"]
logging.getLogger(__name__).addHandler(logging.NullHandler())
