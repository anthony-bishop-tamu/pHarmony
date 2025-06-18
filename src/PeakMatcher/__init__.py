
from ._version import version as __version__
from .PeakMatcher import MatchPeaks, getPeakPositionsFromFile, NoPeaksFoundError
__author__ = 'Anthony C Bishop, A. Joshua Wand; Texas A&M University'
__all__ = ["MatchPeaks"]