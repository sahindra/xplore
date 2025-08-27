# read version from installed package
from importlib.metadata import version
from .xplore import explore_feature

__all__ = ["explore_feature"]
__version__ = version("xplore")