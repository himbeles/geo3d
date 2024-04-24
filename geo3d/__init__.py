from .fit import *
from .plane import *
from .frame import *
from .point import *
from .quaternion import *
from .query import *
from .rotation import *
from .vector import *

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("geo3d")
except PackageNotFoundError:
    # package is not installed
    pass

UnitFrame = Frame.create_unit_frame()
