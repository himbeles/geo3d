import pkg_resources

from .fit import *
from .plane import *
from .frame import *
from .point import *
from .quaternion import *
from .query import *
from .rotation import *
from .vector import *

__version__ = pkg_resources.get_distribution("geo3d").version


UnitFrame = Frame.create_unit_frame()
