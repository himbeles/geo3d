from .geometry import *

import pkg_resources

__version__ = pkg_resources.get_distribution("geo3d").version


UnitFrame = Frame.create_unit_frame()
