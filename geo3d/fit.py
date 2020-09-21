import numpy as np

from .query import centroid
from .geometry import Plane, Vector, Point, VectorLike
from typing import Sequence


def fit_plane(points: Sequence[VectorLike]):
    centr = centroid(points).as_array()
    normal = np.linalg.svd(points - centr)[2][-1]
    return Plane(Vector(normal), Point(centr))