import numpy as np
import math

from .query import centroid, distances_plane_to_points
from .geometry import Plane, Vector, Point, VectorLike
from typing import Sequence, Union, Dict, Tuple


def fit_plane(
    points: Sequence[VectorLike], return_fit_props=False
) -> Union[Plane, Tuple[Plane, Dict]]:
    """Fit plane to points

    Args:
        points: Sequence of points
        return_fit_props: True if dictionary with fit properties should be returned. Defaults to False.

    Returns:
        Plane or tuple (Plane, fit_properties) if return_fit_props==True
    """
    centr = centroid(points).as_array()
    svd = np.linalg.svd(points - centr)
    normal = svd[2][-1]
    plane = Plane(Vector(normal), Point(centr))

    residuals = distances_plane_to_points(plane, points)
    residuals_min = np.min(residuals)
    residuals_max = np.max(residuals)
    flatness = residuals_max - residuals_min

    if return_fit_props:
        fit_props = {
            "rms_error": math.sqrt(np.sum(residuals ** 2) / len(points)),
            "residuals": residuals,
            "residuals_min": residuals_min,
            "residuals_max": residuals_max,
            "flatness": flatness,
        }
        return (plane, fit_props)
    else:
        return plane