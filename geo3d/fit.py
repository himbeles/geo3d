import numpy as np
import math

from .query import centroid, distances_plane_to_points
from .geometry import MultipleVectorLike, Plane, Vector, Point
from typing import Union, Dict, Tuple, overload, Any
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

@overload
def fit_plane(
    points: MultipleVectorLike, return_fit_props: Literal[False] = False
) -> Plane:
    ...


@overload
def fit_plane(
    points: MultipleVectorLike, return_fit_props: Literal[True] = True
) -> Tuple[Plane, Dict[str, Any]]:
    ...


def fit_plane(
    points: MultipleVectorLike, return_fit_props: bool = False
) -> Union[Plane, Tuple[Plane, Dict[str, Any]]]:
    """Fit plane to points

    Args:
        points: Sequence of points
        return_fit_props: True if dictionary with fit properties should be returned. Defaults to False.

    Returns:
        Plane or tuple (Plane, fit_properties) if return_fit_props==True
    """
    centr = centroid(points).as_array()
    svd = np.linalg.svd(np.array(points) - centr)
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