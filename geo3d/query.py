import numpy as np

from .geometry import Frame, MultipleVectorLike, Point, norm_L2, VectorLike, Plane
from .linalg import dot_vec_vec, norm_L2

from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from typing import Sequence


def distance_between_points(pointA: VectorLike, pointB: VectorLike) -> float:
    """Distance between two points.

    Args:
        pointA: 1st point
        pointB: 2nd point

    Returns:
        Euclidean distance between the two points
    """
    return norm_L2(np.asarray(pointA) - np.asarray(pointB))


def distances_plane_to_points(plane: Plane, points: MultipleVectorLike) -> np.ndarray:
    """Shortest distances between plane and points

    The distances are given from the plane to the points, along the plane normal.
    Negative distances indicate points positioned along the negative plane normal.

    Args:
        plane: The queried plane
        points: The sequence of queried points

    Returns:
        np.array of floats: distances for all points
    """

    normal = plane.normal.as_array()
    cartesian_d = dot_vec_vec(normal, plane.point.as_array())
    normal_length = norm_L2(normal)
    return (np.dot(points, normal) - cartesian_d) / normal_length


def centroid(points: MultipleVectorLike) -> Point:
    """Centroid for a sequence of points

    Args:
        points: The queried sequence of points

    Returns:
        Point: Centroid point
    """
    return Point(np.sum(np.array(points), axis=0) / len(points))


def minimize_points_to_points_distance(
    groupA: MultipleVectorLike,
    groupB: MultipleVectorLike,
    return_report=False,
    method="Powell",
    tol=1e-6,
):
    """Find point group transform to minimize point-group-to-point-group distance.

    Returns a transformation (Frame object) that, if applied to all points in point group
    `groupA`, minimizes the distance between all points in `groupA` an the corresponding
    points in `groupB`.

    Args:
        groupA: Sequence of Points.
        groupB: Sequence of Points (same size as groupA).
        return_report: True if report of minimization algorithm should be returned

    Returns:
        Transformation, or tuple of transformation and minimization report if return_report==True
    """
    assert len(np.array(groupA)) >= 3, "Point list A is too short."
    assert len(np.array(groupB)) >= 3, "Point list B is too short."

    # return transform that maps groupA onto groupB with minimum point-to-point distance
    def cost(x):
        [r1, r2, r3, t1, t2, t3] = x
        rot = R.from_rotvec([r1, r2, r3]).as_matrix()
        trans = np.asarray([t1, t2, t3])
        t = Frame(rot, trans)
        c = np.sqrt(
            np.mean(
                np.power(
                    [
                        distance_between_points(pB, Point(pA).transform(t))
                        for (pA, pB) in zip(groupA, groupB)
                    ],
                    2,
                )
            )
        )
        return c

    m = minimize(cost, [0, 0, 0, 0, 0, 0], tol=tol, method=method)
    t = Frame(R.from_rotvec(m["x"][:3]).as_matrix(), m["x"][3:])
    if return_report:
        return t, m
    else:
        return t