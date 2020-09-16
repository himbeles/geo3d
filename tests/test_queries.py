from geo3d import Point
from geo3d.query import distance_between_points, minimize_points_to_points_distance

import pytest


def test_distance_between_points():
    pointA = Point([1, 2, 3])
    pointB = Point([1, 2, 6])
    assert distance_between_points(pointA, pointB) == 3


def test_minimize_points_to_points_distance():
    groupA = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    groupB = [[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0], [0, 0, 1], [0, 0, -1]]
    trafo = minimize_points_to_points_distance(groupA, groupB)
    assert trafo.translation.as_array() == pytest.approx([0, 0, 0], abs=1e-8)
    assert trafo.intrinsic_euler_angles() == pytest.approx([0, 0, 90], abs=1e-6)
