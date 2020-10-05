from geo3d import Point, Vector, Plane
import pytest


def test_plane_from_normal_point():
    normal = Vector([0, 0, -1])
    point = Point([1, 2, 0])
    plane = Plane(normal=normal, point=point)
    assert plane.normal == normal
    assert plane.point == point


def test_plane_as_abcd():
    normal = Vector([0, 0, -1])
    point = Point([1, 2, 0])
    plane = Plane(normal=normal, point=point)
    assert plane.as_abcd() == pytest.approx((0, 0, -1, 0))
