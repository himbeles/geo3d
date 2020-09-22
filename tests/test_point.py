from geo3d import Point, Vector
import numpy as np


def test_point_from_array():
    p = Point([1, 2, 3])
    assert p[2] == 3


def test_point_from_point():
    p = Point(Point([1, 2, 3]))
    assert p[2] == 3


def test_point_from_vector():
    p = Point(Vector([1, 2, 3]))
    assert p[2] == 3


def test_point_from_tuple():
    p = Point((1, 2, 3))
    assert p[2] == 3


def test_point_from_np():
    p = Point(np.array([1, 2, 3]))
    assert p[2] == 3


def test_point_from_np_bare_nocopy():
    a = np.array([1, 2, 3])
    p = Point.from_array(a, copy=False)
    a[2] = 7
    assert p[2] == 7


def test_point_from_np_bare_copy():
    a = np.array([1, 2, 3])
    p = Point.from_array(a, copy=True)
    a[2] = 7
    assert p[2] == 3
