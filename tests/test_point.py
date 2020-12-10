from geo3d import Point, UnitFrame, Vector, express_point_in_frame, express_points_in_frame
import numpy as np
import pytest


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


def test_express_point_in_frame(example_frames):
    fa, fb, fc = example_frames
    p0 = (5, 3, 20)
    p1 = (7.07107, 1.41421, -24.00000)

    res0 = Point(p0).express_in_frame(fa)
    assert res0 == Point(p0).express_in_frame(fa, original_frame=UnitFrame)
    assert isinstance(res0._a, np.ndarray)

    res1 = Point(p0).express_in_frame(fa, original_frame=fb)
    assert res1.as_array() == pytest.approx(p1, abs=1e-4)
    assert isinstance(res1._a, np.ndarray)

    assert express_point_in_frame(
        p0, fa, original_frame=fb
    ).as_array() == pytest.approx(p1, abs=1e-4)

    assert Point(p0).express_in_frame(fa, original_frame=fb) == Point(
        p0
    ).express_in_frame(fa.express_in_frame(fb))


def test_express_points_in_frame(example_frames):
    fa, fb, fc = example_frames
    p0 = (5, 3, 20)
    p1 = (7.07107, 1.41421, -24.00000)

    assert express_points_in_frame(
        np.array([p0, p0]), fa, original_frame=fb
    ) == pytest.approx(np.array([p1, p1]), abs=1e-4)
