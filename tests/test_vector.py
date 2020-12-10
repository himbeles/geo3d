from geo3d import Vector, Point, express_vector_in_frame
import numpy as np
import pytest


def test_vector_from_array():
    p = Vector([1, 2, 3])
    assert p[2] == 3


def test_vector_from_vector():
    p = Vector(Vector([1, 2, 3]))
    assert p[2] == 3


def test_vector_from_point():
    p = Vector(Point([1, 2, 3]))
    assert p[2] == 3


def test_vector_from_tuple():
    p = Vector((1, 2, 3))
    assert p[2] == 3


def test_vector_from_np():
    p = Vector(np.array([1, 2, 3]))
    assert p[2] == 3


def test_vector_from_np_bare_nocopy():
    a = np.array([1, 2, 3])
    p = Vector.from_array(a, copy=False)
    a[2] = 7
    assert p[2] == 7


def test_vector_from_np_bare_copy():
    a = np.array([1, 2, 3])
    p = Vector.from_array(a, copy=True)
    a[2] = 7
    assert p[2] == 3


def test_express_vector_in_frame(example_frames):
    fa, fb, fc = example_frames
    v0 = (1, 3, 0)
    v1 = (2.82843, -1.41421, 0)

    res0 = Vector(v0).express_in_frame(
        fa, original_frame=fb
    )
    assert res0.as_array() == pytest.approx(v1, abs=1e-4)
    assert isinstance(res0._a, np.ndarray)

    assert express_vector_in_frame(
        v0, fa, original_frame=fb
    ).as_array() == pytest.approx(v1, abs=1e-4)

    assert Vector(v0).express_in_frame(fa, original_frame=fb) == Vector(
        v0
    ).express_in_frame(fa.express_in_frame(fb))
