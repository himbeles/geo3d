import pytest

from geo3d import Vector
import numpy as np


def test_vector_from_array():
    p = Vector([1, 2, 3])
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
