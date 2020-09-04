import pytest

from numpy import sqrt

from geo3d import (
    Frame,
    frame_wizard,
    Point,
    Vector,
    UnitFrame,
    RotationMatrix,
    transformation_between_frames,
)


def test_frame_wizard():
    t = frame_wizard([0, 0, 1], [0, 1, 0], "z", "y", [0, 0, 0])
    assert t == UnitFrame


def test_manual_frame_creation():
    rot = RotationMatrix.from_euler_angles("xyz", [90, -45, 45], degrees=True)
    vec = Vector([3, 4, 6])
    f = Frame(rotation_matrix=rot, translation_vector=vec)
    assert f.translation == vec
    assert f.rotation == rot


def test_express_frame_in_frame(example_frames):
    fa,fb,fc = example_frames
    t = fb.express_in_frame(fa)
    assert t.euler_angles("XYZ", degrees=True) == pytest.approx([180, 0, -45])
    assert t.translation.as_array() == pytest.approx([sqrt(2), 0, -4])


def test_transformation_between_frames(example_frames):
    fa,fb,fc = example_frames
    t = transformation_between_frames(fa, fb)
    assert t.euler_angles("XYZ", degrees=True) == pytest.approx([180, 0, -45])
    assert t.translation.as_array() == pytest.approx([1, 1, 4])
