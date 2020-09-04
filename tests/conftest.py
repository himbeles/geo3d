import pytest

from geo3d import frame_wizard, Vector

@pytest.fixture
def example_frames():
    # rotation only from UnitFrame
    fa = frame_wizard(Vector([1, 1, 0]), Vector([1, -1, 0]), "x", "y", origin=[0, 0, 0])
    # translation only from UnitFrame
    fb = frame_wizard(Vector([1, 0, 0]), Vector([0, 1, 0]), "x", "y", origin=[1, 1, 4])
    # rotation and translation from UnitFrame
    fc = frame_wizard(Vector([1, 1, 0]), Vector([1, -1, 0]), "x", "y", origin=[1, 1, 4])
    return fa, fb, fc