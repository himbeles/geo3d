import pytest

from geo3d import Frame, frame_wizard, Point, Vector, UnitFrame

def test_frame_wizard():
    t = frame_wizard([0,0,1], [0,1,0], 'z', 'y', [0,0,0])
    assert t == UnitFrame