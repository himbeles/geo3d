import pytest
from geo3d import Frame, frame_wizard, Point, Vector, UnitFrame, RotationMatrix
import math

def test_point_express_in_frame(example_frames):
    fa,fb,fc = example_frames
    p = Point([1,1,3])
    pt = p.express_in_frame(fc)
    assert pt==Point([0,0,1])

def test_vector_express_in_frame(example_frames):
    fa,fb,fc = example_frames
    v = Vector([1,1,3])
    vt = v.express_in_frame(fc)
    assert vt==Vector([math.sqrt(2),0,-3])

def test_vector_express_in_frame_with_original_frame(example_frames):
    fa,fb,fc = example_frames
    v = Vector([1,1,3])
    vt = v.express_in_frame(fc, original_frame=fa)
    assert vt==Vector([1,1,3])

def test_point_transform(example_frames):
    fa,fb,fc = example_frames
    p = Point([1,1,3])
    pt = p.transform(fc)
    assert pt==Point([1+math.sqrt(2),1,1])

def test_vector_transform(example_frames):
    fa,fb,fc = example_frames
    v = Vector([1,1,3])
    vt = v.transform(fc)
    assert vt==Vector([math.sqrt(2),0,-3])
