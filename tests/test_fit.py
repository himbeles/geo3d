from geo3d.fit import fit_plane
from geo3d import Point, Plane, Vector


def test_fit_plane():
    points = [[1, 0, 0], [0, 1, 0], [-1, 0, 0]]
    plane = fit_plane(points)
    assert plane == Plane(normal=Vector([0, 0, 1]), point=Point([0, 1 / 3, 0]))
