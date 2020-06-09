from .geometry import express_point_in_frame, Frame, normalize
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.optimize import fsolve
from typing import Union, List, Tuple, Any


def _trafo2D(phi, dx, dy):
    rot = R.from_euler("Z", phi, degrees=True).as_matrix()
    trans = [dx, dy, 0]
    return Frame(rot, trans)


def constrained_movement_2D(rs, cs, ds=[0, 0, 0]):
    rs = np.array(rs)
    cs = np.array(cs)
    assert (
        len(rs) == 3 and len(cs) == 3 and len(ds) == 3
    ), "number of constraints must be 3 for a 2D problem."

    def equations(p):
        phi, dx, dy = p
        t = _trafo2D(phi, dx, dy)
        eqs = []
        for (r, c, d) in zip(rs, cs, ds):
            eqs.append((express_point_in_frame(r + d * c, t) - r) @ c)
        return eqs

    phi, dx, dy = fsolve(equations, (0, 0, 0))
    return _trafo2D(phi, dx, dy)


def constrained_movement_3D(surface_points, surface_normals, deltas) -> Frame:
    """Rigid body movement under constraints.

    Calculate the movement of a 3D rigid body under the disturbance 
    of the 6 exact surface constraints. 
    The constraints are given as a list of `surface_points` and `surface_normals`.
    A single surface constraint fixes the surface of the rigid body to the given point, 
    such that the rigid body can only move orthogonal to the surface normal.
    Thus, the constraint reduces the degrees of freedom of the rigid body by one. 

    The movement of the rigid body is calculated for a disturbance of the `surface_points`
    along the `surface_normals` given by the amplitude vector `delta`:
        surface_points[i] -> surface_points[i] + surface_normals[i]*deltas[i]
    
    The calculation returns a coordinate transform between frames fixed to the rigid body, 
    before and after the disturbance. 
    For an explanation of the algorithm, see `doc/constrained_movement.pdf`.

    Args:
        surface_points: List of 6 constrained points on the rigid body surface
        surface_normals: List of 6 normal vectors on the rigid body surface at the position 
            of the `surface_points`, for deltas = [0,0,0,...].
        deltas: List of amplitudes of the disturbances of the `surface_points` along the `surface_normals`

    Returns: 
        coordinate transform between rigid body frames before and after the disturbance
    """
    rs = np.array(surface_points)
    cs = np.array(surface_normals)
    assert (
        len(rs) == 6 and len(cs) == 6 and len(deltas) == 6
    ), "number of constraints must be 6 for a 3D problem."

    def equations(p):
        theta_x, theta_y, theta_z, dx, dy, dz = p
        t = Frame.from_extrinsic_euler_and_translations(
            theta_x, theta_y, theta_z, dx, dy, dz
        )
        eqs = []
        for (r, c, d) in zip(rs, cs, deltas):
            cn = normalize(c)
            eqs.append((express_point_in_frame(r + d * cn, t) - r) @ cn)
        return eqs

    theta_x, theta_y, theta_z, dx, dy, dz = fsolve(equations, (0, 0, 0, 0, 0, 0))
    return Frame.from_extrinsic_euler_and_translations(
        theta_x, theta_y, theta_z, dx, dy, dz
    )
