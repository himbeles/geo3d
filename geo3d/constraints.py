from .geometry import express_point_in_frame, Frame, normalize
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.optimize import fsolve

def _trafo2D(phi, dx, dy): 
    rot = R.from_euler('Z', phi, degrees=True).as_dcm()
    trans = [dx, dy, 0]
    return Frame(rot, trans)

def constrained_movement_2D(rs,cs,ds=[0,0,0]):
    rs = np.array(rs)
    cs = np.array(cs)
    assert len(rs)==3 and len(cs)==3 and len(ds)==3, 'number of constraints must be 3 for a 2D problem.'
    def equations(p):
        phi, dx, dy = p
        t = _trafo2D(phi, dx, dy)
        eqs = []
        for (r,c,d) in zip(rs, cs, ds):
            eqs.append((express_point_in_frame(r+d*c,t)-r)@c)
        return eqs
    phi, dx, dy =  fsolve(equations, (0,0,0))
    return _trafo2D(phi, dx, dy)

def _trafo3D(theta_x, theta_y, theta_z, dx, dy, dz): 
    rot = R.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True).as_dcm()
    trans = [dx, dy, dz]
    return Frame(rot, trans)

def constrained_movement_3D(rs,cs,ds=[0,0,0]):
    rs = np.array(rs)
    cs = np.array(cs)
    assert len(rs)==6 and len(cs)==6 and len(ds)==6, 'number of constraints must be 6 for a 3D problem.'
    def equations(p):
        theta_x, theta_y, theta_z, dx, dy, dz = p
        t = _trafo3D(theta_x, theta_y, theta_z, dx, dy, dz)
        eqs = []
        for (r,c,d) in zip(rs, cs, ds):
            cn = normalize(c)
            eqs.append((express_point_in_frame(r+d*cn,t)-r)@cn)
        return eqs
    theta_x, theta_y, theta_z, dx, dy, dz =  fsolve(equations, (0,0,0,0,0,0))
    return _trafo3D(theta_x, theta_y, theta_z, dx, dy, dz)