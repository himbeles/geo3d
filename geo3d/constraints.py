from .geometry import express_point_in_frame, Frame
from scipy.spatial.transform import Rotation as R
import numpy as np

from scipy.optimize import fsolve
def trafo2D(phi, dx, dy): 
    rot = R.from_euler('Z', phi, degrees=True).as_dcm()
    trans = [dx, dy, 0]
    return Frame(rot, trans)

def constrained_movement_2D(rs,cs,ds=[0,0,0]):
    rs = np.array(rs)
    cs = np.array(cs)
    def equations(p):
        phi, dx, dy = p
        t = trafo2D(phi, dx, dy)
        eqs = []
        for (r,c,d) in zip(rs, cs, ds):
            eqs.append((express_point_in_frame(r+d*c,t)-r)@c)
        return eqs
    phi, dx, dy =  fsolve(equations, (0,0,0))
    return phi, dx, dy