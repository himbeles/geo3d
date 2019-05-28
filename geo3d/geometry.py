import numpy as np
from scipy.spatial.transform import Rotation as R
from .auxiliary import html_table_from_matrix, html_table_from_vector

def normalize(vec):
    """2-norm normalize the given vector"""
    return np.array(vec)/np.linalg.norm(np.array(vec))

class Frame:
    def __init__(self, rot, trans):
        self._rot = np.array(rot)
        self._trans = np.array(trans)
        
    def print(self):
        print("rotation\n", self._rot)
        print("Euler angles (XYZ, extrinsic, deg.)\n", self.euler_angles('xyz', degrees=True))
        print("Euler angles (XYZ, intrinsic, deg.)\n", self.euler_angles('XYZ', degrees=True))
        print("translation\n", self._trans)
        
    def _repr_html_(self):
        html = (
            '''
            <table>
                <tr>
                    <th>rotation matrix</th>
                    <th>Euler angles<br>(XYZ, extr., deg.)</th>
                    <th>Euler angles<br>(XYZ, intr., deg.)</th>
                    <th>translation<br></th>
                </tr>
                <tr><td>'''
            + html_table_from_matrix(self._rot)
            + '</td><td>'
            + html_table_from_vector(self.euler_angles('xyz', degrees=True), indices=['θx','θy','θz'])
            + '</td><td>'
            + html_table_from_vector(self.euler_angles('XYZ', degrees=True), indices=['θx','θy','θz'])
            + '</td><td>'
            + html_table_from_vector(self._trans, indices=['x','y','z'])
            + '</td></tr></table>'
        )
        return html
    
    def euler_angles(self, *args, **kwargs):
        return R.from_dcm(self._rot).as_euler(*args, **kwargs)

    @property
    def translation(self):
        return self._trans
    
    @property
    def rotation(self):
        return self._rot

    def express_in_frame(self, frameA):
        """
        construct trafo T between frameA and frameB such that 
        vB = vA.(T._rot) + T._trans
        where vA, vB represent the same vector expressed in different corrdinate systems

        return the trafo as frame
        """
        return trafo_between_frames(frameA, self)
    
    @classmethod
    def create_unit_frame(cls):
        return Frame(np.identity(3), np.zeros(3))

class Vector:
    def __init__(self, v):
        self.vec = np.array(v)
    
    def express_in_frame(self, new_frame, original_frame=Frame.create_unit_frame()):
        """
        express vector given in original frame in the new frame
        """
        return express_vector_in_frame(self.vec, new_frame, original_frame)
    
    def _repr_html_(self):
        html = (
            html_table_from_vector(self.vec, indices=['x','y','z'])
        )
        return html
    
    def as_array(self):
        return self.vec

    def __array__(self):
        return self.vec

    def __getitem__(self,key):
        return self.vec[key]

    def normalize(self):
        return normalize(self.vec)

class Point:
    def __init__(self, p):
        self.p = np.array(p)
    
    def express_in_frame(self, new_frame, original_frame=Frame.create_unit_frame()):
        """
        express point given in original frame in the new frame
        """
        return express_point_in_frame(self.p, new_frame, original_frame)
    
    def _repr_html_(self):
        html = (
            html_table_from_vector(self.p, indices=['x','y','z'])
        )
        return html

    def as_array(self):
        return self.p

    def __array__(self):
        return self.p

    def __getitem__(self,key):
        return self.p[key]

def construct_frame(new_x, new_y, new_z, origin=[0,0,0]):
    """
    transformation matrix into a new coordinate system where the new x,y,z axes are given by the provided vectors,
    and the origin is given. 
    """
    rot = np.stack([normalize(new_x), normalize(new_y), normalize(new_z)], 1)
    trans = np.array(origin)
    return Frame(rot, trans)

def frame_wizard(primary_vec, secondary_vec, primary_axis, secondary_axis, origin=[0,0,0]):
    """
    this works anaogously to SA Frame Wizard
    """
    assert secondary_axis != primary_axis, 'secondary axis must not equal primary axis: choose from x,y,z'
    primary_vec = normalize(primary_vec)
    secondary_vec = normalize(secondary_vec)
    rot = np.zeros((3,3))
    column_dict = {
        'x': 0,
        'y': 1,
        'z': 2
    }
    primary_index = column_dict.pop(primary_axis)
    secondary_index = column_dict.pop(secondary_axis)
    tertiary_index = list(column_dict.values())[0]
    axes = [primary_index, secondary_index]
    if axes in ([0,1], [1,2], [2,0]):
        signature_perm = 1
    else: 
        signature_perm = -1
    
    # set primary axis in rotation matrix
    rot[:, primary_index] = primary_vec
    
    # construct tertiary and secondary axis
    rot[:, tertiary_index] = signature_perm * normalize(np.cross(primary_vec, secondary_vec))
    rot[:, secondary_index] = -signature_perm * np.cross(primary_vec, rot[:, tertiary_index])
    
    return Frame(rot, np.array(origin))
    

def trafo_between_frames(frameA, frameB):
    """
    construct trafo T between frameA and frameB such that 
    vB = vA.(T._rot) + T._trans
    where vA, vB represent the same vector expressed in different corrdinate systems
    """
    Trot = np.linalg.inv(frameA._rot).dot(frameB._rot)
    Ttrans = (frameB._trans - frameA._trans)@frameA._rot
    return Frame(Trot, Ttrans)

def express_point_in_frame(point, new_frame, original_frame=Frame.create_unit_frame()):
    """
    express point given in old frame in the new frame
    """
    trafo = trafo_between_frames(original_frame, new_frame) 
    return Point((np.array(point) - trafo._trans)@trafo._rot)

def express_vector_in_frame(vector, new_frame, original_frame=Frame.create_unit_frame()):
    """
    express vector given in old frame in the new frame
    """
    trafo = trafo_between_frames(original_frame, new_frame) 
    return Vector(np.array(vector)@trafo._rot)

def rotate_vector(rot, vec):
    vec = np.array(vec)
    if isinstance(rot, R):
        return normalize(rot.as_dcm()@vec)
    elif isinstance(rot, np.ndarray):
        return normalize(rot@vec)
    else:
        raise Exception('rot is not a rotation object or matrix.')