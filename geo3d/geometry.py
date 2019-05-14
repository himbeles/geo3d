import numpy as np
from scipy.spatial.transform import Rotation as R
#from IPython.display import HTML, display

arcsec_in_radian = 1/3600/180*np.pi 
arcsec_in_degrees = 1/3600 

def normalize(vec):
    """2-norm normalize the given vector"""
    return vec/np.linalg.norm(vec)

class Frame():
    def __init__(self, rot, trans):
        self.rot = rot
        self.trans = trans
        
    def print(self):
        print("rotation\n", self.rot)
        print("Euler angles (XYZ, extrinsic, deg.)\n", R.from_dcm(self.rot).as_euler('xyz', degrees=True))
        print("Euler angles (XYZ, intrinsic, deg.)\n", R.from_dcm(self.rot).as_euler('XYZ', degrees=True))
        print("translation\n", self.trans)
        
    def _repr_html_(self):
        def html_table_from_matrix(mat):
            return '<table><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
            '<td>{}</td>'.format('</td><td>'.join('{:1.8f}'.format(_) for _ in row)) 
                for row in mat))
        def html_table_from_vector(vec):
            return '<table><tr><td>{}</td></tr></table>'.format(
            '</td></tr><tr><td>'.join('{:1.5f}'.format(_) for _ in vec))
                
        html = (
            '''
            <style>
                th {text-align: left;}
            </style>
            <table>
                <tr>
                    <th>rotation matrix</th>
                    <th>Euler angles<br>(XYZ, extr., deg.)</th>
                    <th>Euler angles<br>(XYZ, intr., deg.)</th>
                    <th>translation<br></th>
                </tr>
                <tr><td>'''
            + html_table_from_matrix(self.rot)
            + '</td><td>'
            + html_table_from_vector(R.from_dcm(self.rot).as_euler('xyz', degrees=True))
            + '</td><td>'
            + html_table_from_vector(R.from_dcm(self.rot).as_euler('XYZ', degrees=True))
            + '</td><td>'
            + html_table_from_vector(self.trans)
            + '</td></tr></table>'
        )
        return html
    
    def euler_angles(self, *args, **kwargs):
        return R.from_dcm(self.rot).as_euler(*args, **kwargs)
        
unit_frame = Frame(np.identity(3), np.zeros(3))

def construct_frame(new_x, new_y, new_z, origin=[0,0,0]):
    """
    transformation matrix into a new coordinate system where the new x,y,z axes are given by the provided vectors,
    and the origin is given. 
    """
    rot = np.stack([np.array(new_x), np.array(new_y), np.array(new_z)], 1)
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
    vB = vA.(T.rot) + T.trans
    where vA, vB represent the same vector expressed in different corrdinate systems
    """
    Trot = np.linalg.inv(frameA.rot).dot(frameB.rot)
    Ttrans = (frameB.trans - frameA.trans)@frameA.rot
    return Frame(Trot, Ttrans)

def express_point_in_frame(point, new_frame, original_frame=unit_frame):
    """
    express point given in old frame in the new frame
    """
    trafo = trafo_between_frames(original_frame, new_frame) 
    return (np.array(point) - trafo.trans)@trafo.rot

def express_vector_in_frame(vector, new_frame, original_frame=unit_frame):
    """
    express vector given in old frame in the new frame
    """
    trafo = trafo_between_frames(original_frame, new_frame) 
    return normalize(np.array(vector)@trafo.rot)

def rotate_vector(rot, vec):
    vec = np.array(vec)
    if isinstance(rot, R):
        return normalize(rot.as_dcm()@vec)
    elif isinstance(rot, np.ndarray):
        return normalize(rot@vec)
    else:
        raise Exception('rot is not a rotation object or matrix.')