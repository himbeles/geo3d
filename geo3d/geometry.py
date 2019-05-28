import numpy as np
from scipy.spatial.transform import Rotation as R
from .auxiliary import html_table_from_matrix, html_table_from_vector
from typing import Union, List, Tuple, Any

def normalize(vec)->np.ndarray:
    """2-norm normalize the given vector.
    
    Args: 
        vec: non-normalized vector
    Returns:
        normalized vector
    """
    return np.array(vec)/np.linalg.norm(np.array(vec))

class Frame:
    def __init__(self, rot, trans):
        """Frame (transformation) constructor.

        Basic constructor method of a frame object. 
        The arguments rot and trans are taken as 
        the rotation matrix and translation vector of
        a frame-to-frame transformation. 

        Args: 
            rot: 3x3 orthogonal rotation matrix
            trans: 3x1 or 1x3 translation vector
        """
        self._rot = np.array(rot)
        self._trans = np.array(trans)
        
    def __str__(self):
        # basic string representation of a frame
        s = ""
        s += "rotation\n{}".format(self._rot)
        s += "\nEuler angles (XYZ, extrinsic, deg.)\n{}".format(self.euler_angles('xyz', degrees=True))
        s += "\nEuler angles (XYZ, intrinsic, deg.)\n{}".format(self.euler_angles('XYZ', degrees=True))
        s += "\ntranslation\n{}".format(self._trans)
        return s
        
    def _repr_html_(self):
        # html representation of a frame
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
    
    def euler_angles(self, *args, **kwargs)->np.ndarray:
        """Frame rotation Euler angles.

        Args are passed to scipy.spatial.transform.Rotation.as_euler(*args, **kwargs).

        Args:
            seq: Specifies sequence of axes for rotations. Up to 3 characters
                belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
                {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
                rotations cannot be mixed in one function call.
            angles : float or array_like, shape (N,) or (N, [1 or 2 or 3])
                Euler angles specified in radians (`degrees` is False) or degrees
                (`degrees` is True).
                For a single character `seq`, `angles` can be:
                - a single value
                - array_like with shape (N,), where each `angle[i]`
                corresponds to a single rotation
                - array_like with shape (N, 1), where each `angle[i, 0]`
                corresponds to a single rotation
                For 2- and 3-character wide `seq`, `angles` can be:
                - array_like with shape (W,) where `W` is the width of
                `seq`, which corresponds to a single rotation with `W` axes
                - array_like with shape (N, W) where each `angle[i]`
                corresponds to a sequence of Euler angles describing a single
                rotation
            degrees : If True, then the given angles are assumed to be in degrees.
                Default is False.
        
        Returns: 
            Array of frame rotation Euler angles.
        """
        return R.from_dcm(self._rot).as_euler(*args, **kwargs)

    @property
    def translation(self)->'Vector':
        """Frame translation vector.

        Returns: 
            Frame translation vector
        """
        return Vector(self._trans)
    
    @property
    def rotation(self)->np.ndarray:
        """Frame rotation matrix.

        Returns: 
            Frame rotation matrix
        """
        return RotationMatrix(self._rot)

    def express_in_frame(self, frameA: 'Frame')->'Frame':
        """Express this frame in a different frame.

        Construct transformation T between frameA and this frame (frameB) such that 
        vB = vA.(T.rotation) + T.translation
        where vA, vB represent the same vector expressed in frameA and frameB, respectively.
        
        Args:
            frameA: Reference frame to express this frame in. 

        Returns:
            transformation as a new frame object
        """
        return trafo_between_frames(frameA, self)
    
    @classmethod
    def create_unit_frame(cls)->'Frame':
        """Construct unit frame.

        Construct transformation a frame with no rotation and translation.

        Returns:
            new unit frame object
        """
        return Frame(np.identity(3), np.zeros(3))


class Vector:
    def __init__(self, v):
        self._vec = np.array(v)
    
    def express_in_frame(self, new_frame, original_frame=Frame.create_unit_frame()) -> 'Vector':
        """Express this vector in a different frame.

        Express the vector given in the frame `original_frame` in a different frame `new_frame`.

        Args:
            new_frame: Frame to express this vector in. 
            original_frame: Reference frame where the vector is specified in.

        Returns:
            Vector expressed in `new_frame`.
        """
        return express_vector_in_frame(self._vec, new_frame, original_frame)
    
    def _repr_html_(self):
        html = (
            html_table_from_vector(self._vec, indices=['x','y','z'])
        )
        return html
    
    def as_array(self):
        return self._vec

    def __array__(self):
        return self._vec

    def __getitem__(self,key):
        return self._vec[key]

    def normalize(self):
        return normalize(self._vec)

class Point:
    def __init__(self, p):
        self._p = np.array(p)
    
    def express_in_frame(self, new_frame, original_frame=Frame.create_unit_frame()):
        """Express this point in a different frame.

        Express the point given in the frame `original_frame` in a different frame `new_frame`.

        Args:
            new_frame: Frame to express this point in. 
            original_frame: Reference frame where the point is specified in.

        Returns:
            Point expressed in `new_frame`.
        """
        return express_point_in_frame(self._p, new_frame, original_frame)
    
    def _repr_html_(self):
        html = (
            html_table_from_vector(self._p, indices=['x','y','z'])
        )
        return html

    def as_array(self):
        return self._p

    def __array__(self):
        return self._p

    def __getitem__(self,key):
        return self._p[key]

class RotationMatrix:
    def __init__(self, m):
        self._m = np.array(m)

    def _repr_html_(self):
        html = (
            html_table_from_matrix(self._m)
        )
        return html
    
    def as_array(self):
        return self._m

    def __array__(self):
        return self._m

    def __getitem__(self,key):
        return self._m[key]

    @classmethod
    def from_euler_angles(cls, seq: str, angles: np.ndarray, degrees: bool = False):
        """Rotation matrix from Euler angles.
        
        Arguments are passed to scipy.spatial.transform.Rotation.from_euler(*args, **kwargs).

        Args:
            seq: Specifies sequence of axes for rotations. Up to 3 characters
                belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
                {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
                rotations cannot be mixed in one function call.
            angles : float or array_like, shape (N,) or (N, [1 or 2 or 3])
                Euler angles specified in radians (`degrees` is False) or degrees
                (`degrees` is True).
                For a single character `seq`, `angles` can be:
                - a single value
                - array_like with shape (N,), where each `angle[i]`
                corresponds to a single rotation
                - array_like with shape (N, 1), where each `angle[i, 0]`
                corresponds to a single rotation
                For 2- and 3-character wide `seq`, `angles` can be:
                - array_like with shape (W,) where `W` is the width of
                `seq`, which corresponds to a single rotation with `W` axes
                - array_like with shape (N, W) where each `angle[i]`
                corresponds to a sequence of Euler angles describing a single
                rotation
            degrees : If True, then the given angles are assumed to be in degrees.
                Default is False.
        """
        return cls(R.from_euler(seq, angles, degrees=degrees).as_dcm())


def _construct_frame(new_x, new_y, new_z, origin=[0,0,0]):
    """
    transformation matrix into a new coordinate system where the new x,y,z axes are given by the provided vectors,
    and the origin is given. 
    """
    rot = np.stack([normalize(new_x), normalize(new_y), normalize(new_z)], 1)
    trans = np.array(origin)
    return Frame(rot, trans)

def frame_wizard(primary_vec, secondary_vec, primary_axis: str, secondary_axis: str, origin=[0,0,0]):
    """Frame-Wizard-type Frame constructor.

    This constructor of a Frame object works anaogously to the Spatial Analyzer Frame Wizard.
    The primary axis of the frame is chosen as the `primary_vec`. 
    The secondary axis of the frame is the `secondary_vec` projected into the plane spanned by
    `primary_vec` and `seondary_vec`.
    The tertiary axis completes the right-handed frame.
    The corresponding primary and secondary axes labels are given as input arguments
    `primary_axis`, `secondary_axis`.
    The `origin` of the frame can be specified. 

    Args:
        primary_vec: vector specifying the primary axis
        secondary_vec: vector used in the construction of the secondary axis
        primary_axis: label x/z/y of primary axis
        secondary_axis: label x/z/y of secondary axis
        origin: point coordinates of the frame origin
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
    """Transformation matrix between frameA and frameB.

    Construct transformation T between `frameA` and `frameB` such that 
    vB = vA.(T.rotation) + T.translation
    where vA, vB represent the same vector expressed in frameA and frameB, respectively.
    
    Args:
        frameA: Reference frame.
        frameB: Final frame.

    Returns:
        transformation as a new frame object
    """
    Trot = np.linalg.inv(frameA._rot).dot(frameB._rot)
    Ttrans = (frameB._trans - frameA._trans)@frameA._rot
    return Frame(Trot, Ttrans)

def express_point_in_frame(point, new_frame, original_frame=Frame.create_unit_frame())->Point:
    """Express a point in a different frame.

    Express the `point` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        point: 3x1 point object
        new_frame: Frame to express this point in. 
        original_frame: Reference frame where the point is specified in.

    Returns:
        Point expressed in `new_frame`.
    """
    trafo = trafo_between_frames(original_frame, new_frame) 
    return Point((np.array(point) - trafo._trans)@trafo._rot)

def express_vector_in_frame(vector, new_frame, original_frame=Frame.create_unit_frame())->Vector:
    """Express a vector in a different frame.

    Express the `vector` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        vector: 3x1 vector object
        new_frame: Frame to express this vector in. 
        original_frame: Reference frame where the vector is specified in.

    Returns:
        Vector expressed in `new_frame`.
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