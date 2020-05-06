import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
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
    def __init__(self, rotation_matrix, translation_vector):
        """Frame (transformation) constructor.

        Basic constructor method of a frame object. 
        The arguments rot and trans are taken as 
        the rotation matrix and translation vector of
        a frame-to-frame transformation. 

        Args: 
            rotation_matrix: 3x3 orthogonal rotation matrix
            translation_vector: 3x1 or 1x3 translation vector
        """
        self._rot = np.array(rotation_matrix)
        self._trans = np.array(translation_vector)
        
    def __str__(self):
        # basic string representation of a frame
        s = ""
        s += "rotation\n{}".format(self._rot)
        s += "\nEuler angles (xyz, extrinsic, deg.)\n{}".format(self.euler_angles('xyz', degrees=True))
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
                    <th>Euler angles<br>(xyz, extr., deg.)</th>
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

    def SA_pastable_string(self):
        """ Spatial Analyzer compatible string representation
            
        Returns:
            str: SA compatible flattened 4x4 transformation matrix
        """
        p = np.eye(4)
        p[0:3,0:3] = self._rot
        p[0:3,3] = self._trans
        return ' '.join(['{:0.12f}'.format(i) for i in p.flatten()])
        
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

        This is the transformation T0 between `frameA` and `frameB` expressed in frameA.
        Transformation T0 such that T0*frameA = B -> T0=frameB*inv(frameA).
        T0 expressed in frameA becomes: T = inv(frameA) T0 frameA = inv(frameA) frameB.

        This is equivalent to a transformation T between `frameA` and `frameB` such that 
        vA = (T.rotation).vB + T.translation
        where vA, vB represent the same vector expressed in frameA and frameB, respectively:
        frameA.rotation * vA + frameA.translation = frameB.rotation * vB + frameB.translation
        
        Args:
            frameA: Reference frame to express this frame in. 

        Returns:
            transformation as a new frame object
        """
        return express_frame_in_frame(self, frameA)
    
    @classmethod
    def create_unit_frame(cls)->'Frame':
        """Construct unit frame.

        Construct transformation frame with no rotation and translation.

        Returns:
            new unit frame object
        """
        return Frame(np.identity(3), np.zeros(3))

    @classmethod
    def from_SA_pastable_string(cls, SA_string: str)->'Frame':
        """Construct frame from SA transformation string.

        Construct transformation frame from SA transformation string.

        Args:
            SA_string: trafo string from SA
        Returns:
            new frame object
        """
        try:
            a = np.array([float( s ) for s in SA_string.split(' ', 15)]).reshape((4,4))
            rot = a[0:3,0:3]
            trans = a[:3,3]
        except:
            raise Exception('SA string could not be read.')
        return Frame(rot, trans)


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

    def transform(self, trafo):
        """Transform this vector by a given transformation frame.

        Transform this vector by a given transformation frame. Basically the inverse of "express vector in frame".

        Args:
            trafo: Transformation frame 

        Returns:
            vector expressed in the original frame, but transformed.
        """
        return rotate_vector(trafo._rot, self._vec)

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
    
    def transform(self, trafo):
        """Transform this point by a given transformation frame.

        Transform this point by a given transformation frame. Basically the inverse of "express point in frame".

        Args:
            trafo: Transformation frame 

        Returns:
            Point expressed in the original frame but transformed.
        """
        return Point(trafo._rot@np.array(self._p) + trafo._trans)

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
        return cls(R.from_euler(seq, angles, degrees=degrees).as_matrix())


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
    The secondary axis of the frame points along `secondary_vec` 
    projected into the plane perpendicular to `primary_vec` .
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
    """Transformation between frameA and frameB.
    
    Transformation between frameA and frameB, expressed in unit frame.
    Construct transformation T between `frameA` and `frameB` 
    such that T*frameA = B -> T=frameB*inv(frameA)
    
    Args:
        frameA: Reference frame.
        frameB: Final frame.

    Returns:
        transformation as a new frame object
    """
    Trot = frameB._rot.dot(np.transpose(frameA._rot))
    Ttrans = (frameB._trans - frameA._trans)
    return Frame(Trot, Ttrans)

def express_frame_in_frame(frameB, frameA):
    """Express frameB in frameA.

    This is the transformation T0 between `frameA` and `frameB` expressed in frameA.
    Transformation T0 such that T0*frameA = B -> T0=frameB*inv(frameA).
    T0 expressed in frameA becomes: T = inv(frameA) T0 frameA = inv(frameA) frameB.

    This is equivalent to a transformation T between `frameA` and `frameB` such that 
    vA = (T.rotation).vB + T.translation
    where vA, vB represent the same vector expressed in frameA and frameB, respectively:
    frameA.rotation * vA + frameA.translation = frameB.rotation * vB + frameB.translation
    
    Args:
        frameA: Reference frame.
        frameB: Final frame.

    Returns:
        transformation as a new frame object
    """
    Trot = np.transpose(frameA._rot).dot(frameB._rot)
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
    return Point((np.array(point) - trafo._trans)@trafo._rot) # multiplication to the right is the same as with transpose to the left 

def express_points_in_frame(points, new_frame, original_frame=Frame.create_unit_frame()):
    """Express points in a different frame.

    Express the `points` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        points: Nx3 point object
        new_frame: Frame to express this point in. 
        original_frame: Reference frame where the point is specified in.

    Returns:
        Point expressed in `new_frame`.
    """
    trafo = trafo_between_frames(original_frame, new_frame) 
    return (np.array(points) - trafo._trans)@trafo._rot # multiplication to the right is the same as with transpose to the left 

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
        return normalize(rot.as_matrix()@vec)
    elif isinstance(rot, np.ndarray):
        return normalize(rot@vec)
    else:
        raise Exception('rot is not a rotation object or matrix.')

def transform_points(points, trafo):
    return np.dot(np.array(points), trafo._rot.T) + trafo._trans

def distance_between_points(pointA, pointB):
    return np.linalg.norm(np.array(pointA) - np.array(pointB))

def minimize_points_to_points_distance(groupA, groupB, return_report=False, method='Powell', tol=1e-6):
    """Transform point group to minimize point-group-to-point-group distance.

    Returns a transformation (Frame object) that, if applied to all points in point group
    `groupA`, minimizes the distance between all points in `groupA` an the corresponding 
    points in `groupB`.
    
    Args:
        groupA: Array of Points.
        groupB: Array of Points (same size as groupA).
        return_report: True if report of minimization algorithm should be returned

    Returns:
        transformation as a new Frame object or tuple of transformation and minimization report
        if return_report==True
    """
    # return transform that maps groupA onto groupB with minimum point-to-point distance
    def cost(x):  
        [r1, r2, r3, t1, t2, t3] = x 
        rot = R.from_rotvec([r1, r2, r3]).as_matrix()
        trans = np.array([t1, t2, t3])
        t = Frame(rot, trans)
        c = np.sqrt(np.mean(np.power([
            distance_between_points(pB, Point(pA).transform(t)) for (pA,pB) in zip(groupA, groupB)], 2)))
        return c
    m = minimize(cost, [0,0,0,0,0,0], tol=tol, method=method)
    t = Frame(R.from_rotvec(m['x'][:3]).as_matrix(), m['x'][3:])
    if return_report:
        return t, m
    else: 
        return t