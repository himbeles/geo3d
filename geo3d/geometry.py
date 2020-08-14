from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from .auxiliary import html_table_from_matrix, html_table_from_vector
from typing import Union, List, Tuple, Any, Sequence, TypeVar, Optional

RotationMatrixLike = Union[Sequence[Sequence[float]], np.ndarray, "RotationMatrix"]
VectorLike = Union[Sequence[float], np.ndarray, "Vector", "Point"]


def normalize(vec: VectorLike) -> np.ndarray:
    """2-norm normalize the given vector.
    
    Args: 
        vec: non-normalized vector
    Returns:
        normalized vector
    """
    return np.asarray(vec) / np.linalg.norm(np.asarray(vec))


class Frame:
    """A geometric Frame.

    Defined via a translation and rotation transformation from a unit world frame.
    """

    def __init__(
        self, rotation_matrix: RotationMatrixLike, translation_vector: VectorLike
    ) -> None:
        """Frame (transformation) constructor.

        Basic constructor method of a frame object. 
        The arguments rot and trans are taken as 
        the rotation matrix and translation vector of
        a frame-to-frame transformation. 

        Args: 
            rotation_matrix: 3x3 orthogonal rotation matrix
            translation_vector: 3x1 or 1x3 translation vector
        """
        self._rot: np.ndarray = np.array(rotation_matrix)
        self._trans: np.ndarray = np.array(translation_vector)
        assert self._rot.shape == (
            3,
            3,
        ), "Rotation matrix does not have the required shape of (3,3)."
        assert self._trans.shape == (
            3,
        ), "Translation vector does not have the required shape of (1,3) or (3,1)."

    def __str__(self) -> str:
        # basic string representation of a frame
        s = ""
        s += "rotation\n{}".format(self._rot)
        s += "\nEuler angles (xyz, extrinsic, deg.)\n{}".format(
            self.euler_angles("xyz", degrees=True)
        )
        s += "\nEuler angles (XYZ, intrinsic, deg.)\n{}".format(
            self.euler_angles("XYZ", degrees=True)
        )
        s += "\ntranslation\n{}".format(self._trans)
        return "<%s instance at %s>\n%s" % (self.__class__.__name__, id(self), s)

    def _repr_html_(self) -> str:
        # html representation of a frame
        html = (
            """
            <table>
                <tr>
                    <th>rotation matrix</th>
                    <th>Euler angles<br>(xyz, extr., deg.)</th>
                    <th>Euler angles<br>(XYZ, intr., deg.)</th>
                    <th>translation<br></th>
                </tr>
                <tr><td>"""
            + html_table_from_matrix(self._rot)
            + "</td><td>"
            + html_table_from_vector(
                self.euler_angles("xyz", degrees=True), indices=["θx", "θy", "θz"]
            )
            + "</td><td>"
            + html_table_from_vector(
                self.euler_angles("XYZ", degrees=True), indices=["θx", "θy", "θz"]
            )
            + "</td><td>"
            + html_table_from_vector(self._trans, indices=["x", "y", "z"])
            + "</td></tr></table>"
        )
        return html

    def SA_pastable_string(self) -> str:
        """ Spatial Analyzer compatible string representation
            
        Returns:
            SA compatible flattened 4x4 transformation matrix
        """
        p = np.eye(4)
        p[0:3, 0:3] = self._rot
        p[0:3, 3] = self._trans
        return " ".join(["{:0.12f}".format(i) for i in p.flatten()])

    def euler_angles(self, *args, **kwargs) -> np.ndarray:
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
        return R.from_matrix(self._rot).as_euler(*args, **kwargs)

    def extrinsic_euler_angles(self) -> Tuple[float, float, float]:
        """Extrinsic xyz Euler angles (fixed rotation reference axes) of the Frame.

        Returns:
           Rotation angles around extrinsic x,y,z axes (degrees)
        """
        return self.euler_angles("xyz", degrees=True)

    def intrinsic_euler_angles(self) -> Tuple[float, float, float]:
        """Intrinsic xyz Euler angles of the Frame.

        Returns:
           Rotation angles around intrinsic x,y,z axes (degrees)
        """
        return self.euler_angles("XYZ", degrees=True)

    @property
    def translation(self) -> Vector:
        """Frame translation vector.

        Returns: 
            Frame translation vector
        """
        return Vector(self._trans)

    @property
    def rotation(self) -> RotationMatrix:
        """Frame rotation matrix.

        Returns: 
            Frame rotation matrix
        """
        return RotationMatrix(self._rot)

    def express_in_frame(self, reference_frame: Frame) -> Frame:
        """Express this frame in a different frame.

        This is the transformation T0 between `reference_frame` and `self` expressed in reference_frame.
        Transformation T0 such that T0*reference_frame = self -> T0=self*inv(reference_frame).
        T0 expressed in reference_frame becomes: T = inv(reference_frame) T0 reference_frame = inv(reference_frame) self.

        This is equivalent to a transformation T between `reference_frame` and `self` such that 
        vA = (T.rotation).vB + T.translation
        where vA, vB represent the same vector expressed in reference_frame and self, respectively:
        reference_frame.rotation * vA + reference_frame.translation = self.rotation * vB + self.translation
        
        Args:
            reference_frame: Reference frame to express this frame in. 

        Returns:
            Frame expressed in a new reference frame
        """
        return express_frame_in_frame(self, reference_frame)

    @classmethod
    def create_unit_frame(cls) -> Frame:
        """Construct unit frame.

        Construct transformation frame with no rotation and translation.

        Returns:
            New unit frame object
        """
        return Frame(np.identity(3), np.zeros(3))

    @classmethod
    def from_SA_pastable_string(cls, SA_string: str) -> Frame:
        """Construct frame from SA transformation matrix string.

        Args:
            SA_string: transformation matrix string from SA
        Returns:
            New frame object
        """
        try:
            a = np.asarray([float(s) for s in SA_string.split(" ", 15)]).reshape((4, 4))
            rot = a[0:3, 0:3]
            trans = a[:3, 3]
        except:
            raise Exception("SA string could not be read.")
        return Frame(rot, trans)

    @classmethod
    def from_extrinsic_euler_and_translations(
        cls,
        theta_x: float,
        theta_y: float,
        theta_z: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> Frame:
        """Frame from extrinsic xyz Euler angles (fixed rotation reference axes) and translations.

        Args:
            theta_x: rotation angle around extrinsic x-axis (degrees)
            theta_y: rotation angle around extrinsic y-axis (degrees)
            theta_z: rotation angle around extrinsic z-axis (degrees)
            dx: translation along x
            dy: translation along y
            dz: translation along z

        Returns:
            Resulting frame
        """
        rot = R.from_euler("xyz", [theta_x, theta_y, theta_z], degrees=True).as_matrix()
        trans = [dx, dy, dz]
        return Frame(rot, trans)

    @classmethod
    def from_intrinsic_euler_and_translations(
        cls,
        theta_x: float,
        theta_y: float,
        theta_z: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> Frame:
        """Frame from intrinsic xyz Euler angles and translations.

        Args:
            theta_x: rotation angle around intrinsic x-axis (degrees)
            theta_y: rotation angle around intrinsic y-axis (degrees)
            theta_z: rotation angle around intrinsic z-axis (degrees)
            dx: translation along x
            dy: translation along y
            dz: translation along z

        Returns:
            Resulting frame
        """
        rot = R.from_euler("XYZ", [theta_x, theta_y, theta_z], degrees=True).as_matrix()
        trans = [dx, dy, dz]
        return Frame(rot, trans)

    @classmethod
    def from_quat_and_translations(
        cls,
        q0: float,
        q1: float,
        q2: float,
        q3: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> Frame:
        """Frame from quaternion components (scalar last) and translations.

        Args:
            q0: quaternion component 0 (x)
            q1: quaternion component 1 (y)
            q2: quaternion component 2 (x)
            q3: quaternion component 3 (scalar)
            dx: translation along x
            dy: translation along y
            dz: translation along z

        Returns:
            Resulting frame
        """
        rot = R.from_quat([q0, q1, q2, q3]).as_matrix()
        trans = [dx, dy, dz]
        return Frame(rot, trans)

    @classmethod
    def from_orthogonal_vectors(
        new_x: VectorLike,
        new_y: VectorLike,
        new_z: VectorLike,
        origin: VectorLike = [0, 0, 0],
    ) -> Frame:
        """Frame from three orthogonal vectors along the x,y,z axes. 

        Args:
            new_x: Vector along the x-axis
            new_y: Vector along the x-axis
            new_z: Vector along the x-axis
            origin: Origin coordinates. Defaults to [0,0,0].

        Returns:
            Frame: Resulting Frame
        """
        rot = np.stack([normalize(new_x), normalize(new_y), normalize(new_z)], 1)
        trans = origin
        return Frame(rot, trans)


class Vector:
    """A Vector is a container for one set of dX,dY,dZ deltas.

    It is only subject to the rotational part of frame transformations. It is not affected by translations. 
    """

    def __init__(self, v: VectorLike):
        """Initialize a vector from a sequence of dX,dY,dZ deltas. 

        Args:
            v: A sequence of dX,dY,dZ deltas
        """
        self._a: np.ndarray = np.array(v)  # storage as Numpy array

    def express_in_frame(
        self, new_frame: Frame, original_frame: Frame = Frame.create_unit_frame()
    ) -> Vector:
        """Express this vector in a different frame.

        Express the vector given in the frame `original_frame` in a different frame `new_frame`.

        Args:
            new_frame: Frame to express this vector in. 
            original_frame: Reference frame where the vector is specified in. Defaults to Frame.create_unit_frame().

        Returns:
            Vector expressed in `new_frame`.
        """
        return express_vector_in_frame(self._a, new_frame, original_frame)

    def __str__(self) -> str:
        # basic string representation
        return "<%s instance at %s> %s" % (self.__class__.__name__, id(self), self._a)

    def _repr_html_(self):
        html = html_table_from_vector(self._a, indices=["x", "y", "z"])
        return html

    def as_array(self):
        return self._a

    def __array__(self):
        return self._a

    def __getitem__(self, key):
        return self._a[key]

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self._a + other._a)
        elif isinstance(other, Vector):
            return Vector(self._a + other._a)
        else:
            return self._a + other

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self._a - other._a)
        elif isinstance(other, Vector):
            return Vector(self._a - other._a)
        else:
            return self._a - other

    def normalize(self) -> Vector:
        """Normalize the length of this vector to 1.

        Returns:
            Normalized vector
        """
        return Vector(normalize(self._a))

    def length(self) -> float:
        """Length of the vector

        Returns:
            The 2-norm length of the vector.
        """
        return np.linalg.norm(self._a)

    def transform(self, transformation: Frame) -> Vector:
        """Transform this vector by a given transformation frame.

        Apply a transformation to a vector (rotate it), and express it still in the original frame. Basically the inverse of "express vector in frame".

        Args:
            transformation: Transformation frame

        Returns:
            vector expressed in the original frame, but transformed.
        """
        
        #return rotate_vector(self._a, transformation._rot)
        return Vector(transformation._rot @ self._a)

    def __matmul__(self, other: Union[VectorLike, RotationMatrixLike]) -> float:
        return np.dot(self._a, np.asarray(other))

    def __rmatmul__(self, other: Union[VectorLike, RotationMatrixLike]) -> float:
        return np.dot(np.asarray(other), self._a)

    def __mul__(self, other: float) -> Vector:
        return Vector(self._a * other)

    __rmul__ = __mul__


class Point:
    """A Point is a container for one set of X,Y,Z coordinates.

    It is subject to translations and rotations of frame transformations.
    """

    def __init__(self, p: VectorLike):
        """Initialize a Point from a sequence of X,Y,Z coordinates.

        Args:
            p: sequence of X,Y,Z coordinates
        """
        self._a: np.ndarray = np.array(p)  # storage as Numpy array

    def express_in_frame(
        self, new_frame, original_frame: Frame = Frame.create_unit_frame()
    ) -> Point:
        """Express this point in a different frame.

        Express the point given in the frame `original_frame` in a different frame `new_frame`.

        Args:
            new_frame: Frame to express this point in. 
            original_frame: Reference frame where the point is specified in. Defaults to Frame.create_unit_frame().

        Returns:
            Point expressed in `new_frame`.
        """
        return express_point_in_frame(self._a, new_frame, original_frame)

    def __str__(self) -> str:
        # basic string representation
        return "<%s instance at %s> %s" % (self.__class__.__name__, id(self), self._a)

    def _repr_html_(self):
        html = html_table_from_vector(self._a, indices=["x", "y", "z"])
        return html

    def as_array(self):
        return self._a

    def __array__(self):
        return self._a

    def __getitem__(self, key):
        return self._a[key]

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self._a + other._a)
        elif isinstance(other, Vector):
            return Point(self._a + other._a)
        else:
            return self._a + other

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Point):
            return Vector(self._a - other._a)
        elif isinstance(other, Vector):
            return Point(self._a - other._a)
        else:
            return self._a - other

    __rsub__ = __sub__

    def __matmul__(self, other: Union[VectorLike, RotationMatrixLike]) -> float:
        return np.dot(self._a, np.asarray(other))

    def __rmatmul__(self, other: Union[VectorLike, RotationMatrixLike]) -> float:
        return np.dot(np.asarray(other), self._a)

    def transform(self, transformation: Frame) -> Point:
        """Transform this point by a given transformation frame.

        Apply a transformation to a point (move it), and express it still in the original frame. Basically the inverse of "express point in frame".

        Args:
            transformation: Transformation frame 

        Returns:
            Point expressed in the original frame but transformed.
        """
        #return Point(transform_points(self, transformation))
        return Point(transformation._rot@self._a + transformation._trans)


class RotationMatrix:
    """A 3x3 rotation matrix.

    The rotation matrix must be orthogonal. This is not enforced in the initializer. 
    """

    def __init__(self, m: RotationMatrixLike):
        """Initialize a RotationMatrix from any type of 3x3 construct (sequences, np.ndarray, RotationMatrix).

        Args:
            m: Rotation-matrix-like object
        """
        self._a: np.ndarray = np.array(m)  # storage as Numpy array

    def __str__(self) -> str:
        # basic string representation
        return "<%s instance at %s>\n%s" % (self.__class__.__name__, id(self), self._a)

    def _repr_html_(self):
        html = html_table_from_matrix(self._a)
        return html

    def as_array(self):
        return self._a

    def __array__(self):
        return self._a

    def __getitem__(self, key):
        return self._a[key]

    @classmethod
    def from_euler_angles(
        cls, seq: str, angles: Sequence[float], degrees: bool = False
    ) -> RotationMatrix:
        """Rotation matrix from Euler angles.
        
        Arguments are passed to scipy.spatial.transform.Rotation.from_euler(*args, **kwargs).

        Args:
            seq: Specifies sequence of axes for rotations. Up to 3 characters
                belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
                {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
                rotations cannot be mixed in one function call.
            angles: float or array_like, shape (N,) or (N, [1 or 2 or 3])
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
            degrees: If True, then the given angles are assumed to be in degrees.
                Defaults to False.
            
        Returns:
            Rotation matrix
        """
        return cls(R.from_euler(seq, angles, degrees=degrees).as_matrix())


def frame_wizard(
    primary_vec: VectorLike,
    secondary_vec: VectorLike,
    primary_axis: str,
    secondary_axis: str,
    origin: VectorLike = [0, 0, 0],
) -> Frame:
    """Frame-Wizard-type Frame constructor.

    This constructor of a Frame object works analogously to the Spatial Analyzer Frame Wizard.
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

    Returns:
        Constructed Frame
    """
    assert (
        secondary_axis != primary_axis
    ), "secondary axis must not equal primary axis: choose from x,y,z"
    primary_vec = normalize(primary_vec)
    secondary_vec = normalize(secondary_vec)
    rot = np.zeros((3, 3))
    column_dict = {"x": 0, "y": 1, "z": 2}
    primary_index = column_dict.pop(primary_axis)
    secondary_index = column_dict.pop(secondary_axis)
    tertiary_index = list(column_dict.values())[0]
    axes = [primary_index, secondary_index]
    if axes in ([0, 1], [1, 2], [2, 0]):
        signature_perm = 1
    else:
        signature_perm = -1

    # set primary axis in rotation matrix
    rot[:, primary_index] = primary_vec

    # construct tertiary and secondary axis
    rot[:, tertiary_index] = signature_perm * normalize(
        np.cross(primary_vec, secondary_vec)
    )
    rot[:, secondary_index] = -signature_perm * np.cross(
        primary_vec, rot[:, tertiary_index]
    )

    return Frame(rot, origin)


def transformation_between_frames(frameA: Frame, frameB: Frame) -> Frame:
    """Transformation between frameA and frameB.
    
    Transformation between frameA and frameB, expressed in unit frame.
    Construct transformation T between `frameA` and `frameB` 
    such that T*frameA = B -> T=frameB*inv(frameA)
    
    Args:
        frameA: Reference frame.
        frameB: Final frame.

    Returns:
        Transformation between frameA and frameB
    """
    Trot = frameB._rot.dot(np.transpose(frameA._rot))
    Ttrans = frameB._trans - frameA._trans
    return Frame(Trot, Ttrans)


def express_frame_in_frame(input_frame: Frame, reference_frame: Frame):
    """Express input_frame (input_frame) in reference_frame (frameA).

    This is the transformation T0 between `reference_frame` and `input_frame` expressed in reference_frame.
    Transformation T0 such that T0*reference_frame = B -> T0=input_frame*inv(reference_frame).
    T0 expressed in reference_frame becomes: T = inv(reference_frame) T0 reference_frame = inv(reference_frame) input_frame.

    This is equivalent to a transformation T between `reference_frame` and `input_frame` such that 
    vA = (T.rotation).vB + T.translation
    where vA, vB represent the same vector expressed in reference_frame and input_frame, respectively:
    reference_frame.rotation * vA + reference_frame.translation = input_frame.rotation * vB + input_frame.translation
    
    Args:
        input_frame: Input frame.
        reference_frame: Reference frame in which input frame should be expressed.

    Returns:
        Input frame expressed in reference frame
    """
    Trot = np.transpose(reference_frame._rot).dot(input_frame._rot)
    Ttrans = (input_frame._trans - reference_frame._trans) @ reference_frame._rot
    return Frame(Trot, Ttrans)


def express_point_in_frame(
    point: VectorLike,
    new_frame: Frame,
    original_frame: Frame = Frame.create_unit_frame(),
) -> Point:
    """Express a point in a different frame.

    Express the `point` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        point: 3x1 point object
        new_frame: Frame to express this point in. 
        original_frame: Reference frame where the point is specified in. Defaults to UnitFrame.

    Returns:
        Point expressed in `new_frame`.
    """
    trafo = transformation_between_frames(original_frame, new_frame)
    return Point(
        (np.asarray(point) - trafo._trans) @ trafo._rot
    )  # multiplication to the right is the same as with transpose to the left


def express_points_in_frame(
    points: Sequence[VectorLike],
    new_frame: Frame,
    original_frame: Frame = Frame.create_unit_frame(),
) -> Sequence[VectorLike]:
    """Express points in a different frame.

    Express the `points` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        points: Sequence of points.
        new_frame: Frame to express this point in. 
        original_frame: Reference frame where the point is specified in. Defaults to UnitFrame.

    Returns:
        Points expressed in `new_frame`.
    """
    trafo = transformation_between_frames(original_frame, new_frame)
    return (
        np.asarray(points) - trafo._trans
    ) @ trafo._rot  # multiplication to the right is the same as with transpose to the left


def express_vector_in_frame(
    vector: VectorLike,
    new_frame: Frame,
    original_frame: Frame = Frame.create_unit_frame(),
) -> Vector:
    """Express a vector in a different frame.

    Express the `vector` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        vector: 3x1 vector object
        new_frame: Frame to express this vector in. 
        original_frame: Reference frame where the vector is specified in.

    Returns:
        Vector expressed in `new_frame`.
    """
    trafo = transformation_between_frames(original_frame, new_frame)
    return Vector(np.asarray(vector) @ trafo._rot)


def rotate_vector(vec: VectorLike, rot: RotationMatrixLike) -> Vector:
    """Rotate vector using a given rotation matrix.

    Args:
        vec: The input vector.
        rot: The rotation matrix.

    Returns:
        The rotated vector.
    """
    return Vector(np.asarray(rot) @ np.asarray(vec))


def transform_points(
    points: Union[VectorLike, Sequence[VectorLike]], trafo: Frame
) -> np.ndarray:
    return np.dot(np.asarray(points), trafo._rot.T) + trafo._trans
    # rotation can also be written as `np.einsum('ij,kj->ki', t0._rot, np.asarray(points))`


def distance_between_points(pointA: VectorLike, pointB: VectorLike) -> float:
    return np.linalg.norm(np.asarray(pointA) - np.asarray(pointB))


def minimize_points_to_points_distance(
    groupA, groupB, return_report=False, method="Powell", tol=1e-6
):
    """Transform point group to minimize point-group-to-point-group distance.

    Returns a transformation (Frame object) that, if applied to all points in point group
    `groupA`, minimizes the distance between all points in `groupA` an the corresponding 
    points in `groupB`.
    
    Args:
        groupA: Array of Points.
        groupB: Array of Points (same size as groupA).
        return_report: True if report of minimization algorithm should be returned

    Returns:
        Transformation, or tuple of transformation and minimization report if return_report==True
    """
    # return transform that maps groupA onto groupB with minimum point-to-point distance
    def cost(x):
        [r1, r2, r3, t1, t2, t3] = x
        rot = R.from_rotvec([r1, r2, r3]).as_matrix()
        trans = np.asarray([t1, t2, t3])
        t = Frame(rot, trans)
        c = np.sqrt(
            np.mean(
                np.power(
                    [
                        distance_between_points(pB, Point(pA).transform(t))
                        for (pA, pB) in zip(groupA, groupB)
                    ],
                    2,
                )
            )
        )
        return c

    m = minimize(cost, [0, 0, 0, 0, 0, 0], tol=tol, method=method)
    t = Frame(R.from_rotvec(m["x"][:3]).as_matrix(), m["x"][3:])
    if return_report:
        return t, m
    else:
        return t
