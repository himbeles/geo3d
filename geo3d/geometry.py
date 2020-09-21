from __future__ import annotations
from .linalg import (
    add_vec_vec,
    cast_vec_to_array,
    cross_vec_vec,
    dot_vec_vec,
    mult_mat_vec,
    mult_vec_mat,
    mult_vec_sca,
    norm_L2,
    sub_vec_vec,
)
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from numba import njit
from .auxiliary import html_table_from_matrix, html_table_from_vector
from typing import Union, Tuple, Sequence, Optional
from dataclasses import dataclass

RotationMatrixLike = Union[Sequence[Sequence[float]], np.ndarray, "RotationMatrix"]
VectorLike = Union[Sequence[float], np.ndarray, "Vector", "Point"]
QuaternionTuple = Tuple[float, float, float, float]


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
        s += "\nFixed angles (xyz, extrinsic, deg.)\n{}".format(
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
                    <th>Fixed angles<br>(xyz, extr., deg.)</th>
                    <th>Euler angles<br>(xyz, intr., deg.)</th>
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
        """Spatial Analyzer compatible string representation

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

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            return False
        eq_r = np.allclose(self._rot, o._rot, rtol=1e-10)
        eq_t = np.allclose(self._trans, o._trans, rtol=1e-10)
        return eq_r and eq_t

    @classmethod
    def create_unit_frame(cls) -> Frame:
        """Construct unit frame.

        Construct transformation frame with no rotation and translation.

        Returns:
            New unit frame object
        """
        return cls(np.identity(3), np.zeros(3))

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
        return cls(rot, trans)

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
        quat = euler_as_quat(
            theta_x / 180 * math.pi,
            theta_y / 180 * math.pi,
            theta_z / 180 * math.pi,
            intrinsic=False,
        )
        return cls.from_quat_and_translations(*quat, dx, dy, dz)

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
        quat = euler_as_quat(
            theta_x / 180 * math.pi,
            theta_y / 180 * math.pi,
            theta_z / 180 * math.pi,
            intrinsic=True,
        )
        return cls.from_quat_and_translations(*quat, dx, dy, dz)

    @classmethod
    def _from_quat_and_translations_scipy(
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
        return cls(rot, trans)

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
        quat = normalized_quat((q0, q1, q2, q3))
        rot = quat_as_matrix(quat)
        trans = [dx, dy, dz]
        return cls(rot, trans)

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
        rot = np.stack(
            [
                normalized_vector(np.array(new_x)),
                normalized_vector(np.array(new_x)),
                normalized_vector(np.array(new_x)),
            ],
            1,
        )
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

    @classmethod
    def from_array(cls, a: np.ndarray, copy=True):
        """Initialize a vector from a numpy array of dX,dY,dZ deltas.

        Args:
            a: A np.array containing dX,dY,dZ deltas
        """
        obj = cls.__new__(cls)
        if copy:
            obj._a = a.copy()  # storage as Numpy array
        else:
            obj._a = a
        return obj

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

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "<%s at %s> %s" % ("Vector", id(self), self._a)

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
        return Vector.from_array(normalized_vector(self._a), copy=False)

    def length(self) -> float:
        """Length of the vector

        Returns:
            The 2-norm length of the vector.
        """
        return norm_L2(self._a)

    def transform(self, transformation: Frame) -> Vector:
        """Transform this vector by a given transformation frame.

        Apply a transformation to a vector (rotate it), and express it still in the original frame. Basically the inverse of "express vector in frame".

        Args:
            transformation: Transformation frame

        Returns:
            vector expressed in the original frame, but transformed.
        """

        # return rotate_vector(self._a, transformation._rot)
        # return Vector.from_array(transformation._rot @ self._a, copy=False)
        return Vector.from_array(
            transform_vector(transformation._rot, self._a), copy=False
        )

    def __matmul__(self, other: Union[VectorLike, RotationMatrixLike]) -> float:
        return np.dot(self._a, np.asarray(other))

    def __rmatmul__(self, other: Union[VectorLike, RotationMatrixLike]) -> float:
        return np.dot(np.asarray(other), self._a)

    def __mul__(self, other: float) -> Vector:
        return Vector(self._a * other)

    __rmul__ = __mul__

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            return False
        return np.allclose(self._a, o._a, rtol=1e-10)


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

    @classmethod
    def from_array(cls, a: np.ndarray, copy=True):
        """Initialize a Point from a numpy array of of X,Y,Z coordinate.

        Args:
            a: A np.array containing of X,Y,Z coordinate
            copy: Pass array by value or reference
        """
        obj = cls.__new__(cls)
        if copy:
            obj._a = a.copy()  # storage as copied numpy array
        else:
            obj._a = a  # storage as copied numpy array passed by reference
        return obj

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

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "<%s at %s> %s" % ("Point", id(self), self._a)

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

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            return False
        return np.allclose(self._a, o._a, rtol=1e-10)

    def transform(self, transformation: Frame) -> Point:
        """Transform this point by a given transformation frame.

        Apply a transformation to a point (move it), and express it still in the original frame. Basically the inverse of "express point in frame".

        Args:
            transformation: Transformation frame

        Returns:
            Point expressed in the original frame but transformed.
        """
        # return Point(transform_points(self, transformation))
        # return Point.from_array(
        #     transformation._rot @ self._a + transformation._trans, copy=False
        # )
        return Point.from_array(
            transform_point(transformation._rot, transformation._trans, self._a),
            copy=False,
        )


@dataclass
class Plane:
    normal: Vector
    point: Point


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

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            return False
        return np.allclose(self._a, o._a, rtol=1e-10)


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
    primary_vec = normalized_vector(np.array(primary_vec))
    secondary_vec = normalized_vector(np.array(secondary_vec))
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
    rot[:, tertiary_index] = signature_perm * normalized_vector(
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
    original_frame: Optional[Frame] = None,
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
    if original_frame is None:
        return Point.from_array(
            _express_point_bare(new_frame._rot, new_frame._trans, point), copy=False
        )
    else:
        trafo = transformation_between_frames(original_frame, new_frame)
        return Point(
            (np.asarray(point) - trafo._trans) @ trafo._rot
        )  # multiplication to the right is the same as with transpose to the left


def express_points_in_frame(
    points: Sequence[VectorLike],
    new_frame: Frame,
    original_frame: Optional[Frame] = None,
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
    if original_frame is None:
        return _express_points_bare(new_frame._rot, new_frame._trans, points)
    else:
        trafo = transformation_between_frames(original_frame, new_frame)
        return (
            np.asarray(points) - trafo._trans
        ) @ trafo._rot  # multiplication to the right is the same as with transpose to the left


def express_vector_in_frame(
    vector: VectorLike,
    new_frame: Frame,
    original_frame: Optional[Frame] = None,
) -> Vector:
    """Express a vector in a different frame.

    Express the `vector` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        vector: 3x1 vector object
        new_frame: Frame to express this vector in.
        original_frame: Reference frame where the vector is specified in (defaults to UnitFrame).

    Returns:
        Vector expressed in `new_frame`.
    """
    if original_frame is None:
        trafo = new_frame
    else:
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


@njit
def transform_vector(rot, vec):
    return cast_vec_to_array(_transform_vector_bare(rot, vec))


@njit
def _transform_vector_bare(rot, vec):
    return mult_mat_vec(rot, vec)


@njit
def transform_point(rot, trans, p):
    return cast_vec_to_array(_transform_point_bare(rot, trans, p))


@njit
def _transform_point_bare(rot, trans, p):
    return add_vec_vec(mult_mat_vec(rot, p), trans)


@njit
def _express_point_bare(rot, trans, p):
    return mult_vec_mat(sub_vec_vec(p, trans), rot)


@njit
def _express_points_bare(rot, trans, points):
    dim = len(points)
    res = np.empty((dim, 3))
    for i in range(dim):
        res[i] = _express_point_bare(rot, trans, points[i])
    return np.asarray(res)


def transform_points(
    points: Union[VectorLike, Sequence[VectorLike]], trafo: Frame
) -> np.ndarray:
    return np.dot(np.asarray(points), trafo._rot.T) + trafo._trans
    # rotation can also be written as `np.einsum('ij,kj->ki', t0._rot, np.asarray(points))`


@njit
def quat_as_matrix(unit_quat):
    """Represent unit quaternion as rotation matrix.

    Method from scipy.spatial.transform.Rotation,
    jit-compiled by numba for speedup.

    Returns
    -------
    matrix : ndarray, shape (3, 3) or (N, 3, 3)
        Shape depends on shape of inputs used for initialization.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    Examples
    --------
    >>> from scipy.spatial.transform import Rotation as R
    Represent a single rotation:
    >>> r = R.from_rotvec([0, 0, np.pi/2])
    >>> r.as_matrix()
    array([[ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    """
    x = unit_quat[0]
    y = unit_quat[1]
    z = unit_quat[2]
    w = unit_quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = np.empty((3, 3))

    matrix[0, 0] = x2 - y2 - z2 + w2
    matrix[1, 0] = 2 * (xy + zw)
    matrix[2, 0] = 2 * (xz - yw)
    matrix[0, 1] = 2 * (xy - zw)
    matrix[1, 1] = -x2 + y2 - z2 + w2
    matrix[2, 1] = 2 * (yz + xw)
    matrix[0, 2] = 2 * (xz + yw)
    matrix[1, 2] = 2 * (yz - xw)
    matrix[2, 2] = -x2 - y2 + z2 + w2

    return matrix


@njit
def matrix_as_quat(matrix: np.ndarray):
    """Rotation quaternion from rotation matrix.

    Method from scipy.spatial.transform.Rotation,
    jit-compiled by numba for speedup.

    Rotations in 3 dimensions can be represented with 3 x 3 proper
    orthogonal matrices [1]_. If the input is not proper orthogonal,
    an approximation is created using the method described in [2]_.

    Args:
    matrix : array_like, shape (N, 3, 3) or (3, 3)
        A single matrix or a stack of matrices, where ``matrix[i]`` is
        the i-th matrix.

    Returns:
    quaternion array with elements
            q0: quaternion component 0 (x)
            q1: quaternion component 1 (y)
            q2: quaternion component 2 (x)
            q3: quaternion component 3 (scalar)

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    .. [2] F. Landis Markley, "Unit Quaternion from Rotation Matrix",
            Journal of guidance, control, and dynamics vol. 31.2, pp.
            440-442, 2008.
    """
    # matrix = np.asarray(matrix, dtype=float)

    decision_matrix = np.empty(4)
    decision_matrix[0] = matrix[0][0]
    decision_matrix[1] = matrix[1][1]
    decision_matrix[2] = matrix[2][2]
    decision_matrix[3] = decision_matrix[:3].sum()

    choice = decision_matrix.argmax()

    quat = np.empty(4)

    if choice != 3:
        i = choice
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[i] = 1 - decision_matrix[3] + 2 * matrix[i, i]
        quat[j] = matrix[j, i] + matrix[i, j]
        quat[k] = matrix[k, i] + matrix[i, k]
        quat[3] = matrix[k, j] - matrix[j, k]

    else:
        quat[0] = matrix[2, 1] - matrix[1, 2]
        quat[1] = matrix[0, 2] - matrix[2, 0]
        quat[2] = matrix[1, 0] - matrix[0, 1]
        quat[3] = 1 + decision_matrix[3]

    return normalized_quat(quat)


@njit
def normalized_vector(vec) -> np.ndarray:
    """Return unit vector array

    by dividing by Euclidean (L2) norm

    Args:
        vec array with elements x,y,z

    Returns:
        array shape (3,) vector divided by L2 norm
    """
    res = np.empty(3)
    n = norm_L2(vec)
    res[0] = vec[0] / n
    res[1] = vec[1] / n
    res[2] = vec[2] / n
    return res


@njit
def normalized_quat(q) -> QuaternionTuple:
    """Return unit quaternion

    by dividing by Euclidean (L2) norm

    Args:
        q array with elements
            q0: quaternion component 0 (x)
            q1: quaternion component 1 (y)
            q2: quaternion component 2 (x)
            q3: quaternion component 3 (scalar)

    Returns:
        array shape (4,) vector divided by L2 norm
    """
    n = norm_L2(q)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


@njit
def elementary_quat(axis_idx, angle) -> QuaternionTuple:
    """Rotation quaternion for an elementary rotation around x, y, or z.

    Args:
        axis_idx: the axis index (0 for x, 1 for y, 2 for z)
        angle: the rotation angle in rad

    Returns:
        rotation unit quaternion
    """
    quat = np.zeros(4)
    quat[axis_idx] = math.sin(angle / 2)
    quat[3] = math.cos(angle / 2)
    return quat


@njit
def compose_quats(p, q) -> QuaternionTuple:
    """Compose rotations expressed by quaternions p,q

    jit-compiled version of scipy _compose_quat

    Args:
        p: First rotation unit quaternion
        q: Second rotation unit quaternion

    Returns:
        composed rotation quaternion
    """
    product = np.empty(4)
    product[3] = p[3] * q[3] - dot_vec_vec(p[:3], q[:3])
    product[:3] = add_vec_vec(
        mult_vec_sca(q[:3], p[3]),
        add_vec_vec(mult_vec_sca(p[:3], q[3]), cross_vec_vec(p[:3], q[:3])),
    )
    return product


@njit
def euler_as_quat(theta_x: float, theta_y: float, theta_z: float, intrinsic=False):
    """Express Euler angles composition as quaternion

    Args:
        theta_x: Rotation angle around x in rad
        theta_y: Rotation angle around y in rad
        theta_z: Rotation angle around z in rad
        intrinsic: Accept intrinsic or extrinsic (fixed) Euler angles as input. Defaults to False.

    Returns:
        Rotation quaternion
    """
    result = elementary_quat(0, theta_x)
    angles = (theta_x, theta_y, theta_z)

    for i in range(1, 3):
        a = angles[i]
        if intrinsic:
            result = compose_quats(result, elementary_quat(i, a))
        else:
            result = compose_quats(elementary_quat(i, a), result)

    return result


@njit
def quat_angle(quat) -> float:
    """Rotation angle of a given quaternion in rad.

    Angle can go from 0 to pi around the rotation axis
    along the first three quaternion entries.

    Args:
        quat: rotation unit quaternion

    Returns:
        rotation angle
    """
    # w > 0 to ensure 0 <= angle <= pi
    sign = (
        -1 if quat[3] < 0 else 1
    )  # flip sign of last quat entry if < 0, meaning |angle|>pi
    return 2 * math.atan2(norm_L2(quat[:3]), sign * quat[3])


@njit
def quat_twist_angle(quat, twist_axis) -> float:
    # https://stackoverflow.com/questions/3684269/component-of-a-quaternion-rotation-around-an-axis
    d = normalized_vector(twist_axis)  # twist axis
    proj = dot_vec_vec(quat[0:3], d)  # quaternion rotation projected onto twist axis
    p = mult_vec_sca(d, proj)

    twist_quat = normalized_quat((p[0], p[1], p[2], quat[3]))

    # invert angle sign when proj is negative
    sign = -1 if proj < 0 else 1
    angle = sign * quat_angle(twist_quat)
    return angle


@njit
def quat_as_rotvec(quat):
    # w > 0 to ensure 0 <= angle <= pi
    sign = -1 if quat[3] < 0 else 1
    q = (quat[0] * sign, quat[1] * sign, quat[2] * sign, quat[3] * sign)

    angle = 2 * math.atan2(norm_L2(q[:3]), q[3])
    scale = angle / math.sin(angle / 2)
    rotvec = mult_vec_sca(q[:3], scale)
    return rotvec
