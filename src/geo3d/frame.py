from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from .quaternion import euler_as_quat, normalized_quat, quat_as_matrix
from .rotation import RotationMatrix, RotationMatrixLike
from .types import VectorLike
from .vector import Vector, normalized_vector

from .auxiliary import html_table_from_matrix, html_table_from_vector


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
        self._rot: npt.NDArray[np.float64] = np.array(rotation_matrix)
        self._trans: npt.NDArray[np.float64] = np.array(translation_vector)
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
        return "\n".join([" ".join([f"{i:0.12f}" for i in l]) for l in p])

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

    def extrinsic_euler_angles(self) -> np.ndarray:
        """Extrinsic xyz Euler angles (fixed rotation reference axes) of the Frame.

        Returns:
           Rotation angles around extrinsic x,y,z axes (degrees)
        """
        return self.euler_angles("xyz", degrees=True)

    def intrinsic_euler_angles(self) -> np.ndarray:
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
        return cls.from_quat_and_translations(
            quat[0], quat[1], quat[2], quat[3], dx, dy, dz
        )

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
        return cls.from_quat_and_translations(
            quat[0], quat[1], quat[2], quat[3], dx, dy, dz
        )

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

    @staticmethod
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
                normalized_vector(np.array(new_y)),
                normalized_vector(np.array(new_z)),
            ],
            1,
        )
        trans = origin
        return Frame(rot, trans)


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
    Trot: npt.NDArray[np.float64] = np.transpose(reference_frame._rot).dot(
        input_frame._rot
    )
    Ttrans = (input_frame._trans - reference_frame._trans) @ reference_frame._rot
    return Frame(Trot, Ttrans)
