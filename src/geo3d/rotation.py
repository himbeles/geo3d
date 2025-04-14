from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from .auxiliary import html_table_from_matrix

RotationMatrixLike = Union[Sequence[Sequence[float]], npt.ArrayLike, "RotationMatrix"]


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

    def __array__(self, copy=None):
        if copy in [None, False]:
            return self._a
        else :
            return self._a.copy()

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
