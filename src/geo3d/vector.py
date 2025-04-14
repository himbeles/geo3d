from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np
import numpy.typing as npt
from numba import njit

from . import frame, point
from .auxiliary import html_table_from_vector
from .linalg import cast_vec_to_array, mult_mat_vec, mult_vec_mat, norm_L2
from .rotation import RotationMatrixLike
from .types import MultipleVectorLike, VectorLike, VectorTuple


class Vector:
    """A Vector is a container for one set of dX,dY,dZ deltas.

    It is only subject to the rotational part of frame transformations. It is not affected by translations.
    """

    def __init__(self, v: VectorLike):
        """Initialize a vector from a sequence of dX,dY,dZ deltas.

        Args:
            v: A sequence of dX,dY,dZ deltas
        """
        self._a: npt.NDArray[np.float64] = np.array(v)  # storage as Numpy array

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
        self, new_frame: frame.Frame, original_frame: Optional[frame.Frame] = None
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

    def __array__(self, copy=None):
        if copy in [None, False]:
            return self._a
        else :
            return self._a.copy()

    def __getitem__(self, key):
        return self._a[key]

    @overload
    def __add__(self, other: point.Point) -> point.Point:
        ...

    @overload
    def __add__(self, other: Vector) -> Vector:
        ...

    def __add__(self, other: VectorLike):
        if isinstance(other, point.Point):
            return point.Point(self._a + other._a)
        elif isinstance(other, Vector):
            return Vector(self._a + other._a)
        else:
            return self._a + np.array(other)

    __radd__ = __add__

    @overload
    def __sub__(self, other: point.Point) -> point.Point:
        ...

    @overload
    def __sub__(self, other: Vector) -> Vector:
        ...

    def __sub__(self, other: VectorLike):
        if isinstance(other, point.Point):
            return point.Point(self._a - other._a)
        elif isinstance(other, Vector):
            return Vector(self._a - other._a)
        else:
            return self._a - np.array(other)

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

    def transform(self, transformation: frame.Frame) -> Vector:
        """Transform this vector by a given transformation frame.

        Apply a transformation to a vector (rotate it), and express it still in the original frame.
        This performs the inverse operation to a vector compared to `express_vector_in_frame`.

        Args:
            transformation: Transformation frame

        Returns:
            vector expressed in the original frame, but transformed by `transformation`.
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


def express_vectors_in_frame(
    vectors: MultipleVectorLike,
    new_frame: frame.Frame,
    original_frame: Optional[frame.Frame] = None,
) -> np.ndarray:
    """Express vectors in a different frame.

    Express the `vectors` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        vectors: Sequence of vectors.
        new_frame: Frame to express these vectors in.
        original_frame: Reference frame where the vectors are specified in. Defaults to UnitFrame.

    Returns:
        Vectors expressed in `new_frame`.
    """
    if original_frame is None:
        new_frame_in_orig_frame = new_frame
    else:
        new_frame_in_orig_frame = frame.express_frame_in_frame(
            new_frame, original_frame
        )
    return _express_vectors_bare(new_frame_in_orig_frame._rot, vectors)


def express_vector_in_frame(
    vector: VectorLike,
    new_frame: frame.Frame,
    original_frame: Optional[frame.Frame] = None,
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
        new_frame_in_orig_frame = new_frame
    else:
        new_frame_in_orig_frame = frame.express_frame_in_frame(
            new_frame, original_frame
        )

    return Vector(_express_vector_bare(new_frame_in_orig_frame._rot, vector))


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
def _express_vector_bare(rot, v) -> VectorTuple:
    return mult_vec_mat(v, rot)


@njit
def _express_vectors_bare(rot, vectors):
    dim = len(vectors)
    res = np.empty((dim, 3))
    for i in range(dim):
        res[i] = _express_vector_bare(rot, vectors[i])
    return np.asarray(res)


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
