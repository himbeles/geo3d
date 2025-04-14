from __future__ import annotations

from typing import Optional, Union, overload

import numpy as np
import numpy.typing as npt
from numba import njit

from . import frame, vector
from .auxiliary import html_table_from_vector
from .linalg import (
    add_vec_vec,
    cast_vec_to_array,
    mult_mat_vec,
    mult_vec_mat,
    sub_vec_vec,
)
from .rotation import RotationMatrixLike
from .types import MultipleVectorLike, VectorLike, VectorTuple


class Point:
    """A Point is a container for one set of X,Y,Z coordinates.

    It is subject to translations and rotations of frame transformations.
    """

    def __init__(self, p: VectorLike):
        """Initialize a Point from a sequence of X,Y,Z coordinates.

        Args:
            p: sequence of X,Y,Z coordinates
        """
        self._a: npt.NDArray[np.float64] = np.array(p)  # storage as Numpy array

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
            obj._a = a  # storage as numpy array passed by reference
        return obj

    def express_in_frame(
        self, new_frame, original_frame: Optional[frame.Frame] = None
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

    def __array__(self, copy=None):
        if copy in [None, False]:
            return self._a
        else :
            return self._a.copy()

    def __getitem__(self, key: int):
        return self._a[key]

    @overload
    def __add__(self, other: Point) -> Point:
        ...

    @overload
    def __add__(self, other: vector.Vector) -> Point:
        ...

    def __add__(self, other: VectorLike) -> Union[Point, np.ndarray]:
        if isinstance(other, Point):
            return Point(self._a + other._a)
        elif isinstance(other, vector.Vector):
            return Point(self._a + other._a)
        else:
            return self._a + np.array(other)

    __radd__ = __add__

    @overload
    def __sub__(self, other: Point) -> vector.Vector:
        ...

    @overload
    def __sub__(self, other: vector.Vector) -> Point:
        ...

    def __sub__(self, other: VectorLike) -> Union[vector.Vector, Point, np.ndarray]:
        if isinstance(other, Point):
            return vector.Vector(self._a - other._a)
        elif isinstance(other, vector.Vector):
            return Point(self._a - other._a)
        else:
            return self._a - np.array(other)

    __rsub__ = __sub__

    def __matmul__(
        self, other: Union[VectorLike, RotationMatrixLike]
    ) -> Union[float, np.ndarray]:
        return np.dot(self._a, np.asarray(other))

    def __rmatmul__(self, other: Union[VectorLike, RotationMatrixLike]) -> float:
        return np.dot(np.asarray(other), self._a)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, self.__class__):
            return False
        return np.allclose(self._a, o._a, rtol=1e-10)

    def transform(self, transformation: frame.Frame) -> Point:
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


def express_point_in_frame(
    point: VectorLike,
    new_frame: frame.Frame,
    original_frame: Optional[frame.Frame] = None,
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
        new_frame_in_orig_frame = new_frame
    else:
        new_frame_in_orig_frame = frame.express_frame_in_frame(
            new_frame, original_frame
        )

    return Point(
        _express_point_bare(
            new_frame_in_orig_frame._rot, new_frame_in_orig_frame._trans, point
        )
    )


def express_points_in_frame(
    points: MultipleVectorLike,
    new_frame: frame.Frame,
    original_frame: Optional[frame.Frame] = None,
) -> np.ndarray:
    """Express points in a different frame.

    Express the `points` given in the frame `original_frame` in a different frame `new_frame`.

    Args:
        points: Sequence of points.
        new_frame: Frame to express these points in.
        original_frame: Reference frame where the point are specified in. Defaults to UnitFrame.

    Returns:
        Points expressed in `new_frame`.
    """
    if original_frame is None:
        new_frame_in_orig_frame = new_frame
    else:
        new_frame_in_orig_frame = frame.express_frame_in_frame(
            new_frame, original_frame
        )
    return _express_points_bare(
        new_frame_in_orig_frame._rot, new_frame_in_orig_frame._trans, points
    )


@njit
def transform_point(rot, trans, p):
    return cast_vec_to_array(_transform_point_bare(rot, trans, p))


@njit
def _transform_point_bare(rot, trans, p):
    return add_vec_vec(mult_mat_vec(rot, p), trans)


@njit
def _express_point_bare(rot, trans, p) -> VectorTuple:
    return mult_vec_mat(sub_vec_vec(p, trans), rot)


@njit
def _express_points_bare(rot, trans, points):
    dim = len(points)
    res = np.empty((dim, 3))
    for i in range(dim):
        res[i] = _express_point_bare(rot, trans, points[i])
    return np.asarray(res)


def transform_points(points: MultipleVectorLike, trafo: frame.Frame) -> np.ndarray:
    return np.dot(np.asarray(points), trafo._rot.T) + trafo._trans
    # rotation can also be written as `np.einsum('ij,kj->ki', t0._rot, np.asarray(points))`
