from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from .linalg import dot_vec_vec
from .point import Point
from .vector import Vector, normalized_vector


@dataclass
class Plane:
    normal: Vector
    point: Point

    def as_abcd(self) -> Tuple[float, float, float, float]:
        """ABCD components of the plane

        Defining the plane via
        a*x + b*y + c*z + d = 0

        Returns:
            tuple (a,b,c,d)
        """
        n: npt.NDArray[np.float64] = normalized_vector(self.normal.as_array())
        return (n[0], n[1], n[2], -1 * dot_vec_vec(n, self.point.as_array()))
