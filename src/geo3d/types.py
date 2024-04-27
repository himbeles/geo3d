from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .point import Point
    from .vector import Vector

VectorTuple = Tuple[float, float, float]
VectorLike = Union[Sequence[float], VectorTuple, npt.ArrayLike, "Vector", "Point"]
MultipleVectorLike = Union[Sequence[VectorLike], npt.NDArray[np.float64]]
