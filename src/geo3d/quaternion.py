import math
from typing import Tuple

import numpy as np
from numba import njit

from . import vector
from .linalg import add_vec_vec, cross_vec_vec, dot_vec_vec, mult_vec_sca, norm_L2

QuaternionTuple = Tuple[float, float, float, float]


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
def elementary_quat(axis_idx, angle) -> np.ndarray:
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
def compose_quats(p, q) -> np.ndarray:
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
def euler_as_quat(
    theta_x: float, theta_y: float, theta_z: float, intrinsic=False
) -> np.ndarray:
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
    d = vector.normalized_vector(twist_axis)  # twist axis
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
