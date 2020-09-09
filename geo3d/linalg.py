from numba import njit
import numpy as np
from math import sqrt

@njit
def add_vec_vec(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


@njit
def mult_vec_sca(v, s):
    return (v[0] * s, v[1] * s, v[2] * s)


@njit
def dot_vec_vec(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


@njit
def mult_mat_vec(m, v):
    return (dot_vec_vec(m[0], v), dot_vec_vec(m[1], v), dot_vec_vec(m[2], v))

@njit
def cast_vec_to_array(vec):
    a = np.empty(3)
    a[0] = vec[0]
    a[1] = vec[1]
    a[2] = vec[2]
    return a

@njit
def norm_L2(vec) -> float:
    s = 0
    for v in vec:
        s += v ** 2
    return sqrt(s)

