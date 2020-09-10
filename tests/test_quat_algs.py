from geo3d import R, normalized_quat, matrix_as_quat, quat_as_matrix
import numpy as np


def test_normalized_quat():
    quat = np.array([1,2,3,5])
    quatn = np.array(normalized_quat(quat))
    assert np.allclose(quatn, quat/np.linalg.norm(quat))

def test_quat_as_matrix():
    quat = np.array([1,2,3,5])
    quat = quat / np.linalg.norm(quat)
    rot = R.from_quat(quat)
    matrix1 = rot.as_matrix()
    matrix2 = quat_as_matrix(quat)
    assert np.allclose(matrix1, matrix2) 

def test_matrix_as_quat():
    quat1 = np.array([1,2,3,5])
    quat1 = quat1 / np.linalg.norm(quat1)
    rot = R.from_quat(quat1)
    matrix = rot.as_matrix()
    quat2 = matrix_as_quat(matrix)
    assert np.allclose(quat1, quat2) 