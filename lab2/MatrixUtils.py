import numpy as np
from MatrixSolver import MatrixSolver
import math

def norm_matrix(matrix):
    abs_matrix = np.absolute(matrix)
    norm = abs_matrix[abs_matrix.sum(axis=1).argmax()].sum()
    return norm


def norm_vec(vec):
    return np.absolute(vec).max()


def frobenius_norm_vec(vec):
    norm = 0
    for i in range(vec.shape[0]):
        norm += vec[i] * vec[i]
    return math.sqrt(norm)


def inverse(matrix):
    ones = np.matrix(np.eye(matrix.shape[0]))
    rows, columns = ones.shape[0], ones.shape[1]
    inv = np.matrix(np.zeros(([rows, columns])))
    for i in range(columns):
        inv[:, i] = MatrixSolver(matrix, ones[:, i]).gauss_elimination_pivot()[:, 0]
    return inv


def cond(matrix):
    norm_inv_a = norm_matrix(inverse(matrix))
    norm_a = norm_matrix(matrix)
    return norm_inv_a * norm_a


def err_estimate(con, b, db):
    return con * (norm_vec(db) / norm_vec(b))


def spectral_radius(matrix):
    eigval = np.linalg.eig(matrix)[0]
    return np.max(np.abs(eigval))
