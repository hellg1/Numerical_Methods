import numpy as np
import MatrixUtils


def sum1(h, x, i):
    summ = 0
    for j in range(0, i + 1):
        summ += h[i, j] * x[j]
    return summ


def sum2(h, x, i):
    summ = 0
    num_rows, num_cols = h.shape
    for j in range(i + 1, num_rows):
        summ += h[i, j] * x[j]
    return summ


def get_h(matrix):
    d_diag = np.diag(matrix)
    d = np.diag(d_diag)
    d_inv = MatrixUtils.inverse(d)
    e = np.eye(matrix.shape[0])
    h = e - d_inv @ matrix
    return h


def transform_h(h):
    h_l = np.tril(h, -1)
    h_r = np.triu(h, 0)
    return h_l, h_r


def get_seidel_transition_matrix(h):
    h_l, h_r = transform_h(h)
    e = np.eye(h.shape[0])
    e_hl_inv = MatrixUtils.inverse(e - h_l)
    return e_hl_inv


def get_g(matrix, b):
    d_diag = np.diag(matrix)
    d = np.diag(d_diag)
    d_inv = MatrixUtils.inverse(d)
    g = d_inv @ b
    return np.asmatrix(g)
