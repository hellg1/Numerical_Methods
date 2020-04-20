import numpy as np
from collections import namedtuple
import MatrixSolver
import MatrixUtils
import IterativeSolverHelper

Result = namedtuple('Result', ['res', 'err'])


class IterativeSolver1:
    def __init__(self, matrix, b):
        self.matrix = matrix
        self.b = b
        self.h = IterativeSolverHelper.get_h(matrix)
        self.g = IterativeSolverHelper.get_g(matrix, b)
        self.x_gauss = MatrixSolver.MatrixSolver(matrix, b).gauss_elimination()

    def simple_iteration(self, err):
        k = 0
        x_cur = np.matrix(np.zeros(self.matrix.shape[0])).T
        err_k = 0
        while self.a_priori_err_estimate(x_cur) > err:
            k += 1
            x_cur = self.h @ x_cur + self.g
            err_k = MatrixUtils.norm_vec(self.x_gauss - x_cur)
        return Result(x_cur, err_k), k

    def simple_k_iterations(self, iterations):
        x_cur = np.matrix(np.zeros(self.matrix.shape[0])).T
        err = 0
        for i in range(iterations):
            x_cur = self.h @ x_cur + self.g
            err = MatrixUtils.norm_vec(self.x_gauss - x_cur)
        return Result(x_cur, err)

    def a_priori_err_estimate(self, k):
        h_norm = MatrixUtils.norm_matrix(self.h)
        g_norm = MatrixUtils.norm_vec(self.g)
        err = (h_norm ** k) / (1 - h_norm) * g_norm
        return err

    def a_posteriori_err_estimate(self, x_k, x_k_1):
        h_norm = MatrixUtils.norm_matrix(self.h)
        err = h_norm / (1 - h_norm) * MatrixUtils.norm_vec(x_k - x_k_1)
        return err

    def lusternik_approximation(self, x_k, x_k_1):
        r = MatrixUtils.spectral_radius(self.h)
        x_lust = x_k_1 + (x_k - x_k_1) / (1 - r)
        err = MatrixUtils.norm_vec(x_lust - self.x_gauss)
        return Result(x_lust, err)

    def seidel(self, k):
        h_l, h_r = IterativeSolverHelper.transform_h(self.h)
        e = np.eye(self.h.shape[0])
        e_hl_inv = MatrixUtils.inverse(e - h_l)
        x_cur = np.matrix(np.zeros(self.matrix.shape[0])).T
        err = 0
        for i in range(k):
            x_cur = e_hl_inv @ h_r @ x_cur + e_hl_inv @ self.g
            err = MatrixUtils.norm_vec(self.x_gauss - x_cur)
        return Result(x_cur, err)

    def upper_relaxation(self, k):
        q = 2 / (1 + np.sqrt(1 - MatrixUtils.spectral_radius(self.h) ** 2))
        num_rows, num_cols = self.h.shape
        x_cur = np.matrix(np.zeros(self.matrix.shape[0])).T
        x_new = x_cur
        err = 0
        for j in range(k):
            for i in range(num_rows):
                sum_1 = IterativeSolverHelper.sum1(self.h, x_cur, i)
                sum_2 = IterativeSolverHelper.sum2(self.h, x_new, i)
                x_cur[i] = x_cur[i] + q * (sum_1 + sum_2 - x_cur[i] + self.g[i])
            x_new = x_cur
            err = MatrixUtils.norm_vec(self.x_gauss - x_new)
        return Result(x_new, err)
