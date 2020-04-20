import numpy as np


class MatrixSolver:
    def __init__(self, matrix, b):
        self.matrix = matrix
        self.b = b
        self.e = 10 ** (-5)

    def gauss_elimination(self):
        n = len(self.b)
        aug = np.hstack((self.matrix, np.reshape(self.b, (n, 1))))
        for col in range(n):
            for row in range(col + 1, n):
                pivot = aug[row, col] / aug[col, col]
                #if abs(pivot) < self.e:
                #    raise Exception("too small pivot element")
                aug[row, :] -= pivot * aug[col, :]
        x = np.zeros_like(self.b)
        for row in range(n - 1, -1, -1):
            x[row] = aug[row, -1] / aug[row, row]
            for col in range(row + 1, n):
                x[row] -= aug[row, col] * x[col] / aug[row, row]
        return x

    def gauss_elimination_pivot(self):
        n = len(self.b)
        aug = np.hstack((self.matrix, np.reshape(self.b, (n, 1))))
        for col in range(n):
            ind = np.argmax(np.abs(aug[col:, col]))
            if ind != col:
                aug[[col, ind + col], :] = aug[[ind + col, col], :]
            for row in range(col + 1, n):
                pivot = aug[row, col] / aug[col, col]
                #if abs(pivot) < self.e:
                #    raise Exception("too small pivot element")
                aug[row, :] -= pivot * aug[col, :]
        x = np.zeros_like(self.b)
        for row in range(n - 1, -1, -1):
            x[row] = aug[row, -1] / aug[row, row]
            for col in range(row + 1, n):
                x[row] -= aug[row, col] * x[col] / aug[row, row]
        return x

    def get_gauss_pivot(self):
        return self.gauss_elimination_pivot()[:, -1]

    def get_gauss(self):
        return self.gauss_elimination()[:, -1]
