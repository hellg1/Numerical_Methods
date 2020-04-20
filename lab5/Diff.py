import math


def p(x): return (6 + x) / (7 + 3 * x)


def q(x): return -(1 - x / 2)


def r(x): return 1 + math.cos(x) / 2


def f(x): return 1 - x / 3


a1, a2, a3, b1, b2, b3 = -2, -1, 0, 0, 1, 0
a, b = -1, 1


def coeff(n):
    A, B, C, G = [], [], [], []
    h = (b - a) / n
    X = [a + i * h for i in range(n + 1)]
    A.append(0)
    B.append(-a1 - a2 / h)
    C.append(-a2 / h)
    G.append(a3)
    for i in range(1, n):
        A.append(-p(X[i]) / (h ** 2) - q(X[i]) / (2 * h))
        B.append(-2 * p(X[i]) / (h ** 2) - r(X[i]))
        C.append(-p(X[i]) / (h ** 2) + q(X[i]) / (2 * h))
        G.append(f(X[i]))
    A.append(-b2 / h)
    B.append(-b1 - b2 / h)
    C.append(0)
    G.append(b3)
    return A, B, C, G, X


def coeff2(n):
    A, B, C, G = [], [], [], []
    h = (b - a) / n
    X = [a - h / 2 + i * h for i in range(n + 2)]
    A.append(0)
    B.append(-a1 / 2 - a2 / h)
    C.append(a1 / 2 - a2 / h)
    G.append(a3)
    for i in range(1, n + 1):
        A.append(-p(X[i]) / (h ** 2) - q(X[i]) / (2 * h))
        B.append(-2 * p(X[i]) / (h ** 2) - r(X[i]))
        C.append(-p(X[i]) / (h ** 2) + q(X[i]) / (2 * h))
        G.append(f(X[i]))
    A.append(b1 / 2 - b2 / h)
    B.append(-b1 / 2 - b2 / h)
    C.append(0)
    G.append(b3)
    return A, B, C, G, X


def sweep(n, A, B, C, G):
    S, T = [], []
    S.append(C[0] / B[0])
    T.append(-G[0] / B[0])
    for i in range(1, n + 1):
        S.append(C[i] / (B[i] - A[i] * S[i - 1]))
        T.append((A[i] * T[i - 1] - G[i]) / (B[i] - A[i] * S[i - 1]))
    Y = [0 for i in range(n + 1)]
    Y[-1] = T[-1]
    for i in range(n - 1, -1, -1):
        Y[i] = S[i] * Y[i + 1] + T[i]
    return S, T, Y


def runge(n, o, Y1):
    A2, B2, C2, G2, X2 = coeff2(2 * n)
    S2, T2, Y0 = sweep(2 * n + 1, A2, B2, C2, G2)
    Y2 = [(Y0[i] + Y0[i + 1]) / 2 for i in range(2 * n + 1)]
    Y = []
    for i in range(n + 1):
        Y.append(Y2[i * 2] + (Y2[i * 2] - Y1[i]) / (2 ** o - 1))
    return Y
