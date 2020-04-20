import Diff
import plotly
import plotly.graph_objs as go

Y_ex1 = [0.346839, 0.457697, 0.524009, 0.560029, 0.576575, 0.581422, 0.579933,
         0.575686, 0.571024, 0.567483, 0.566115]

Y_ex2 = [0.346839, 0.408863, 0.457697, 0.495448, 0.524009, 0.545051, 0.560029,
         0.570186, 0.576575, 0.580077, 0.581422, 0.581211, 0.579933, 0.577984,
         0.575686, 0.573296, 0.571024, 0.56904, 0.567483, 0.566473, 0.566115]


def rnd(arr, n):
    return list(map(lambda x: round(x, n), arr))


def table1(n):
    A, B, C, G, X = Diff.coeff2(n)
    S, T, Y = Diff.sweep(n + 1, A, B, C, G)
    trace = go.Table(
        header=dict(values=['i', 'xi', 'Ai', 'Bi', 'Ci', 'Gi', 'Si', 'Ti', 'Yi']),
        cells=dict(values=[[i for i in range(n + 2)], rnd(X, 2), rnd(A, 5), rnd(B, 5), rnd(C, 5), rnd(G, 5), rnd(S, 5),
                           rnd(T, 5), rnd(Y, 5)]))
    data = [trace]
    plotly.offline.plot(data)


def table2(n, Y_ex):
    A1, B1, C1, G1, X1 = Diff.coeff(n)
    A2, B2, C2, G2, X2 = Diff.coeff2(n)
    S1, T1, Y1 = Diff.sweep(n, A1, B1, C1, G1)
    S2, T2, Y0 = Diff.sweep(n + 1, A2, B2, C2, G2)
    Y3 = [(Y0[i] + Y0[i + 1]) / 2 for i in range(n + 1)]
    Y1 = Diff.runge(n, 1, Y1)
    Y2 = Diff.runge(n, 2, Y3)
    trace = go.Table(
        header=dict(values=['x', 'Y_ex', 'Y_ut(h)', 'Y_ut - Y_ex', 'Y_ut(h^2)', 'Y_ut - Y_ex']),
        cells=dict(
            values=[rnd(X1, 2), Y_ex, rnd(Y1, 6), [round(abs(Y1[i] - Y_ex[i]), 6) for i in range(n + 1)], rnd(Y2, 5),
                    [round(abs(Y2[i] - Y_ex[i]), 10) for i in range(n + 1)]]))
    data = [trace]
    plotly.offline.plot(data)


if __name__ == '__main__':
    table1(10)
    table1(20)
    table2(10, Y_ex1)
    table2(20, Y_ex2)
