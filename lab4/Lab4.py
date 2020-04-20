import FredgolmSolver

a, b = 0, 1


def task_1():
    u3, b3, nu3 = FredgolmSolver.un(3)
    print("x           a           (a+b)/2        b")
    print("u3(x)      ", round(u3(a), 8), "       ", round(u3((a + b) / 2), 8), "   ", round(u3(b), 8))
    u4, b4, nu4 = FredgolmSolver.un(4)
    print("u4(x)      ", round(u4(a), 8), "       ", round(u4((a + b) / 2), 8), "   ", round(u4(b), 8))
    print("delta      ", max(abs(u4(a) - u3(a)), abs(u4((a + b) / 2) - u3((a + b) / 2)), abs(u4(b) - u3(b))))
    print("a posterior ", FredgolmSolver.estimate(u3, b3, nu3))
    print()


def task_2():
    print("x           a           (a+b)/2        b")
    u2 = FredgolmSolver.solve(2)
    print("u2(x)      ", round(u2(a), 8), "       ", round(u2((a + b) / 2), 8), "   ", round(u2(b), 8))
    u4 = FredgolmSolver.solve(4)
    print("u4(x)      ", round(u4(a), 8), "       ", round(u4((a + b) / 2), 8), "     ", round(u4(b), 8))

    n = 8
    while max(abs(u4(a) - u2(a)), abs(u4((a + b) / 2) - u2((a + b) / 2)), abs(u4(b) - u2(b))) > 10e-5:
        u2 = u4
        u4 = FredgolmSolver.solve(n)
        print("u" + str(n) + "(x)     ", round(u4(a), 8), "       ", round(u4((a + b) / 2), 8), "   ", round(u4(b), 8))
        n *= 2
    print("u" + str(n) + "(x)-u" + str(n / 2) + "(x)   ",
          max(abs(u4(a) - u2(a)), abs(u4((a + b) / 2) - u2((a + b) / 2)), abs(u4(b) - u2(b))))


if __name__ == '__main__':
    task_1()
    task_2()
