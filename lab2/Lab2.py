from IterativeSolver import IterativeSolver1
import MatrixSolver
import MatrixUtils
import IterativeSolverHelper
import numpy as np

if __name__ == '__main__':
    a = np.matrix([
        [12.785723, 1.534675, -3.947418],
        [1.534675, 9.709232, 0.918435],
        [-3.947418, 0.918435, 7.703946]
    ])
    b = np.matrix([[9.60565], [7.30777], [4.21575]])
    solver = IterativeSolver1(a,b)
    print("1. Exact solution = \n{}\n".format(solver.x_gauss))
    print("2. ||H|| = {}\n".format(MatrixUtils.norm_matrix(solver.h)))
    k = 7
    print("3. A priori error estimation for k = " +str(k)+ ": {}\n".format(solver.a_priori_err_estimate(k)))
    k_iterations = solver.simple_k_iterations(7)
    lusternik_app = solver.lusternik_approximation(solver.simple_k_iterations(7).res, solver.simple_k_iterations(6).res)
    print(("4. Simple iteration for k = 7: \n" +
           "x7 = \n{}\n"
           "Error = {}\n" +
           "A posteriori error = {}\n"
           "Lusternik approximation = \n{}\n" +
           "Lusternik approximation error = {}\n")
          .format(k_iterations.res,
                  k_iterations.err,
                  solver.a_posteriori_err_estimate(solver.simple_k_iterations(7).res, solver.simple_k_iterations(6).res),
                  lusternik_app.res,
                  lusternik_app.err))
    seidel = solver.seidel(7)
    print(("5. Seidel for k = 7:\n" +
           "x7 = \n{}\n"
           "\nError = {}\n".format(seidel.res, seidel.err)))
    transition_matrix = IterativeSolverHelper.get_seidel_transition_matrix(solver.h)
    print("6. Spectral Radius H = {}\n".format(MatrixUtils.spectral_radius(transition_matrix)))
    relax = solver.upper_relaxation(7)
    print(("7. Upper relaxation for k = 7:\n" +
           "x7 = \n{}\n" +
           "Error = {}\n").format(relax.res, relax.err))
