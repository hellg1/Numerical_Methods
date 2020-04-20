#include <iostream>
#include <math.h>
#include <iomanip>
#include <tuple>

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "inverse.h"


using namespace boost::numeric::ublas;
using namespace std;

int sign(double x)
{
	if (x > 0) return 1;
	if (x == 0) return 0;
	if (x < 0) return -1;
}

void print_matrix(boost::numeric::ublas::matrix<double> a)
{
	
	std::cout << a(0, 0) << " " << a(0, 1) << " " << a(0, 2) << std::endl;
	std::cout << a(1, 0) << " " << a(1, 1) << " " << a(1, 2) << std::endl;
	std::cout << a(2, 0) << " " << a(2, 1) << " " << a(2, 2) << std::endl;
}

void get_Jacobi_eigen(matrix<double> A, double err)
{
	identity_matrix<double> X1(3);
	matrix<double> X(3, 3); X = X1;
	double max_err = 1e6;
	matrix<double> X2(3, 3);
	while (max_err > err)
	{
		max_err = 0; int ik = 0; int jk = 0;
		for (int i = 0; i < 3; i++)
			for (int j = i + 1; j < 3; j++)
				if (abs(A(i, j)) > max_err)
				{
					max_err = abs(A(i, j));
					ik = i;
					jk = j;
				}
		if (max_err > err)
		{
			double d = sqrt((A(ik, ik) - A(jk, jk)) * (A(ik, ik) - A(jk, jk)) + 4 * A(ik, jk) * A(ik, jk));
			double c = sqrt((1 + abs(A(ik, ik) - A(jk, jk)) / d) / 2);
			double s = sign(A(ik, jk) * (A(ik, ik) - A(jk, jk))) * sqrt((1 - abs(A(ik, ik) - A(jk, jk)) / d) / 2);
			identity_matrix<double> V(3);
			
			matrix<double> V1(3, 3); V1 = V;
			V1(ik, ik) = c; V1(jk, jk) = c;
			V1(ik, jk) = -s; V1(jk, ik) = s;
			matrix<double> A1(3, 3); A1 = A;
			for (int i = 0; i < 3; i++)
				if ((i != ik) && (i != jk))
				{
					A(i, ik) = c * A1(i, ik) + s * A1(i, jk);
					A(ik, i) = c * A1(i, ik) + s * A1(i, jk);
					A(i, jk) = c * A1(i, jk) - s * A1(i, ik);
					A(jk, i) = c * A1(i, jk) - s * A1(i, ik);
				}
			A(ik, ik) = c * c * A1(ik, ik) + 2 * c * s * A1(ik, jk) + s * s * A1(jk, jk);
			A(jk, jk) = s * s * A1(ik, ik) - 2 * c * s * A1(ik, jk) + c * c * A1(jk, jk);
			A(ik, jk) = 0;
			A(jk, ik) = 0;
			X = prod(X, V1);
		}
	}
	X = trans(X);
	boost::numeric::ublas::vector<double> v0(3);
	boost::numeric::ublas::vector<double> v1(3);
	boost::numeric::ublas::vector<double> v2(3);
	for (int i = 0; i < 3; i++)
	{
		v0(i) = X(0, i);
	}
	v0 /= norm_2(v0);
	for (int i = 0; i < 3; i++)
	{
		v1(i) = X(1, i);
	}
	v1 /= norm_2(v1);
	for (int i = 0; i < 3; i++)
	{
		v2(i) = X(2, i);
	}
	v2 /= norm_2(v2);
	for (int i = 0; i < 3; i++)
		X(0, i) = v0(i);
	for (int i = 0; i < 3; i++)
		X(1,i) = v1(i);
	for (int i = 0; i < 3; i++)
		X(2,i) = v2(i);
	cout << "Eigenvalues are: " << setprecision(9) << A(0, 0) << " " << A(1, 1) << " " << A(2, 2) << endl;
	cout << "Matrix of eigenvectors:" << endl << trans(X) << endl;
}

tuple<double, boost::numeric::ublas::vector<double>, int> eig_max_pow(matrix<double> A, double err)
{
	boost::numeric::ublas::vector<double> v(3); v(0) = 1; v(1) = 1; v(2) = 1;
	boost::numeric::ublas::zero_vector<double> v0(3);
	boost::numeric::ublas::vector<double> eig_max(v0);
	int iters = 0;
	while (1)
	{
		++iters;
		boost::numeric::ublas::vector<double> tmp_v(v);
		boost::numeric::ublas::vector<double> tmp_max(eig_max);
		v = prod(A, v);
		for (int i = 0; i < 3; i++)
			eig_max[i] = v[i] / tmp_v[i];
		if ((abs(eig_max[0] - tmp_max[0]) < err) && (abs(eig_max[1] - tmp_max[1]) < err)
			&& (abs(eig_max[2] - tmp_max[2]) < err))
			break;
	}
	double max = INT_MIN;
	for (int i = 0; i < 3; i++)
		if (abs(eig_max(i)) > max)
			max = eig_max(i);
	return make_tuple(max, v / norm_2(v), iters);
}

tuple<double, boost::numeric::ublas::vector<double>, int> eig_max_scal(matrix<double> A, double err)
{
	boost::numeric::ublas::vector<double> v(3); v(0) = 1; v(1) = 1; v(2) = 1;
	boost::numeric::ublas::zero_matrix<double> A0(3, 3);
	boost::numeric::ublas::matrix<double> eig_max(A0);
	int iters = 0;
	while (1)
	{
		++iters;
		boost::numeric::ublas::vector<double> tmp_v(v);
		boost::numeric::ublas::matrix<double> tmp_max(eig_max);
		v = prod(A, v);
		eig_max(0, 0) = inner_prod(trans(v), trans(tmp_v)) / inner_prod(trans(tmp_v), trans(tmp_v));
		if (abs(eig_max(0, 0) - tmp_max(0, 0)) < err) break;
	}
	return make_tuple(eig_max(0, 0), v / norm_2(v), iters);
}

tuple<double, boost::numeric::ublas::vector<double>> spec_bound(matrix<double> A, double err)
{
	double eig_A = get<0>(eig_max_pow(A, err));
	identity_matrix<double> E1(3); matrix<double> E(E1);
	matrix<double> B = A - eig_A*E;
	double eig_B = get<0>(eig_max_pow(B, err));
	return make_tuple(eig_A + eig_B, get<1>(eig_max_pow(A + B, err)));
}

tuple<double, boost::numeric::ublas::vector<double>, int> vilandt(matrix<double> A, double eig_0, double err)
{
	double eig_max = eig_0;
	int iters = 0; bool flag = 0;
	boost::numeric::ublas::vector<double> v(3);
	identity_matrix<double> E1(3); matrix<double> E(E1);
	while (1)
	{
		++iters;
		matrix<double> W (A - eig_max * E);
		v = get<1>(eig_max_scal(gjinverse<double>(W, flag), err));
		double tmp_max = eig_max;
		eig_max += 1 / get<0>(eig_max_scal(gjinverse<double>(W, flag), err));
		if ((abs(eig_max - tmp_max) < err) && (eig_max * tmp_max > 0)) break;
	}
	return make_tuple(eig_max, -v, iters);
}

int main()
{
	boost::numeric::ublas::matrix<double> A(3,3);
	A(0, 0) = -0.90701;
	A(0, 1) = -0.27716;
	A(0, 2) = 0.44570;
	A(1,0) = -0.27716;
	A(1, 1) = 0.63372;
	A(1, 2) = 0.07774;
	A(2,0) = 0.44570;
	A(2,1) = 0.07774;
	A(2, 2) = -0.95535;

	cout << "Given matrix A:" << endl << A<<endl<<endl<<endl;
	
	cout << "1) Jacobi method" << endl;
	get_Jacobi_eigen(A, 1e-6);
	cout << endl << endl << "2) Power method" <<endl;
	cout<< "Max in absolute value eigenvalue of matrix A is "<<get<0>(eig_max_pow(A, 1e-3)) <<endl 
		<< "Eigenvector for this eigenvalue is "<<get<1>(eig_max_pow(A, 1e-3)) << endl 
		<< "The process of finding took "<<get<2>(eig_max_pow(A, 1e-3)) << " iterations." << endl;
	cout << endl << endl<<"3) Scalar method"<<endl;
	cout << "Max in absolute value eigenvalue of matrix A is " << get<0>(eig_max_scal(A, 1e-6)) << endl 
		 << "Eigenvector for this eigenvalue is "<<get<1>(eig_max_scal(A, 1e-6)) << endl 
		 << "The process of finding took " << get<2>(eig_max_scal(A, 1e-6)) << " iterations." << endl;
	cout << endl << endl << "4) Spectrum bound"<<endl;
	cout << "Spectrum bound equals " << get<0>(spec_bound(A, 1e-3)) << endl 
		 << "Eigenvector is " << get<1>(spec_bound(A, 1e-3)) << endl;
	cout << endl << endl << "5) Vilandt method" <<endl;
	cout << "Eigenvalue is "<<get<0>(vilandt(A, get<0>(eig_max_scal(A, 1e-3)), 1e-3)) << endl 
		 << "Eigenvector is "<<get<1>(vilandt(A, get<0>(eig_max_scal(A, 1e-3)), 1e-3)) << endl 
		 << "The process of finding took " << get<2>(vilandt(A, get<0>(eig_max_scal(A, 1e-3)), 1e-3)) <<" iterations."<< endl;
}