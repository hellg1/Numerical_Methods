#include <iostream>
#include <algorithm>
#include <list>
#include <iomanip>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>

using namespace std;
using namespace boost::numeric::ublas;

matrix<double> input_matrix(boost::numeric::ublas::vector<double> v)
{
	matrix<double> m(3,4);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
			m(i,j) = v(i * 4 + j);
	}
	return m;
}

matrix<double> check_matrix(matrix<double> m)
{
	unsigned int size = 3;
	list<int> deleteColumn;
	for (int i = 0; i < size; i++)
	{
		int count = 0;
		for (int j = 0; j < size; j++)
		{
			if (fabs(m(j,i)) < 1e-6)
				count++;
		}

		if (count == size)
			deleteColumn.push_back(i);
	}

	list<int> deleteRow;
	for (int i = 0; i < size; i++)
	{
		int count = 0;
		for (int j = 0; j < size + 1; j++)
		{
			if (fabs(m(i,j)) < 1e-12)
				count++;
		}

		if (count == size + 1)
			deleteRow.push_back(i);
	}

	matrix<double> newm(size - deleteRow.size(), 4 - deleteColumn.size());

	if (!deleteColumn.empty() or !deleteRow.empty())
	{
		int col_offset = 0;
		int row_offset = 0;
		for (int i = 0; i < size + 1; i++)
		{
			if (find(deleteColumn.begin(), deleteColumn.end(), i) == deleteColumn.end())
			{
				for (int j = 0; j < size; j++)
				{
					if (find(deleteRow.begin(), deleteRow.end(), j) == deleteRow.end())
						newm(j - row_offset, i - col_offset) = m(j, i);
					else
						row_offset++;
				}
			}
			else col_offset++;

			row_offset = 0;
		}
	}
	else newm = m;

	return newm;
}


void Gauss(matrix<double> m)
{
	unsigned int size = 3;
	for (int i = 0; i < size; i++)
	{
		double main_element = abs(m(i, i));
		int index = i;

		for (int j = i + 1; j < size; j++)
		{
			if (abs(m(j, i)) > main_element)
			{
				main_element = abs(m(j,i));
				index = j;
			}
		}

		if (index != i)
		{
			for (int j = i; j < size; j++)
				swap(m(i,j), m(index, j));
		}

		double melement = m(i, i);
		m(i, i) = 1;
		for (int j = i + 1; j < size + 1; j++)
			m(i, j) /= melement;

		for (int j = i + 1; j < size; j++)
		{
			double element = m(j, i);
			m(j, i) = 0;

			if (element != 0)
			{
				for (int k = i + 1; k < size + 1; k++)
					m(j, k) -= element * m(i, k);
			}
		}
	}

	// solution
	cout << "\nSystem solution by Gauss method:";

	double* x = new double[size];
	for (int i = size - 1; i > -1; i--)
	{
		x[i] = 0;
		double root = m(i, size);
		for (int j = size; j > i; j--)
			root -= x[j] * m(i, j);

		x[i] = root;
		cout << "\nx_" << i << " = " << x[i];
	}
}

void Jordan(matrix<double> m)
{

	unsigned int size = 3;
	matrix<double> rM(3, 4);

	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
		{
			if (i == j) rM(i, j) = 1;
			else rM(i, j) = 0;
		}

	matrix<double> unionM(3, 6);

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			unionM(i, j) = m(i, j);
			unionM(i, j+size) = rM(i, j);
		}
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < 2 * size; j++)
			unionM(i, j) /= m(i, i);

		for (int j = i + 1; j < size; j++)
		{
			double coef = unionM(j, i) / unionM(i, i);
			for (int k = 0; k < 2 * size; k++)
				unionM(j, k) -= unionM(i, k) * coef;
		}

		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
				m(j, k) = unionM(j, k);
		}
	}

	for (int i = size - 1; i > -1; i--)
	{
		for (int j = 2 * size - 1; j > -1; j--)
			unionM(i, j) /= m(i, i);

		for (int j = i - 1; j > -1; j--)
		{
			double coef = unionM(j, i) / unionM(i, i);

			for (int k = 2 * size - 1; k > -1; k--)
				unionM(j, k) -= unionM(i, k) * coef;
		}
	}

	cout << "\n\nInverse matrix by Jordan:\n";

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			cout << setw(11)<< unionM(i, j+size) << " ";

		cout << "\n";
	}
}

void LU(matrix<double> m)
{
	unsigned int size = 3;

	matrix<double> L(3, 3);
	matrix<double> U(m);

	for (int i = 0; i < size; i++)
		for (int j = i; j < size; j++)
			L(j, i) = U(j, i) / U(i, i);

	for (int i = 1; i < size; i++)
	{
		for (int j = i - 1; j < size; j++)
			for (int k = j; k < size; k++)
				L(k, j) = U(k, j) / U(j, j);

		for (int j = i; j < size; j++)
			for (int k = i - 1; k < size; k++)
				U(j, k) -= L(j, i-1) * U(i-1, k);
	}

	double det = 1;
	for (int i = 0; i < size; i++)
		det *= U(i, i);
	
	cout << "\nThe determinant of matrix of coefficients using LU-decomposition equals " << det <<endl<<endl;
}

int main()
{
	boost::numeric::ublas::vector<double> default_coef(12);
	default_coef(0) = 12.785723;
	default_coef(1) = 1.534675;
	default_coef(2) = -3.947418;
	default_coef(3) = 9.60565;
	default_coef(4) = 1.534675;
	default_coef(5) = 9.709232;
	default_coef(6) = 0.918435;
	default_coef(7) = 7.30777;
	default_coef(8) = -3.947418;
	default_coef(9) = 0.918435;
	default_coef(10) = 7.703946;
	default_coef(11) = 4.21575;
	
	matrix<double> matrix;
	cout << "The input matrix is: " << endl;
	matrix = input_matrix(default_coef);
	cout << matrix << endl;
	matrix = check_matrix(matrix);

	Gauss(matrix);
	Jordan(matrix);
	LU(matrix);

	return 0;
}

