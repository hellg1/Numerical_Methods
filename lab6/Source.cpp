#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

#define space (14)
#define number_of (100)

using namespace std;

#define eps 1e-10

vector<vector<double>> net;


void add_row(vector<vector<double>>& matrix, int dest, int source, double coeff) {
	for (int i = 0; i < matrix[0].size(); ++i) {
		matrix[dest][i] += matrix[source][i] * coeff;
	}
}

void print(vector<vector<double>> matrix) {
	cout << "printing table" << endl;
	for (const auto& row : matrix) {
		for (const auto& e : row) {
			cout << setw(space) << e;
		}
		cout << endl;
	}
	cout << "\n";
}

void simple(vector<vector<double>>& matrix) {
	for (int k = 0; k < matrix.size(); ++k) {
		if (abs(matrix[k][k]) < eps) {
			for (int p = k + 1; p < matrix.size(); ++p) {
				if (abs(matrix[p][k]) > eps) {
					swap(matrix[p], matrix[k]);
					break;
				}
			}
		}

		for (int i = k + 1; i < matrix.size(); ++i) {
			if (abs(matrix[k][k]) > eps) {
				add_row(matrix, i, k, -matrix[i][k] / matrix[k][k]);
			}
		}
	}
}

vector<double> sum(vector<double> l, vector<double> r) {
	auto arg = ((l.size() > r.size()) ? (l) : (r));
	auto arg2 = ((l.size() > r.size()) ? (r) : (l));
	vector<double> res(arg);
	for (int i = 0; i < arg2.size(); ++i) {
		res[i] += arg2[i];
	}
	return res;
}


double det(vector<vector<double>> m) {
	return (m[0][0] * m[1][1] - m[0][1] * m[1][0]);
}


vector<vector<double>> submatrix(vector<vector<double>> m, int r, int c) {
	vector<vector<double>> res;
	int count_i = 0;
	for (int i = 0; i < m.size(); ++i) {
		int count_j = 0;
		if (i == r) {
			count_i = 1;
			continue;
		}
		res.push_back({});
		for (int j = 0; j < m[i].size(); ++j) {
			if (j == c) {
				count_j = 1;
				continue;
			}
			res[i - count_i].push_back(m[i][j]);
		}
	}

	return res;
}


double calculate(vector<vector<double>> m) {
	simple(m);

	double r = 1;
	for (int i = 0; i < m.size(); ++i) {
		r *= m[i][i];
	}
	return r;
}


vector<vector<double>> construct_the_matrix(vector<vector<double>> m, int c) {
	vector<vector<double>> res;
	for (int i = 0; i < m.size(); ++i) {
		res.push_back({});
		for (int j = 0; j < m[i].size() - 1; ++j) {
			if (j == c) {
				res[i].push_back(m[i][m[0].size() - 1]);
				continue;
			}
			res[i].push_back(m[i][j]);
		}
	}
	return res;
}


vector<double> solve_system(vector<vector<double>> m) {

	vector<double> res;
	for (int i = 0; i < m.size(); ++i) {
		double d1, d2;
		d1 = calculate(construct_the_matrix(m, i));
		d2 = calculate(construct_the_matrix(m, m.size() + 10));
		res.push_back(d1 / d2);
	}

	return res;
}


double get_by_number(int n, int max_n) {
	double delta = 2.0 / max_n;
	return -1.0 + delta * n;
}

double get_h(int max_n) {
	return 2.0 / max_n;
}

double get_A(int n, int max_n) {
	double x = get_by_number(n, max_n);
	if (max_n < 10) {
		cout << "current x " << x << endl;
	}
	double res = 0;
	res += (x - 2) / (x + 2) / get_h(max_n) / get_h(max_n);
	res -= x / 2.0 / get_h(max_n);
	return res;
}

double get_B(int n, int max_n) {
	double x = get_by_number(n, max_n);
	double res = 0;
	res -= 2 * (x - 2) / (x + 2) / get_h(max_n) / get_h(max_n);
	res -= sin(x);
	res += 1;

	return res;
}

double get_C(int n, int max_n) {
	double x = get_by_number(n, max_n);
	double res = 0;
	res += (x - 2) / (x + 2) / get_h(max_n) / get_h(max_n);
	res += x / 2.0 / get_h(max_n);

	return res;
}

double get_G(int n, int max_n) {
	double x = get_by_number(n, max_n);
	double res = 0;
	res += x * x;

	return res;
}

vector<double> construct_matrix(int max_n) {
	vector<vector<double>> m(max_n + 1, vector<double>(max_n + 2, 0));
	m[0][0] = 1;
	m[max_n][max_n] = 1;
	vector<double> s(max_n + 1, 0);
	vector<double> t(max_n + 1, 0);
	vector<double> A(max_n + 1, 0);
	vector<double> B(max_n + 1, 0);
	vector<double> C(max_n + 1, 0);
	vector<double> G(max_n + 1, 0);
	B[0] = -1;
	B[max_n] = -1;

	s[0] = 0;
	t[0] = 0;

	for (int i = 1; i < max_n; ++i) {
		m[i][i - 1] = get_A(i, max_n);
		m[i][i] = get_B(i, max_n);
		m[i][i + 1] = get_C(i, max_n);
		m[i][max_n + 1] = get_G(i, max_n);

		A[i] = get_A(i, max_n);
		B[i] = -get_B(i, max_n);
		C[i] = get_C(i, max_n);
		G[i] = get_G(i, max_n);
		if (max_n < 10) {
			cout << "____ " << get_G(i, max_n) << endl;
		}

	}


	vector<double> res(max_n + 1, -888.888);
	for (int i = 1; i <= max_n; ++i) {
		s[i] = C[i] / (B[i] - A[i] * s[i - 1]);
		t[i] = (A[i] * t[i - 1] - G[i]) / (B[i] - A[i] * s[i - 1]);
	}
	res[max_n] = t[max_n];
	for (int i = max_n - 1; i >= 0; --i) {
		res[i] = s[i] * res[i + 1] + t[i];
	}
	




	if (max_n == 10) {
		cout << "TABLE 1:" << endl << setw(space) << "x_i:"
			<< setw(space) << "A_i:"
			<< setw(space) << "B_i:"
			<< setw(space) << "C_i:"
			<< setw(space) << "G_i:"
			<< setw(space) << "s_i:"
			<< setw(space) << "t_i:"
			<< setw(space) << "y_i" << endl;
		for (int i = 0; i < max_n + 1; ++i) {
			cout << setw(space) << get_by_number(i, max_n)
				<< setw(space) << A[i]
				<< setw(space) << B[i]
				<< setw(space) << C[i]
				<< setw(space) << G[i]
				<< setw(space) << s[i]
				<< setw(space) << t[i]
				<< setw(space) << res[i] << endl;
		}
	}
	return res;
}


double func_x_0(double x) {
	//    double res = 0;
	//
	return x * x;

}

double get_by_n_x(int n, int max_n) {
	double delta = 1.0 / max_n;
	return delta * n;
}

void init_net(int n_x, int n_t, int key) {
	//    ++n_t;
	//    ++n_x;
	net = vector<vector<double>>(n_t + 1, vector<double>(n_x + 2, -888.8));
	double delta_t = 0.1 / (net.size() - 1);
	double delta_x = 1.0 / (net[0].size() - 2);
	net[0][net[0].size() - 1] = 0;
	for (int n_t = 1; n_t < net.size(); ++n_t) {
		net[n_t][net[n_t].size() - 1] = net[n_t - 1][net[n_t].size() - 1] + delta_t;
	}
	if (key == 0) { ///function is u(x,t) = x^2 + t
		for (int i = 0; i <= n_x; ++i) {
			net[0][i] = func_x_0(get_by_n_x(i, n_x));
		}
		for (int j = 1; j < net.size(); ++j) {
			net[j][0] = net[j - 1][0] + delta_t;
			net[j][net[0].size() - 2] = net[j - 1][net[0].size() - 2] + delta_t;
		}

	}
}

void init_net_1(int n_x, int n_t) {
	//    ++n_t;
	//    ++n_x;
	net = vector<vector<double>>(n_t + 1, vector<double>(n_x + 2, -888.8));

	double delta_t = 0.1 / (net.size() - 1);
	double delta_x = 1.0 / (net[0].size() - 2);
	//    cout << "    deltas are (dt, dx): " << delta_t << " & " << delta_x << endl;
	net[0][net[0].size() - 1] = 0;
	for (int n_t = 1; n_t < net.size(); ++n_t) {
		net[n_t][net[n_t].size() - 1] = net[n_t - 1][net[n_t].size() - 1] + delta_t;
	}
	for (int i = 0; i <= n_x; ++i) {
		net[0][i] = sin(1) + cos(2 * i * delta_x);
		//        cout << i << " & " <<  i * delta_x << " " << net[0][i] << endl;
	}
	for (int j = 1; j < net.size(); ++j) {
		net[j][net[0].size() - 2] = sin(2 * j * delta_t + 1) + cos(2);
		net[j][0] = 1 + sin(2 * j * delta_t + 1);
	}
	for (int n_t = 1; n_t < net.size(); ++n_t) {
		for (int n_x = 1; n_x < net[n_t].size() - 2; ++n_x) {
			double x = n_x * delta_x;
			double t = n_t * delta_t;
			net[n_t][n_x] = net[n_t - 1][n_x] + delta_t *
				((net[n_t - 1][n_x + 1] - 2 * net[n_t - 1][n_x] +
					net[n_t - 1][n_x - 1]) / delta_x / delta_x + 4 * cos(2 * x) +
					2 * cos(2 * t + 1));
		}
	}

}

void print_table() {
	cout << endl << setw(space) << "t|x";
	double delta_t = 0.1 / (net.size() - 1);
	double delta_x = 1.0 / (net[0].size() - 2);
	for (int i = 0; i < net[0].size() - 1; ++i) {
		cout << setw(space) << i * delta_x;
	}
	cout << endl;
	for (int i = 0; i < net.size(); ++i) {
		cout << setw(space) << i * delta_t;
		for (int j = 0; j < net[0].size() - 1; ++j) {
			cout << setw(space) << net[i][j];
		}
		cout << endl;
	}
	//    cout << endl;
}

void print_table_2() {
	cout << endl << setw(space) << "t|x";
	double delta_t = 0.1 / (net.size() - 1);
	double delta_x = 1.0 / (net[0].size() - 2);
	for (int i = 0; i < net[0].size() - 1; ++i) {
		cout << setw(space) << i * delta_x;
	}
	cout << endl;
	for (int i = 0; i < net.size(); i += 2) {
		cout << setw(space) << i * delta_t;
		for (int j = 0; j < net[0].size() - 1; ++j) {
			cout << setw(space) << net[i][j];
		}
		cout << endl;
	}
	cout << endl;
}

void print_table_custom(vector<vector<double>> m) {
	cout << endl << setw(space) << "t|x";
	double delta_t = 0.1 / (m.size() - 1);
	double delta_x = 1.0 / (m[0].size() - 2);
	for (int i = 0; i < m[0].size() - 1; ++i) {
		cout << setw(space) << i * delta_x;
	}
	cout << endl;
	for (int i = 0; i < m.size(); i += 1) {
		cout << setw(space) << i * delta_t;
		for (int j = 0; j < m[0].size() - 1; ++j) {
			cout << setw(space) << m[i][j];
		}
		cout << endl;
	}
	//    cout << endl;
}

void process_table_0() { /////// u(x,t) = x^2 + t
	double delta_t = 0.1 / (net.size() - 1);
	double delta_x = 1.0 / (net[0].size() - 2);
	//    cout << "    deltas are (dt, dx): " << delta_t << " & " << delta_x << endl;
	//    net[0][net[0].size() - 1] = 0;
	for (int n_t = 1; n_t < net.size(); ++n_t) {
		for (int n_x = 1; n_x < net[n_t].size() - 2; ++n_x) {
			net[n_t][n_x] = net[n_t - 1][n_x] + delta_t *
				((net[n_t - 1][n_x + 1] - 2 * net[n_t - 1][n_x] +
					net[n_t - 1][n_x - 1]) / delta_x / delta_x - 1);
		}
	}
}

vector<vector<double>> cut_table_2(vector<vector<double>> m) {
	vector<vector<double>> res;
	for (int n_t = 0; n_t < m.size(); n_t += 2) {
		res.push_back({});
		for (int n_x = 0; n_x < m[0].size(); n_x += 2) {
			res[res.size() - 1].push_back(m[n_t][n_x]);
		}
	}
	return res;
}

vector<vector<double>> cut_table(vector<vector<double>> m) {
	vector<vector<double>> res;
	for (int n_t = 0; n_t < m.size(); n_t += 2) {
		res.push_back({});
		for (int n_x = 0; n_x < m[0].size() - 1; n_x += 2) {
			res[res.size() - 1].push_back(m[n_t][n_x]);
		}
		res[res.size() - 1].push_back(m[n_t][m[n_t].size() - 1]);

	}
	return res;

}

double min(double a, double b)
{
	return a < b ? a : b;
}

double max(double a, double b)
{
	return a > b ? a : b;
}

double get_diff_0(vector<vector<double>> l_m, vector<vector<double>> r_m) {
	double delta_t = 0.1 / (l_m.size() - 1);
	double delta_x = 1.0 / (l_m[0].size() - 2);
	double result = 0;
	int l_size = min(l_m.size(), r_m.size());
	int l_l_size = min(l_m[0].size() - 1, r_m[0].size() - 1);
	for (int i = 0; i < l_size; i += 1) {

		for (int j = 0; j < l_l_size; ++j) {
			result = max(abs(l_m[i][j] - r_m[i][j]), result);
		}
	}
	return result;
}

void fill_perfect_0(vector<vector<double>> & p) {
	double delta_t = 0.1 / (p.size() - 1);
	double delta_x = 1.0 / (p[0].size() - 2);
	for (int n_t = 1; n_t < p.size(); ++n_t) {
		for (int n_x = 1; n_x < p[n_t].size() - 2; ++n_x) {
			double x = n_x * delta_x;
			double t = n_t * delta_t;
			p[n_t][n_x] = 2 * (x) * (x)+(t);
		}
	}
}

void fill_perfect_sin(vector<vector<double>> & p) {
	double delta_t = 0.1 / (p.size() - 1);
	double delta_x = 1.0 / (p[0].size() - 2);
	for (int n_t = 1; n_t < p.size(); ++n_t) {
		for (int n_x = 1; n_x < p[n_t].size() - 2; ++n_x) {
			double x = n_x * delta_x;
			double t = n_t * delta_t;
			p[n_t][n_x] = sin(2 * t + 1) + cos(2 * x);
		}
	}
}

void print_differences(vector<vector<double>> l, vector<vector<double>> r, vector<vector<double>> exp) {
	cout << endl;
	cout << setw(space) << "h"
		<< setw(space) << "dt"
		<< setw(space) << "norm(exp-u)"
		<< setw(space) << "norm(u-u_2)" << endl;
	int n = l.size() - 1;
	cout << setw(space) << 1.0 / (l[0].size() - 2)
		<< setw(space) << 0.1 / (l.size() - 1)
		<< setw(space) << get_diff_0(exp, l)
		<< setw(space) << "" << endl;
	cout << setw(space) << 1.0 / 2.0 / (l[0].size() - 2)
		<< setw(space) << 0.1 / 2 / (l.size() - 1)
		<< setw(space) << get_diff_0(exp, r)
		<< setw(space) << get_diff_0(r, l) << endl;
}


vector<vector<double>> construct_matrix_by_k_0(vector<vector<double>> matrix, int k) {
	int n_x = net[0].size() - 2;
	int n_t = net.size() - 1;
	double delta_t = 0.1 / (net.size() - 1);
	double delta_x = 1.0 / (net[0].size() - 2);
	vector<vector<double>> m(n_t + 1, vector<double>(n_t + 2, 0));
	m[0][0] = 1;
	m[0][m[0].size() - 1] = k * delta_t;
	m[m.size() - 1][m[0].size() - 2] = 1;
	m[m.size() - 1][m[0].size() - 1] = 1 + k * delta_t;
	for (int i = 1; i < m.size() - 1; ++i) {
		m[i][i - 1] = 1.0 / delta_x / delta_x; 
		m[i][i] = -2.0 / delta_x / delta_x - 1.0 / delta_t; 
		m[i][i + 1] = 1.0 / delta_x / delta_x; 
		m[i][m[i].size() - 1] = -matrix[k - 1][i] / delta_t + 1;
	}
	return m;
}

vector<vector<double>> construct_matrix_by_k_1(vector<vector<double>> matrix, int k) {
	int n_x = net[0].size() - 2;
	int n_t = net.size() - 1;
	double delta_t = 0.1 / (net.size() - 1);
	double delta_x = 1.0 / (net[0].size() - 2);
	vector<vector<double>> m(n_t + 1, vector<double>(n_t + 2, 0));
	m[0][0] = 1;
	m[0][m[0].size() - 1] = sin(2 * k * delta_t + 1) + 1;
	m[m.size() - 1][m[0].size() - 2] = 1;
	m[m.size() - 1][m[0].size() - 1] = cos(2) + sin(2 * k * delta_t + 1);
	for (int i = 1; i < m.size() - 1; ++i) {
		m[i][i - 1] = 1.0 / delta_x / delta_x; 
		m[i][i] = -2.0 / delta_x / delta_x - 1.0 / delta_t; 
		m[i][i + 1] = 1.0 / delta_x / delta_x; 
		m[i][m[i].size() - 1] = -matrix[k - 1][i] / delta_t - (4 * cos(2 * i * delta_x) + 2 * cos(2 * k * delta_t + 1));
	}
	return m;
}

vector<vector<double>> fill_net_with_weights_0() {
	int n_x = net[0].size() - 2;
	int n_t = net.size() - 1;

	double delta_t = 0.1 / (net.size() - 1);
	init_net(n_t, n_x, 0);


	auto matrix = net;
	for (auto& e : matrix) {
		e.pop_back();
	}


	for (int k = 1; k < matrix.size(); ++k) {
		auto system = construct_matrix_by_k_0(matrix, k);
		auto result = solve_system(system);
		matrix[k] = result;

	}
	for (int i = 0; i < matrix.size(); ++i) {
		matrix[i].push_back(i * delta_t);
	}
	return matrix;
}




vector<vector<double>> fill_net_with_weights_1() {
	int n_x = net[0].size() - 2;
	int n_t = net.size() - 1;

	double delta_t = 0.1 / (net.size() - 1);
	init_net_1(n_t, n_x);


	auto matrix = net;
	for (auto& e : matrix) {
		e.pop_back();
	}

	for (int k = 1; k < matrix.size(); ++k) {
		auto system = construct_matrix_by_k_1(matrix, k);
		auto result = solve_system(system);
		matrix[k] = result;
	}
	for (int i = 0; i < matrix.size(); ++i) {
		matrix[i].push_back(i * delta_t);
	}
	return matrix;
}


int main() {
	string s;
	for (int i = 0; i < 200; ++i) {
		s += "_";
	}
	s += "\n";

	cout << setprecision(space - 7);

	cout << s << "\nTable for approximation of x^2 + t:\nwith n = 5";
	init_net(5, 5, 0);
	process_table_0();
	print_table();
	auto temp_0 = net;
	auto perfect = net;
	fill_perfect_0(perfect);


	cout << "\nwith  n = 10";
	init_net(10, 10, 0);
	process_table_0();
	auto cut = cut_table(net);
	print_table_custom(cut);

	print_differences(temp_0, cut, perfect);


	cout << s << endl;

	cout << "Table for weighted approximation of x^2 + t:\nwith n = 5";
	init_net(5, 5, 0);
	auto matrix = fill_net_with_weights_0();
	print_table_custom(matrix);
	init_net(10, 10, 0);
	auto matrix_2 = fill_net_with_weights_0();

	matrix_2 = cut_table(matrix_2);
	cout << "with n = 10";
	print_table_custom(matrix_2);
	print_differences(matrix, matrix_2, perfect);


	cout << s << "\nTable for approximation of sin(2 * t + 1) + cos(2 * x):\nwith n = 5";
	init_net_1(5, 5);
	print_table();
	auto perfect_sin = net;
	auto temp_sin = net;
	fill_perfect_sin(perfect_sin);

	
	cout << "with n = 10";
	init_net_1(10, 10);
	
	auto cut_sin = cut_table(net);

	print_table_custom(cut_sin);
	print_differences(temp_sin, cut_sin, perfect_sin);
	cout << s << endl;


	cout << "Table for weighted approximation of sin(2 * t + 1) + cos(2 * x):\nwith n = 5";
	init_net(5, 5, 0);
	auto matrix_3 = fill_net_with_weights_1();
	print_table_custom(matrix_3);
	init_net(10, 10, 0);
	auto matrix_3_2 = fill_net_with_weights_1();
	cout << "with n = 10";
	matrix_3_2 = cut_table(matrix_3_2);
	print_table_custom(matrix_3_2);
	print_differences(matrix_3, matrix_3_2, perfect_sin);

	return 0;
}
