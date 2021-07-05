#ifndef GAUSS_STYLE_DATA_H
#define GAUSS_STYLE_DATA_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <limits>
#include <igl/min_quad_with_fixed.h>

struct gauss_style_data
{
	// user should tune these
	double lambda = 0.0;

	// relative maximal Chebyshev distance in a step
	double reldV = std::numeric_limits<float>::max();

	// g function parameters per group
	std::vector<double> mu;
	std::vector<double> sigma;
	std::vector<double> caxiscontrib;
	std::vector<Eigen::VectorXd> r_weights;
	std::vector<Eigen::VectorXd> n_weights;
	std::vector<Eigen::VectorXd> style_R;
	std::vector<Eigen::MatrixXd> style_N;
	Eigen::VectorXi FGroups;

	// Mesh related
	std::vector<Eigen::MatrixXi> hEList;
	Eigen::MatrixXi EFList;
	Eigen::MatrixXi FEList;
	Eigen::MatrixXi EVList;

	Eigen::MatrixXd nf_stars;
  Eigen::MatrixXd e_ij_stars;

	// Global Step
	std::vector<Eigen::MatrixXd> dVList;
	std::vector<Eigen::VectorXd> WVecList;
	Eigen::SparseMatrix<double> K, L;
	Eigen::MatrixXd RAll;

	// Constraints
	Eigen::MatrixXd bc;
	Eigen::VectorXi b;

	igl::min_quad_with_fixed_data<double> solver_data;

	void reset()
	{
		reldV = std::numeric_limits<float>::max();

		hEList.clear();
		EFList = Eigen::MatrixXi();
		EVList = Eigen::MatrixXi();
		FEList = Eigen::MatrixXi();

		nf_stars = Eigen::MatrixXd();
		e_ij_stars = Eigen::MatrixXd();

		dVList.clear();
		WVecList.clear();

		K = Eigen::SparseMatrix<double>();
		L = Eigen::SparseMatrix<double>();

		igl::min_quad_with_fixed_data<double> solver_data;
	}
};

#endif
