#ifndef G_FUNCTION_H
#define G_FUNCTION_H

#include <Eigen/Core>

double g_value(
  Eigen::Vector3d x,
  Eigen::MatrixXd N,
  Eigen::VectorXd w_discrete,
  Eigen::VectorXd D,
  Eigen::VectorXd w_circle,
  double sigma,
  double caxiscontrib);

Eigen::Vector3d g_grad(
  Eigen::Vector3d x,
  Eigen::MatrixXd N,
  Eigen::VectorXd w_discrete,
  Eigen::VectorXd D,
  Eigen::VectorXd w_circle,
  double sigma,
  double caxiscontrib);


Eigen::Matrix3d g_hessian(
  Eigen::Vector3d x,
  Eigen::MatrixXd N,
  Eigen::VectorXd w_discrete,
  Eigen::VectorXd D,
  Eigen::VectorXd w_circle,
  double sigma,
  double caxiscontrib);

#endif
