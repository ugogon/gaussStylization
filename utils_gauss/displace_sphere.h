#ifndef DISPLACE_SPHERE_H
#define DISPLACE_SPHERE_H

// #include <iostream>
#include <Eigen/Core>

#include <igl/parallel_for.h>
#include <gauss_style_data.h>
#include <g_function.h>

void displace_sphere(
  Eigen::MatrixXd& V,
  Eigen::MatrixXd& U,
  gauss_style_data& data,
  int group);
#endif
