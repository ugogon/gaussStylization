#pragma once

#include <iostream>
#include <ctime>
#include <vector>
#include <Eigen/Core>
#include <mutex>

// include libigl functions
#include <igl/columnize.h>
#include <igl/slice.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/fit_rotations.h>
#include <igl/parallel_for.h>
#include <igl/per_vertex_normals.h>

// include cube flow functions
#include <gauss_style_data.h>
#include <g_function.h>
#include <energy.h>

void gauss_style_single_iteration(
  const Eigen::Ref<Eigen::MatrixXd> V,
  Eigen::Ref<Eigen::MatrixXd> U,
  Eigen::Ref<Eigen::MatrixXi> F,
  gauss_style_data& data,
  int iter_cnt);
