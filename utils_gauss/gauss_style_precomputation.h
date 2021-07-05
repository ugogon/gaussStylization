#pragma once

#include <iostream>
#include <ctime>
#include <vector>
#include <Eigen/Core>

// include libigl functions
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/arap_rhs.h>
#include <igl/columnize.h>
#include <igl/slice.h>
#include <igl/min_quad_with_fixed.h>

// include cube flow functions
#include <gauss_style_data.h>

void gauss_style_precomputation(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  gauss_style_data& data);
