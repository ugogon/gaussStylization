#include "gauss_style_precomputation.h"
#include <igl/adjacency_list.h>
#include <igl/edge_topology.h>
#include <igl/doublearea.h>

void gauss_style_precomputation(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  gauss_style_data& data)
{
  using namespace Eigen;
  using namespace std;

  data.reset();

  Eigen::MatrixXi EV, FE, EF;
  igl::edge_topology(V, F, data.EVList, data.FEList, data.EFList);
  EV = data.EVList;
  FE = data.FEList;
  EF = data.EFList;

  if (data.FGroups.rows() != EF.rows()){
			data.FGroups = VectorXi::Zero(EF.rows());
	}

  igl::cotmatrix(V, F, data.L);

  vector<vector<int>> adjFList, VI;
  igl::vertex_triangle_adjacency(V.rows(), F, adjFList, VI);

  igl::arap_rhs(V, F, V.cols(), igl::ARAP_ENERGY_TYPE_SPOKES_AND_RIMS, data.K);

  data.hEList.resize(V.rows());
  data.WVecList.resize(V.rows());
  data.dVList.resize(V.rows());
  vector<int> adjF;
  for (int ii = 0; ii < V.rows(); ii++)
  {
    adjF = adjFList[ii];

    data.hEList[ii].resize(adjF.size() * 3, 2);
    data.WVecList[ii].resize(adjF.size() * 3);
    for (int jj = 0; jj < adjF.size(); jj++)
    {
      int e0 = FE(adjF[jj],0);
      int e1 = FE(adjF[jj],1);
      int e2 = FE(adjF[jj],2);

      data.hEList[ii](3 * jj, 0) = EV(e0,0);
      data.hEList[ii](3 * jj, 1) = EV(e0,1);
      data.hEList[ii](3 * jj + 1, 0) = EV(e1,0);
      data.hEList[ii](3 * jj + 1, 1) = EV(e1,1);
      data.hEList[ii](3 * jj + 2, 0) = EV(e2,0);
      data.hEList[ii](3 * jj + 2, 1) = EV(e2,1);

      data.WVecList[ii](3 * jj) = data.L.coeff(EV(e0,0), EV(e0,1));
      data.WVecList[ii](3 * jj + 1) = data.L.coeff(EV(e1,0), EV(e1,1));
      data.WVecList[ii](3 * jj + 2) = data.L.coeff(EV(e2,0), EV(e2,1));
    }

    // compute [dV] matrix for each vertex
    data.dVList[ii].resize(3, adjF.size() * 3);
    MatrixXd V_hE0, V_hE1;
    igl::slice(V, data.hEList[ii].col(0), 1, V_hE0);
    igl::slice(V, data.hEList[ii].col(1), 1, V_hE1);

    data.dVList[ii] = (V_hE1 - V_hE0).transpose();
  }

  data.e_ij_stars = MatrixXd::Zero(EV.rows(),3);
  igl::parallel_for(EV.rows(),
    [&data, &V](const int i) {
      int v1 = data.EVList(i,0);
      int v2 = data.EVList(i,1);
      data.e_ij_stars.row(i) = V.row(v2) - V.row(v1);
    });

  Eigen::Matrix<double,Eigen::Dynamic,3> newNormals;
  igl::per_face_normals(V, F, newNormals);
  data.nf_stars = newNormals;

  igl::min_quad_with_fixed_precompute(data.L, data.b, SparseMatrix<double>(), false, data.solver_data);

  data.RAll = Eigen::Matrix<double,3,Eigen::Dynamic>::Zero(3,3*V.rows());
  for (int i = 0; i < 3*V.rows(); i++) data.RAll(i%3,i) = 1.0;

}
