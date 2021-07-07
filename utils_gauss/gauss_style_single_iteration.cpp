#include "gauss_style_single_iteration.h"




void gauss_style_single_iteration(
  const Eigen::Ref<Eigen::MatrixXd> V,
  Eigen::Ref<Eigen::MatrixXd> U,
  Eigen::Ref<Eigen::MatrixXi> F,
  gauss_style_data& data,
  int iter_cnt) {

  using namespace Eigen;
  using namespace std;

  Eigen::Matrix<double,Eigen::Dynamic,3> newNormals;
  igl::per_face_normals(U, F, newNormals);

  MatrixXd E_target_edges_rhs = MatrixXd::Zero(V.rows(), 3);
  int edges_cnt = data.EFList.rows();
  int faces_cnt = data.FEList.rows();
  data.e_ij_stars = MatrixXd::Zero(edges_cnt,3);
  igl::parallel_for(edges_cnt,
    [&data, &U](const int i) {
      int v1 = data.EVList(i,0);
      int v2 = data.EVList(i,1);
      data.e_ij_stars.row(i) = U.row(v2) - U.row(v1);
    });

  data.nf_stars = newNormals;

  MatrixXd A_stars = Matrix<double,3,Dynamic>::Zero(3,3*faces_cnt);

  MatrixXd u_fij = MatrixXd::Zero(faces_cnt,3);

  VectorXd g_values;

  for (size_t iter = 0; iter < iter_cnt; iter++) {

  igl::parallel_for(faces_cnt,
    [&data, &u_fij, &A_stars, &g_values](const int f) {

      int group = data.FGroups(f);

      Vector3d n = data.nf_stars.row(f);

      // calc gradient and hessian of g w.r.t n (equation 8, 9)
      Vector3d gg = g_grad(n, data.style_N[group], data.n_weights[group], data.style_R[group], data.r_weights[group], data.sigma[group], data.caxiscontrib[group]);
      Matrix3d Hg = g_hessian(n, data.style_N[group], data.n_weights[group], data.style_R[group], data.r_weights[group], data.sigma[group], data.caxiscontrib[group]);

      Vector3d gr = Vector3d::Zero();
      Matrix3d Hr = Matrix3d::Zero();
      for (size_t j=0; j<3; j++)
      {
        int e = data.FEList(f,j);
        int v1 = data.EVList(e,0);
        int v2 = data.EVList(e,1);
        double w_ij = data.L.coeff(v1, v2);
        Vector3d e_ij_star = data.e_ij_stars.row(e);
        Hr += data.lambda*w_ij*e_ij_star*e_ij_star.transpose();
        gr += data.lambda*w_ij*(data.nf_stars.row(f)*e_ij_star + u_fij(f,j))*e_ij_star;
      }

      // Newton grad:
      Vector3d gn = (gr-gg);
      Vector3d gn_newton = (Hr-Hg).ldlt().solve(-gn);

      // project out the component in direction of n and use as gradient step:
      Matrix3d Pt = (Matrix3d::Identity() - n*n.transpose());
      Vector3d d = Pt * gn_newton;

      // use gradient step if netwon points in wrong direction
      if (d.dot(gn) > 0 ) {
        d = -0.1 * Pt * gn;
      }

      // update current normal
      n += d;
      n.normalize();
      // save (for g calc)
      data.nf_stars.row(f) = n;
      A_stars.block<3,3>(0,f*3) =  n*n.transpose();

  }, 1000);

  igl::parallel_for(edges_cnt,
    [&data, &U, &A_stars, &u_fij](const int ij) {

      int f1 = data.EFList(ij,0);
      int f2 = data.EFList(ij,1);
      int v1 = data.EVList(ij,0);
      int v2 = data.EVList(ij,1);
      Vector3d e_ij = U.row(v2) - U.row(v1);
      Vector3d e_ij_star;
      double w_ij = data.L.coeff(v1, v2);
      Matrix<double, 3, 3> A =  Matrix<double, 3, 3>::Identity();

      Vector3d rhs = e_ij;

      if (f1 != -1){
          int group = data.FGroups(f1);
          A += data.mu[group] * A_stars.block<3,3>(0,f1*3);
          for (size_t k = 0; k < 3; k++) {
            if (data.FEList(f1,k) == ij){
              rhs -= data.mu[group] * u_fij(f1,k) * data.nf_stars.row(f1);
            }
          }
      }
      if (f2 != -1){
          int group = data.FGroups(f2);
          A += data.mu[group] * A_stars.block<3,3>(0,f2*3);
          for (size_t k = 0; k < 3; k++) {
            if (data.FEList(f2,k) == ij){
              rhs -= data.mu[group] * u_fij(f2,k) * data.nf_stars.row(f2);
            }
          }
      }

      e_ij_star = A.ldlt().solve(rhs);
      data.e_ij_stars.row(ij) = e_ij_star;

  }, 1000);

  igl::parallel_for(faces_cnt,
    [&data, &U, &A_stars, &u_fij](const int f) {

      for (size_t j=0; j<3; j++)
      {
        int e = data.FEList(f,j);
        u_fij(f,j) += data.e_ij_stars.row(e)*data.nf_stars.row(f).transpose();
      }

  }, 1000);
  }


  // construct contribution to the right hand side (equation 16)
  for (size_t i = 0; i < edges_cnt; i++)
  {
    int v1 = data.EVList(i,0);
    int v2 = data.EVList(i,1);

    double w_ij = data.L.coeff(v1, v2);
    {
      E_target_edges_rhs.row(v1) += -w_ij * data.e_ij_stars.row(i);
      E_target_edges_rhs.row(v2) += w_ij * data.e_ij_stars.row(i);
    }
  }

  // LOCAL STEP: FIT ROTATIONS./ga
  igl::parallel_for(U.rows(),
    [&data, &U](const int i) {

    MatrixXi hE = data.hEList[i];
    MatrixXd edgeStarts, edgeEnds;
    igl::slice(U, hE.col(0), 1, edgeStarts);
    igl::slice(U, hE.col(1), 1, edgeEnds);

    Matrix3d SB = data.dVList[i] * data.WVecList[i].asDiagonal() *
      (edgeEnds - edgeStarts);

    Matrix3d Rik;
    Matrix3d RSB = SB;
    igl::polar_svd3x3(RSB,Rik);
    Rik.transposeInPlace();
    data.RAll.block<3,3>(0,i*3) = Rik;

  }, 1000);

  // GLOBAL STEP
  MatrixXd Upre = U;
  {
    VectorXd Rcol;
    igl::columnize(data.RAll, V.rows(), 2, Rcol);
    VectorXd Bcol = data.K * Rcol;

    for (int dim = 0; dim < V.cols(); dim++)
    {
      VectorXd Uc, Bc, bcc;
      // constraint
      Bc = (Bcol.block(dim * V.rows(), 0, V.rows(), 1) + data.lambda * (E_target_edges_rhs.col(dim)))/(1+data.lambda);

      bcc = data.bc.col(dim);
      min_quad_with_fixed_solve(
        data.solver_data, Bc, bcc, VectorXd(), Uc);
      U.col(dim) = Uc;
    }
  }
  data.reldV = (U-Upre).cwiseAbs().maxCoeff() / (U-V).cwiseAbs().maxCoeff();
}
