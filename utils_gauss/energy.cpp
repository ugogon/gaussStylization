#include "energy.h"


double arap_energy(const Eigen::Ref<Eigen::MatrixXd> U, const gauss_style_data &data){
  double energy = 0;
  for (size_t i = 0; i < data.dVList.size(); i++) {
    Eigen::MatrixXi hE = data.hEList[i];
    Eigen::MatrixXd edgeStarts, edgeEnds;
    igl::slice(U, hE.col(0), 1, edgeStarts);
    igl::slice(U, hE.col(1), 1, edgeEnds);

    Eigen::Matrix3d SB = data.dVList[i] * data.WVecList[i].asDiagonal() * (edgeEnds - edgeStarts);

    Eigen::Matrix3d Rik = data.RAll.block<3,3>(0,i*3);
    energy += ((edgeEnds - edgeStarts) - data.dVList[i].transpose() * Rik).rowwise().squaredNorm().transpose() * data.WVecList[i];
  }
  return energy;
}

double orginal_energy_without_arap(const Eigen::Ref<Eigen::MatrixXd> U, const gauss_style_data &data, const Eigen::Ref<Eigen::MatrixXd> N){
  double energy = 0;
  for (size_t f = 0; f < data.FEList.rows(); f++) {
    int group = data.FGroups(f);
    energy -= data.mu[group]*g_value(N.row(f), data.style_N[group], data.n_weights[group], data.style_R[group], data.r_weights[group], data.sigma[group], data.caxiscontrib[group]);
  }
  return energy;
}

double coupling_energy_without_arap(const Eigen::Ref<Eigen::MatrixXd> U, const gauss_style_data &data, const Eigen::Ref<Eigen::MatrixXd> nf_stars, const Eigen::Ref<Eigen::MatrixXd> e_ij_stars){
  double energy = 0;
  for (size_t ij = 0; ij < data.EFList.rows(); ij++) {
    int f1 = data.EFList(ij,0);
    int f2 = data.EFList(ij,1);
    int v1 = data.EVList(ij,0);
    int v2 = data.EVList(ij,1);
    Eigen::Vector3d e_ij = U.row(v2) - U.row(v1);
    double w_ij = data.L.coeff(v1, v2);
    Eigen::Vector3d e_ij_star = e_ij_stars.row(ij);
    Eigen::Vector3d diff = (e_ij_star-e_ij);
    energy += w_ij/2.*diff.transpose()*diff;
  }
  energy *= data.lambda;
  for (size_t f = 0; f < data.FEList.rows(); f++) {
    int group = data.FGroups(f);
    energy -= data.mu[group]*g_value(nf_stars.row(f), data.style_N[group], data.n_weights[group], data.style_R[group], data.r_weights[group], data.sigma[group], data.caxiscontrib[group]);
  }

  return energy;
}

double lagrange_without_arap(const Eigen::Ref<Eigen::MatrixXd> U, const gauss_style_data &data, const Eigen::Ref<Eigen::MatrixXd> nf_stars, const Eigen::Ref<Eigen::MatrixXd> e_ij_stars, const Eigen::Ref<Eigen::MatrixXd> u_fij){
  double energy = coupling_energy_without_arap(U, data, nf_stars, e_ij_stars);
  for (size_t f = 0; f < data.FEList.rows(); f++) {
    int group = data.FGroups(f);
    for (size_t ij = 0; ij < 3; ij++) {
      int e = data.FEList(ij,ij);
      int v1 = data.EVList(e,0);
      int v2 = data.EVList(e,1);
      double w_ij = data.L.coeff(v1, v2);
      double dot = nf_stars.row(ij)*e_ij_stars.row(ij).transpose();
      energy += data.mu[group]*(u_fij(f,ij)*dot+(w_ij/2.)*dot*dot);
    }
  }
  return energy;
}
