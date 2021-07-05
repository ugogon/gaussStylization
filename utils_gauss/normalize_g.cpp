#include "normalize_g.h"

void normalize_g(gauss_style_data &data) {
  // calc normalizing coefficients w_k for the function
  // g(x) = \sum_k w_k exp(sigma n_k.T x)
  // s.t. g(n_k) = 1 \forall k=1,...,K (n_k target normals)
  // N: (K,3) target normals
  //for each group
  for (size_t g = 0; g < data.style_N.size(); g++) {
    /* code */
    const Eigen::MatrixXd N = data.style_N[g];
    const Eigen::MatrixXd D = data.style_R[g];

    double K = N.rows();
    if (K > 0){
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K,K);

      for (int i=0; i<K; ++i) {
        for (int j=0; j<K; ++j) {
          A(i,j) = exp(data.sigma[g] * N.row(i).dot(N.row(j)));
        }
      }
      // TODO: really psd?
      data.n_weights[g] = A.ldlt().solve(Eigen::MatrixXd::Ones(K,1));
    }
    // normalize axis
    K = D.size();
    if (K > 0){
      Eigen::Vector3d axis(0,1,0);
      Eigen::MatrixXd B = Eigen::MatrixXd::Zero(K,K);
      for (int i=0; i<K; ++i) {
        for (int j=0; j<K; ++j) {
          B(i,j) = exp(data.sigma[g] * (1 - pow(D(i)-D(j),2)));
        }
      }
      // TODO: really psd?
      data.r_weights[g] = B.ldlt().solve(Eigen::MatrixXd::Ones(K,1));
    }
  }
}
