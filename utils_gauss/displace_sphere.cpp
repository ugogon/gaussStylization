#include "displace_sphere.h"


void displace_sphere(
  Eigen::MatrixXd& V,
  Eigen::MatrixXd& U,
  gauss_style_data& data,
  int group)
{
    const Eigen::MatrixXd N = data.style_N[group];
    const Eigen::MatrixXd D = data.style_R[group];

    U = Eigen::MatrixXd::Zero(V.rows(),3);
    igl::parallel_for(U.rows(),
    [&data, N, &V, &U, &D, group](const int i) {
        U.row(i) = V.row(i)*g_value(V.row(i), N, data.n_weights[group], D, data.r_weights[group], data.sigma[group], data.caxiscontrib[group]);
    }, 1000);

    // the normalization is no longer needed now that we have the n_weights (at
    // least for the discrete mode)
    double max = 0;
    for (size_t i = 0; i < V.rows(); i++)
    {
        if (max < U.row(i).norm()){
            max = U.row(i).norm();
        }
    }

    U /= max;
}
