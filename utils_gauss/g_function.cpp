#include <g_function.h>

double g_value(
  Eigen::Vector3d x,
  Eigen::MatrixXd N,
  Eigen::VectorXd w_discrete,
  Eigen::VectorXd D,
  Eigen::VectorXd w_circle,
  double sigma,
  double caxiscontrib)
{
   double K = N.rows();
   assert(K == w_discrete.rows());

   double value = 0;

   // discrete contribution
   for (int k=0; k<K; ++k) {
      value += w_discrete(k) * exp(sigma*N.row(k).dot(x));
   }

   // circle contribution
   if (D.size() > 0){
     value *= caxiscontrib;

     double R = D.size();
     assert(R == w_circle.rows());
     Eigen::Vector3d caxis(0., 1., 0.);
     double aw = caxis.dot(x);
     for (int k=0; k<R; ++k) {
        value +=  w_circle(k) * exp(sigma*(1. - pow(aw-D(k),2.)));
     }
   }

   return value;
}

Eigen::Vector3d g_grad(
  Eigen::Vector3d x,
  Eigen::MatrixXd N,
  Eigen::VectorXd w_discrete,
  Eigen::VectorXd D,
  Eigen::VectorXd w_circle,
  double sigma,
  double caxiscontrib)
{
   double K = N.rows();

   Eigen::Vector3d grad = Eigen::Vector3d::Zero();

   // discrete contribution
   for (int k=0; k<K; ++k) {
      grad += w_discrete(k) * sigma * N.row(k) * exp(sigma*N.row(k).dot(x));
   }

   // circle contribution
   if (D.size() > 0){
     grad *= caxiscontrib;
     double R = D.size();
     Eigen::Vector3d caxis(0., 1., 0.);
     double aw = caxis.dot(x);
     // derivative from w*exp(sigma*(1-(a'*x-d)^2))
     double d = 0.6;
     for (int k=0; k<R; ++k) {
       grad += w_circle(k) * sigma * caxis * -2. * exp(sigma*(1. - pow(aw-D(k),2.))) * (aw-D(k));
     }
   }

   return grad;
}

Eigen::Matrix3d g_hessian(
  Eigen::Vector3d x,
  Eigen::MatrixXd N,
  Eigen::VectorXd w_discrete,
  Eigen::VectorXd D,
  Eigen::VectorXd w_circle,
  double sigma,
  double caxiscontrib)
{
  double K = N.rows();

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  // discrete contribution
  for (int k=0; k<K; ++k) {
    H += w_discrete(k) * sigma*sigma * exp(sigma*N.row(k).dot(x)) * N.row(k).transpose() * N.row(k);
  }

  // circle contribution
  if (D.size() > 0){
    H *= caxiscontrib;

    double R = D.size();

    Eigen::Vector3d caxis(0., 1., 0.);
    double aw = caxis.dot(x);
    for (int k=0; k<R; ++k) {
      H += w_circle(k) * 2.*sigma*(2.*sigma*pow(aw-D(k),2)-1.) * exp(sigma*(1.-pow(aw-D(k),2.))) * caxis*caxis.transpose();
    }
  }

  return H;
}
