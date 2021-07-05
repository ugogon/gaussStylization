#ifndef ENERGY_H
#define ENERGY_H

#include <igl/parallel_for.h>
#include <Eigen/Core>

#include "gauss_style_data.h"
#include "g_function.h"

double arap_energy(const Eigen::Ref<Eigen::MatrixXd> U, const gauss_style_data &data);
double orginal_energy_without_arap(const Eigen::Ref<Eigen::MatrixXd> U, const gauss_style_data &data, const Eigen::Ref<Eigen::MatrixXd> N);
double coupling_energy_without_arap(const Eigen::Ref<Eigen::MatrixXd> U, const gauss_style_data &data, const Eigen::Ref<Eigen::MatrixXd> nf_stars, const Eigen::Ref<Eigen::MatrixXd> e_ij_stars);
double lagrange_without_arap(const Eigen::Ref<Eigen::MatrixXd> U, const gauss_style_data &data, const Eigen::Ref<Eigen::MatrixXd> nf_stars, const Eigen::Ref<Eigen::MatrixXd> e_ij_stars, const Eigen::Ref<Eigen::MatrixXd> u_fij);

#endif
