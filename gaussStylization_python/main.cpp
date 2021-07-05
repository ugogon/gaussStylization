#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/snap_points.h>
#include <igl/upsample.h>
#include <igl/per_face_normals.h>

#include <gauss_style_data.h>
#include <gauss_style_precomputation.h>
#include <gauss_style_single_iteration.h>
#include <normalize_unitbox.h>

#include "normalize_g.h"

#include "gauss_style_precomputation.h"
#include "gauss_style_single_iteration.h"

#include "displace_sphere.h"
#include "energy.h"

#include <chrono>
#include <string>
#include <ctime>
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Core>


namespace py = pybind11;
using namespace Eigen;
using namespace std;

tuple<MatrixXd,MatrixXd,MatrixXi> loadMesh(string meshName){
	string file = meshName;
	MatrixXd U,V;
	MatrixXi F;
	igl::readOBJ(file, V, F);
	normalize_unitbox(V);
	RowVector3d meanV = V.colwise().mean();
	V = V.rowwise() - meanV;
	U = V;
	return make_tuple(U,V,F);
}

void add_style(MatrixXd N, VectorXd D, gauss_style_data &data) {
	data.mu.push_back(1);
	data.sigma.push_back(4);
	data.caxiscontrib.push_back(0.5);
	data.n_weights.push_back(VectorXd(0));
	data.r_weights.push_back(VectorXd(0));
	data.style_R.push_back(D);
	data.style_N.push_back(N);
	normalize_g(data);
};

void init_constraints(Eigen::Ref<MatrixXd> V, Eigen::Ref<MatrixXi> F, gauss_style_data &data){
	data.bc.resize(1,3);
	data.bc << V.row(F(0,0));

	data.b.resize(1);
	data.b << F(0,0);
}

Eigen::Matrix<double,Eigen::Dynamic,3> calc_normals(Eigen::Ref<MatrixXd> V, Eigen::Ref<MatrixXi> F){
	Eigen::Matrix<double,Eigen::Dynamic,3> Normals;
	igl::per_face_normals(V, F, Normals);
	return Normals;
}

template <typename type, typename... options>
class myclass: public py::class_<type,options...> {
	using py::class_<type,options...>::class_;
	using py::class_<type,options...>::def_property;
	public:
		template <typename C, typename D, typename... Extra>
		myclass &def_viewreadwrite(const char *name, D C::*pm, const Extra&... extra) {
			 static_assert(std::is_same<C, type>::value || std::is_base_of<C, type>::value, "def_viewreadwrite() requires a class member (or base class member)");
			 py::cpp_function fget([pm](type &c) -> D& { return c.*pm; }, py::is_method(*this)),
										fset([pm](type &c, const D &value) { c.*pm = value; }, py::is_method(*this));
			 def_property(name, fget, fset, py::return_value_policy::reference_internal, extra...);
			 return *this;
		}
};

PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXd>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::MatrixXd>);

PYBIND11_MODULE(gauss_stylization, m){
	py::bind_vector<std::vector<double>>(m, "VectorDouble");
	py::bind_vector<std::vector<Eigen::VectorXd>>(m, "VectorVectorXd");
	py::bind_vector<std::vector<Eigen::MatrixXd>>(m, "VectorMatrixXd");

	myclass<gauss_style_data>(m, "data")
		.def_viewreadwrite("FGroups", &gauss_style_data::FGroups)
		.def_viewreadwrite("bc", &gauss_style_data::bc)
		.def_viewreadwrite("b", &gauss_style_data::b)
		.def_viewreadwrite("nf_stars", &gauss_style_data::nf_stars)
		.def_viewreadwrite("e_ij_stars", &gauss_style_data::e_ij_stars)
		.def(py::init<>())
		.def("reset", &gauss_style_data::reset)
		.def_readwrite("mu", &gauss_style_data::mu)
		.def_readwrite("lamda", &gauss_style_data::lambda)
		.def_readwrite("sigma", &gauss_style_data::sigma)
		.def_readwrite("caxiscontrib", &gauss_style_data::caxiscontrib)

		.def_readonly("reldV", &gauss_style_data::reldV)
		.def_readonly("FEList", &gauss_style_data::FEList)
		.def_readonly("n_weights", &gauss_style_data::n_weights)
		.def_readonly("r_weights", &gauss_style_data::r_weights)
		.def_readonly("style_R", &gauss_style_data::style_R)
		.def_readonly("style_N", &gauss_style_data::style_N)

		.def_readonly("hEList", &gauss_style_data::hEList)
		.def_readonly("EFList", &gauss_style_data::EFList)
		.def_readonly("FEList", &gauss_style_data::FEList)
		.def_readonly("EVList", &gauss_style_data::EVList)

		.def_readonly("dVList", &gauss_style_data::dVList)
		.def_readonly("WVecList", &gauss_style_data::WVecList)
		.def_readonly("K", &gauss_style_data::K)
		.def_readonly("L", &gauss_style_data::L)
		.def_readonly("RAll", &gauss_style_data::RAll)
	;
	m.def("add_style", &add_style);
	m.def("gauss_style_precomputation", &gauss_style_precomputation);
	m.def("gauss_style_single_iteration", &gauss_style_single_iteration);
	m.def("init_constraints", &init_constraints);
	m.def("loadMesh", &loadMesh);
	m.def("arap_energy", &arap_energy);
	m.def("orginal_energy_without_arap", &orginal_energy_without_arap);
	m.def("coupling_energy_without_arap", &coupling_energy_without_arap);
	m.def("lagrange_without_arap", &lagrange_without_arap);
	m.def("calc_normals", &calc_normals);
}
