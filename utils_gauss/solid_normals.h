#pragma once
#include <Eigen/Core>

class solid_normals {
public:
	static Eigen::MatrixXd axis() {
		Eigen::MatrixXd axisnormals(2, 3);
		axisnormals <<
		0,  1,  0,
		0, -1,  0;
		auto axis_norms = axisnormals.rowwise().norm();
		axisnormals = axisnormals.array().colwise() / axis_norms.array();
		return axisnormals;
	}

	static Eigen::MatrixXd cube() {
		Eigen::MatrixXd cubenormals(6, 3);
		cubenormals <<
			0, 0, 1,
			0, 0, -1,
			0, 1, 0,
			0, -1, 0,
			1, 0, 0,
			-1, 0, 0;
		auto cube_norms = cubenormals.rowwise().norm();
		cubenormals = cubenormals.array().colwise() / cube_norms.array();
		return cubenormals;
	}

	static Eigen::MatrixXd pyramid() {
		Eigen::MatrixXd pyramidnormals(5, 3);
		pyramidnormals <<
			0, -1, 0,
			0.5, 0.5, 0,
			-0.5, 0.5, 0,
			0, 0.5, 0.5,
			0, 0.5, -0.5;
		auto pyramid_norms = pyramidnormals.rowwise().norm();
		pyramidnormals = pyramidnormals.array().colwise() / pyramid_norms.array();
		return pyramidnormals;
	}

	static Eigen::MatrixXd tetrahedron() {
		Eigen::MatrixXd tetnormals(4, 3);
		tetnormals <<
			0.4714045524597168, 0.8164966106414795, 0.3333333432674408,
			-0.942808985710144, 0.0, 0.3333333134651184,
			0.4714045524597168, -0.8164966106414795, 0.3333333134651184,
			0.0, 0.0, -1.0;
		auto tet_norms = tetnormals.rowwise().norm();
		tetnormals = tetnormals.array().colwise() / tet_norms.array();
		return tetnormals;
	}

	static Eigen::MatrixXd octahedron() {
		Eigen::MatrixXd octahedronnormals(8, 3);
		octahedronnormals <<
			-1, 1, -1,
			1, 1, -1,
			-1, 1, 1,
			1, 1, 1,
			-1, -1, 1,
			1, -1, -1,
			-1, -1, -1,
			1, -1, 1;
		auto octahedron_norms = octahedronnormals.rowwise().norm();
		octahedronnormals = octahedronnormals.array().colwise() / octahedron_norms.array();
		return octahedronnormals;
	}

	static Eigen::MatrixXd plane() {
		Eigen::MatrixXd planenormals(1, 3);
		planenormals <<
			0, -1, 0;
		return planenormals;
	}

	static Eigen::MatrixXd dodecahedron() {
		Eigen::MatrixXd dodecanormals(12, 3);
		dodecanormals <<
			-0.0, 0.8506507873535156, 0.5257311463356018,
			0.8506507873535156, 0.5257311463356018, -0.0,
			0.5257311463356018, -0.0, 0.8506507873535156,
			0.0, 0.8506508469581604, -0.5257311463356018,
			0.8506508469581604, -0.5257311463356018, 0.0,
			-0.5257311463356018, 0.0, 0.8506508469581604,
			-0.8506507873535156, 0.5257311463356018, 0.0,
			0.0, -0.8506507873535156, 0.5257311463356018,
			0.5257311463356018, 0.0, -0.8506507873535156,
			-0.5257311463356018, 0.0, -0.8506508469581604,
			-0.8506508469581604, -0.5257311463356018, 0.0,
			0.0, -0.8506508469581604, -0.5257311463356018;
		return dodecanormals;
	}

	static Eigen::MatrixXd icosahedron() {
		Eigen::MatrixXd iconormals(20, 3);
		iconormals <<
			0.5773502588272095, 0.5773502588272095, 0.5773502588272095,
			0.5773502588272095, 0.5773502588272095, -0.5773502588272095,
			0.5773502588272095, -0.5773502588272095, 0.5773502588272095,
			0.5773502588272095, -0.5773502588272095, -0.5773502588272095,
			-0.5773502588272095, 0.5773502588272095, 0.5773502588272095,
			-0.5773502588272095, 0.5773502588272095, -0.5773502588272095,
			-0.5773502588272095, -0.5773502588272095, 0.5773502588272095,
			-0.5773502588272095, -0.5773502588272095, -0.5773502588272095,
			0.35682207345962524, 0.9341722726821899, 0.0,
			-0.35682207345962524, 0.9341722726821899, 0.0,
			0.35682207345962524, -0.9341722726821899, 0.0,
			-0.35682207345962524, -0.9341722726821899, -0.0,
			0.9341722726821899, 0.0, 0.35682207345962524,
			0.9341722726821899, 0.0, -0.35682207345962524,
			-0.9341722726821899, 0.0, 0.35682207345962524,
			-0.9341722726821899, -0.0, -0.35682207345962524,
			0.0, 0.35682207345962524, 0.9341722726821899,
			0.0, -0.35682207345962524, 0.9341722726821899,
			0.0, 0.35682207345962524, -0.9341722726821899,
			-0.0, -0.35682207345962524, -0.9341722726821899;
		return iconormals;
	}

	static Eigen::MatrixXd prism() { // sheared cube
		Eigen::MatrixXd prismnormals(6, 3);
		prismnormals <<
			-1.0, -0.0, 0.0,
			0.0, 0.8447584509849548, -0.5351477265357971,
			1.0, -0.0, 0.0,
			0.0, -0.8447584509849548, 0.5351477265357971,
			0.0, 0.0, -1.0,
			0.0, -0.0, 1.0;
		return prismnormals;
	}

	static Eigen::MatrixXd truncatedPyramid() {
		Eigen::MatrixXd tpyramidnormals(6, 3);
		tpyramidnormals <<
			0, -1, 0,
			0.5, 0.5, 0,
			-0.5, 0.5, 0,
			0, 0.5, 0.5,
			0, 0.5, -0.5,
			0, 1, 0;
		auto pyramid_norms = tpyramidnormals.rowwise().norm();
		tpyramidnormals = tpyramidnormals.array().colwise() / pyramid_norms.array();
		return tpyramidnormals;
	}

	static Eigen::MatrixXd triangularPrism() {
		Eigen::MatrixXd tprismnormals(5, 3);
		tprismnormals <<
			-0.0, 0.0, -0.9999999403953552,
			0.0, 0.0, 1.0,
			0.8660253286361694, 0.5000000596046448, 0.0,
			8.603188916822546e-08, -1.0, 0.0,
			-0.8660253882408142, 0.5, 0.0;
		return tprismnormals;
	}

	static Eigen::MatrixXd hexPrism() {
		Eigen::MatrixXd hexprismnormals(8, 3);
		hexprismnormals <<
			0.0, 0.0, -1.0,
			0.0, 0.0, 0.9999999403953552,
			-0.5, 0.8660253882408142, 0.0,
			0.5000000596046448, 0.8660253286361694, 0.0,
			0.5000001192092896, -0.8660253286361694, 0.0,
			-1.0, -5.960464477539063e-08, 0.0,
			1.0, 0.0, 0.0,
			-0.4999999403953552, -0.8660253882408142, 0.0;
		return hexprismnormals;
	}

	static Eigen::MatrixXd hexPyramid() {
		Eigen::MatrixXd hexprismnormals(7, 3);
		hexprismnormals <<
			0.0, 0.0, -1.0,
			-0.3902907073497772, 0.6760033965110779, 0.625054121017456,
			0.390290766954422, 0.6760033369064331, 0.625054121017456,
			0.3902907967567444, -0.6760033369064331, 0.625054121017456,
			-0.7805814146995544, -4.652627794143882e-08, 0.625054121017456,
			0.7805814146995544, 0.0, 0.625054121017456,
			-0.39029067754745483, -0.6760033965110779, 0.625054121017456;
		return hexprismnormals;
	}
};
