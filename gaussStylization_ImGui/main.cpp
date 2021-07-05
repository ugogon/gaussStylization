#include <igl/unproject_onto_mesh.h>
#include <igl/snap_points.h>
#include <igl/upsample.h>

#include <igl/opengl/glfw/imgui/ext/ImGuiMenu.h>
// SELECTION (HOTFIX version)
#include <igl/opengl/glfw/imgui/ext/SelectionPlugin.h>

#include <igl/AABB.h>
#include <igl/screen_space_selection.h>

#include <imgui/imgui.h>

#include <Eigen/Core>

// #include <ctime>
#include <vector>
#include <iostream>
#include <filesystem>

#include "get_bounding_box.h"
#include "normalize_unitbox.h"

#include "cube_style_data.h"
#include "cube_style_precomputation.h"
#include "cube_style_single_iteration.h"

#include "displace_sphere.h"
#include "solid_normals.h"
#include "normalize_g.h"

#include "gauss_style_data.h"
#include "gauss_style_precomputation.h"
#include "gauss_style_single_iteration.h"


// SELECTION HOTFIX (https://github.com/alecjacobson/libigl-issue-1656-hot-fix/blob/main/main.cpp)
namespace igl{ namespace opengl{ namespace glfw{ namespace imgui{
class PrePlugin: public igl::opengl::glfw::imgui::ImGuiMenu
{
public:
  PrePlugin(){};
  IGL_INLINE virtual bool pre_draw() override { ImGuiMenu::pre_draw(); return false;}
  IGL_INLINE virtual bool post_draw() override { return false;}
};
class PostPlugin: public igl::opengl::glfw::imgui::ImGuiMenu
{
public:
  PostPlugin(){};
  IGL_INLINE virtual bool pre_draw() override { return false;}
  IGL_INLINE virtual bool post_draw() override {  ImGui::Render(); ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); return false;}
};
}}}}


#ifndef MESH_PATH
#define MESH_PATH "../../meshes/"
#endif

#ifndef OUTPUT_PATH
#define OUTPUT_PATH "../"
#endif

#define WIDTH 960
#define HEIGHT 1079

std::function<Eigen::MatrixXd()> Normals[12] { solid_normals::cube, solid_normals::pyramid, solid_normals::tetrahedron, solid_normals::octahedron, solid_normals::plane, solid_normals::dodecahedron, solid_normals::icosahedron, solid_normals::prism, solid_normals::truncatedPyramid, solid_normals::triangularPrism, solid_normals::hexPrism, solid_normals::hexPyramid};

// state of the mode
struct Mode
{
  Eigen::MatrixXd CV; // point constraint
  int numCV = 0;
  Eigen::MatrixXi CV_row_col;
  bool place_constraints = true;
  bool viewGfunc = true;
  bool running = false;
  bool gauss = true;
  unsigned int meshNr = 3;
  int group = 0;
  Eigen::ArrayXi groups;
  bool select_mode = false;
  int iter = 1;
} state;


int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;

  // data initialization
  // ---------------------------------------------------------------------------
  cube_style_data cube_data;
  gauss_style_data gauss_data;
  cube_data.lambda = 4;
  gauss_data.lambda = 4;

  auto add_style = [&](MatrixXd N, VectorXd D) {
    gauss_data.mu.push_back(1);
  	gauss_data.sigma.push_back(4);
  	gauss_data.caxiscontrib.push_back(0.5);
    gauss_data.n_weights.push_back(VectorXd(0));
    gauss_data.r_weights.push_back(VectorXd(0));
    gauss_data.style_R.push_back(D);
    gauss_data.style_N.push_back(N);
    normalize_g(gauss_data);
  };
  auto remove_style = [&](int k) {
    assert(gauss_data.mu.size() > 1);
    for (size_t i = 0; i < gauss_data.FGroups.size(); i++) {
      if (gauss_data.FGroups(i) > k){
        gauss_data.FGroups(i) -= 1;
      }
    }
    gauss_data.mu.erase(gauss_data.mu.begin()+k);
  	gauss_data.sigma.erase(gauss_data.sigma.begin()+k);
  	gauss_data.caxiscontrib.erase(gauss_data.caxiscontrib.begin()+k);
    gauss_data.n_weights.erase(gauss_data.n_weights.begin()+k);
    gauss_data.r_weights.erase(gauss_data.r_weights.begin()+k);
    gauss_data.style_R.erase(gauss_data.style_R.begin()+k);
    gauss_data.style_N.erase(gauss_data.style_N.begin()+k);
  };

  MatrixXd Mempty(0,0);
  VectorXd Vempty(0);
  add_style(solid_normals::cube(),Vempty);


  // load mesh
  // ---------------------------------------------------------------------------
	MatrixXd V, U, VO;
	MatrixXi F;

  // for the selection tool
  VectorXd W;
  Array<double,Dynamic,1> and_visible;
  igl::AABB<MatrixXd, 3> tree;

  // collect all files in MESH_PATH
  std::vector<string> fileNames;
  std::vector<filesystem::path> paths;
  copy(filesystem::directory_iterator(MESH_PATH), filesystem::directory_iterator(), std::back_inserter(paths));
  sort(paths.begin(), paths.end());
  for (const auto& entry : paths) {
    // extract file names
    stringstream nameStream;
    nameStream << entry.filename();
    string name = nameStream.str();
    fileNames.push_back(name.substr(1, name.length() - 6));
  };

  auto loadMesh = [&](int nr) {
	  string file = MESH_PATH + fileNames.at(nr) + ".obj";
	  igl::readOBJ(file, V, F);
	  normalize_unitbox(V);
    RowVector3d meanV = V.colwise().mean();
    V = V.rowwise() - meanV;
    U = V;
    VO = V;
    W = VectorXd::Zero(V.rows());
    and_visible = Array<double,Dynamic,1>::Zero(V.rows());
    gauss_data.FGroups = VectorXi::Zero(F.rows());
    state.select_mode = false;
    tree.init(V,F);
  };
  loadMesh(state.meshNr);

  // load sphere for g-function view
  // ---------------------------------------------------------------------------
  MatrixXd SphereV, osV;
  MatrixXd Disp;
	MatrixXi SphereF, osF;
  MatrixXd SphereN;
  {
    string file = MESH_PATH + (string)"sphere.obj";
    igl::readOBJ(file, osV, osF);
    igl::upsample(osV, osF, SphereV, SphereF);
    igl::per_vertex_normals(SphereV, SphereF, SphereN);
  }

  // initialize viewer and plugins (selection HOTFIX, see definition of PRE, POST above)
  // ---------------------------------------------------------------------------
  igl::opengl::glfw::Viewer viewer;

  igl::opengl::glfw::imgui::PrePlugin PRE;
  igl::opengl::glfw::imgui::PostPlugin POST;
  igl::opengl::glfw::imgui::ext::SelectionPlugin selection_plugin;
  igl::opengl::glfw::imgui::ext::ImGuiMenu menu;
  igl::opengl::glfw::imgui::ext::ImGuiMenu groupmenu;
  igl::opengl::glfw::imgui::ext::ImGuiMenu stylemenu;
  viewer.plugins.push_back(&PRE);
  viewer.plugins.push_back(&selection_plugin);
	viewer.plugins.push_back(&menu);
  viewer.plugins.push_back(&groupmenu);
  viewer.plugins.push_back(&stylemenu);
  viewer.plugins.push_back(&POST);

  viewer.data().point_size = 5;
  viewer.core().viewport = Vector4f(WIDTH, 0, WIDTH, HEIGHT);
  int right_view = viewer.core_list[0].id;
  int left_view = viewer.append_core(Vector4f(WIDTH, 0, WIDTH, HEIGHT));


  int Sphere = viewer.append_mesh();
  int Model = viewer.append_mesh();
  viewer.data(Model).set_visible(false, right_view);
  viewer.data(Sphere).set_visible(false, left_view);

  // selection plugin config
  // ---------------------------------------------------------------------------
  W = VectorXd::Zero(V.rows());
  and_visible = Array<double,Dynamic,1>::Zero(V.rows());

  const auto update = [&]()
  {
    // const bool was_face_based = viewer.data().face_based;
    VectorXd S = W;

    // calculate facewise groups
    for (int i = 0; i < F.rows(); i++) {
      // select most frequent group
      if (S(F(i, 0)) + S(F(i, 1)) + S(F(i,2)) > 1 ) {
        gauss_data.FGroups(i) = state.group;
      }
    }
  };

  // recompute state.CV function (point constraints)
  // ---------------------------------------------------------------------------
  int maxCV = 100;
  state.CV_row_col.resize(maxCV,2); // assume nor more than 100 constraints
  const auto & resetCV = [&](){
    for (int ii=0; ii<state.numCV; ii++) {
        int fid = state.CV_row_col(ii,0);
        int c   = state.CV_row_col(ii,1);
        RowVector3d new_c = V.row(F(fid,c));
        state.CV.row(ii) = new_c;
    }
  };


  // colors
  // ---------------------------------------------------------------------------
  const RowVector3d red(250.0/255, 114.0/255, 104.0/255);
  const RowVector3d green(100.0/255, 255/255, 104.0/255);
  const RowVector3d blue(149.0/255, 217.0/255, 244.0/255);
  const RowVector3d orange(250.0/255, 240.0/255, 104.0/255);
  const RowVector3d black(0., 0., 0.);
  const RowVector3d gray(200.0/255, 200.0/255, 200.0/255);
  const auto darkblue = blue * 0.5;
  const auto darkred = red * 0.5;

  // color palette 1
  MatrixXd cp1 (9, 3);
  cp1 << 38./255., 70./255., 83./255.,
        42./255., 157./255., 143./255.,
        233./255., 196./255., 106./255.,
        244./255., 162./255., 97./255.,
        231./255., 111./255., 81./255.,
        0./255., 157./255., 143./255.,
        233./255., 0/255., 106./255.,
        244./255., 162./255., 0./255.,
        231./255., 111./255., 255./255.;

  // ---------------------------------------------------------------------------
  // ---------------------- main draw function ---------------------------------
  // ---------------------------------------------------------------------------
  const auto & draw = [&](){

    // single iteration step
    // -------------------------------------------------------------------------
    if (state.running) {
      if (state.gauss) {
        gauss_style_single_iteration(V, U, F, gauss_data, state.iter);
      }
      else {
        cube_style_single_iteration(V, U, cube_data);
      }
    }

    // show g-function (right window)
    // -------------------------------------------------------------------------
    if (state.viewGfunc){
      viewer.data(Sphere).clear();
      displace_sphere(SphereV, Disp, gauss_data, state.group);
      viewer.data(Sphere).set_mesh(Disp, SphereF);
      MatrixXd N_vertices;
      igl::per_vertex_normals(Disp, SphereF, N_vertices);
      viewer.data(Sphere).set_normals(N_vertices);
      viewer.data(Sphere).show_lines = false;
      viewer.data(Sphere).set_colors(RowVector3d(1., 1., 1.));

      // handles for g manipulation
      if (gauss_data.style_R[state.group].size() > 0){
        MatrixXd rings = MatrixXd::Zero(gauss_data.style_R[state.group].size(), 3);
        for (size_t i = 0; i < gauss_data.style_R[state.group].size(); i++){
          rings(i,1) = gauss_data.style_R[state.group](i);
          rings(i,0) = sqrt(1-pow(gauss_data.style_R[state.group](i),2));
        }
        viewer.data(Sphere).set_points(rings,orange);
      } else {
        viewer.data(Sphere).set_points(gauss_data.style_N[state.group],orange);
      }

      // draw axis
      viewer.data(Sphere).line_width = 3;
      RowVector3d origin(0., 0., 0.);
      viewer.data(Sphere).add_edges(origin, RowVector3d::Unit(0)*1.2, red);
      viewer.data(Sphere).add_edges(origin, RowVector3d::Unit(1)*1.2, green);
      viewer.data(Sphere).add_edges(origin, RowVector3d::Unit(2)*1.2, blue);
    }

    // idle mode
    // -------------------------------------------------------------------------
    if (state.place_constraints){
      viewer.data().clear();
      viewer.data().face_based = true;
      viewer.data().set_mesh(V,F);

      if (state.gauss) {
        MatrixXd clrs(F.rows(), 3);
        for (int i = 0; i < F.rows(); ++i) {
          clrs.row(i) = cp1.row(gauss_data.FGroups(i));
        }
        viewer.data().set_colors(clrs);
      } else {
        viewer.data().set_colors(blue);
      }
      viewer.data().set_points(state.CV, red);

      // draw bounding box
      MatrixXd V_box;
      MatrixXi E_box;
      get_bounding_box(V, V_box, E_box);
      viewer.data().add_points(V_box, red);
      for (unsigned i=0;i<E_box.rows(); ++i)
        viewer.data().add_edges(V_box.row(E_box(i,0)),V_box.row(E_box(i,1)),red);
    }
    // running
    // -------------------------------------------------------------------------
    else {

      viewer.data().clear();
      viewer.data().face_based = true;
      viewer.data().set_mesh(U,F);

      if (state.gauss) {
        MatrixXd clrs(F.rows(), 3);
        for (int i = 0; i < F.rows(); ++i) {
          clrs.row(i) = cp1.row(gauss_data.FGroups(i))/2. + MatrixXd::Ones(1, 3) / 2.;
        }
        viewer.data().set_colors(clrs);
      } else {
        viewer.data().set_colors((blue + MatrixXd::Ones(1, 3)) / 2.) ;
      }
      viewer.data().set_points(state.CV, red);

      // draw bounding box
      MatrixXd V_box;
      MatrixXi E_box;
      get_bounding_box(U, V_box, E_box);
      viewer.data().add_points(V_box, red);
      for (unsigned i=0;i<E_box.rows(); ++i)
        viewer.data().add_edges(V_box.row(E_box(i,0)),V_box.row(E_box(i,1)),red);

    }

    // draw input normals
    // -------------------------------------------------------------------------
    RowVector3d origin(0., 0., 0.);
    RowVector3d color = darkblue;
    for (unsigned i = 0; i < gauss_data.style_N[state.group].rows(); i++) {
      viewer.data().add_edges(origin, gauss_data.style_N[state.group].row(i), color);
    }
    // draw origin
    // -------------------------------------------------------------------------
    viewer.data().add_points(origin.transpose(), black);
  };

  // reset
  // ---------------------------------------------------------------------------
  const auto & reset = [&](bool reset_constraints=true){
    state.place_constraints = true; // switch mode
    state.running = false;

    if (reset_constraints) {
        resetCV();
        state.CV = MatrixXd();
        state.CV_row_col = MatrixXi();
        state.CV_row_col.resize(maxCV,2);
        state.numCV = 0;
    }
    V = VO;
    U = V;
    if (!reset_constraints) {
        resetCV();
    }
    draw();
  };

  // ---------------------------------------------------------------------------
  // ------------------------- Interactions ------------------------------------
  // ---------------------------------------------------------------------------
  // for the g func handles
  int sel = -1;

  // mesh rotation parameters
  // ---------------------------------------------------------------------------
  double theta_x = 10.0;
  double theta_y = 10.0;
  double theta_z = 10.0;
  Matrix3d Rx, Ry, Rz;
  Rx << 1., 0., 0.,
        0., cos(theta_x/180.*3.14), -sin(theta_x/180.*3.14),
        0., sin(theta_x/180.*3.14), cos(theta_x/180.*3.14);

  Ry << cos(theta_y/180.*3.14), 0., sin(theta_y/180.*3.14),
        0., 1., 0.,
        -sin(theta_y/180.*3.14), 0, cos(theta_y/180.*3.14);

  Rz << cos(theta_z/180.*3.14), -sin(theta_z/180.*3.14), 0.,
        sin(theta_z/180.*3.14), cos(theta_z/180.*3.14), 0.,
        0., 0., 1;

  const auto rotate = [&](Matrix3d Ri)
  {
    U = (Ri * U.transpose()).transpose();
    if (state.place_constraints) {
      V = (Ri * V.transpose()).transpose();
      resetCV();
      draw();
    }
  };
  // when key pressed do
  // ---------------------------------------------------------------------------
  viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod) {
    switch(key) {
      case '>':
      case '.': {
        gauss_data.lambda += 0.1;
        cube_data.lambda = gauss_data.lambda;
        break;
      }
      case '<':
      case ',': {
        gauss_data.lambda -= 0.1;
        cube_data.lambda = gauss_data.lambda;
        break;
      }
      case 'R':
      case 'r': {
        reset(false);
        break;
      }
      case 'Q':
      case 'q': {
        rotate(Rx);
        break;
      }
      case 'W':
      case 'w': {
        rotate(Rx.transpose());
        break;
      }
      case 'A':
      case 'a': {
        rotate(Ry);
        break;
      }
      case 'S':
      case 's': {
        rotate(Ry.transpose());
        break;
      }
      case 'X':
      case 'x': {
        rotate(Rz);
        break;
      }
      case 'Y':
      case 'y': {
        rotate(Rz.transpose());
        break;
      }
      // case 'N':
      // case 'n': {
      //   auto current = viewer.core(left_view).viewport;
      //   if (state.viewGfunc){
      //     viewer.resize(current[2],current[3]);
      //   } else {
      //     viewer.resize(current[2]*2,current[3]);
      //   }
      //   state.viewGfunc = !state.viewGfunc;
      //
      //   draw();
      //   break;
      // }
      case ' ': {
        // start/stop stylization
        if(state.CV.rows()==0) {
            // if not constraint points, then set the F(0,0) to be the contrained point
            state.CV = MatrixXd();
            state.CV.resize(1,3);
            RowVector3d new_c = V.row(F(0,0));
            state.CV.row(0) << new_c;
            state.CV_row_col = MatrixXi();
            state.CV_row_col.resize(maxCV,2);
            state.CV_row_col.row(0) << 0, 0;
            state.numCV = 1;
        }
        if (!state.running && state.place_constraints) {
          // set constrained points and pre computation
          U = V;
          if (state.gauss) {
            igl::snap_points(state.CV, V, gauss_data.b);
            gauss_data.bc.setZero(gauss_data.b.size(), 3);
            for (int ii=0; ii<gauss_data.b.size(); ii++)
              gauss_data.bc.row(ii) = V.row(gauss_data.b(ii));
            gauss_style_precomputation(V, F, gauss_data);
          }
          else {
            igl::snap_points(state.CV, V, cube_data.b);
            cube_data.bc.setZero(cube_data.b.size(), 3);
            for (int ii=0; ii<cube_data.b.size(); ii++)
              cube_data.bc.row(ii) = V.row(cube_data.b(ii));
            cube_style_precomputation(V, F, cube_data);
          }
        }
        state.place_constraints = false;
        state.running = !state.running;
        break;
      }
      case 'D':
      case 'd': {
        // delete current normal/ring
        if (sel != -1){
          if (gauss_data.style_R[state.group].size() > 0){
            // delete ring
            for (int i = sel+1; i < gauss_data.style_R[state.group].size(); ++i)
            {
              gauss_data.style_R[state.group](i-1) = gauss_data.style_R[state.group](i);
            }
            gauss_data.style_R[state.group].conservativeResize(gauss_data.style_R[state.group].size()-1);
          } else {
            // delete normal
            for (int i = sel+1; i < gauss_data.style_N[state.group].rows(); ++i)
            {
              gauss_data.style_N[state.group].row(i-1) = gauss_data.style_N[state.group].row(i);
            }
            gauss_data.style_N[state.group].conservativeResize(gauss_data.style_N[state.group].rows()-1,gauss_data.style_N[state.group].cols());
          }
        }
        sel = -1;
        normalize_g(gauss_data);
        break;
      }
      default:
        return false;
    }
    draw();
    return true;
  };

  // on resize
  viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer &v, int w, int h) {
    if (state.viewGfunc){
      v.core().viewport = Vector4f(0, 0, w / 2, h);
      v.core(right_view).viewport = Vector4f(w / 2, 0, w - (w / 2), h);
    } else {
      v.core().viewport = Vector4f(0, 0, w, h);
      v.core(right_view).viewport = Vector4f(w, 0, 0, h);
    }
    return true;
  };

  // default mode: keep drawing the current mesh
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &)->bool {
    if(viewer.core().is_animating)
        draw();
    return false;
  };

  // mouse interactions
  RowVector3f last_mouse;
  bool add_normals = false;
  // when click mouse
  viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer&, int, int)->bool {
    // add constraint (click on the left side of the window)
    if(state.place_constraints && viewer.current_mouse_x < viewer.core().viewport(2)){
      // Find closest point on mesh to mouse position
      double x = viewer.current_mouse_x;
      double y = viewer.core().viewport(3) - viewer.current_mouse_y;
      int fid;
      Vector3f bary;
      if(igl::unproject_onto_mesh(Vector2f(x,y), viewer.core().view, viewer.core().proj, viewer.core().viewport, V, F, fid, bary)) {
        long c;
        bary.maxCoeff(&c);

        state.CV_row_col.row(state.numCV) << fid, c;
        state.numCV++;

        RowVector3d new_c = V.row(F(fid,c));
        if(state.CV.size()==0 || (state.CV.rowwise()-new_c).rowwise().norm().minCoeff() > 0)
        {
            state.CV.conservativeResize(state.CV.rows()+1,3);
            state.CV.row(state.CV.rows()-1) = new_c;
            draw();
            return true;
        }
      }
    }
    // Add or change something on g function (click on the right side of the window)
    else if(state.viewGfunc && viewer.current_mouse_x > viewer.core().viewport(2)){
      // Find closest point on mesh to mouse position
      double x = viewer.current_mouse_x;
      double y = viewer.core().viewport(3) - viewer.current_mouse_y;
      last_mouse = RowVector3f(x,y,0);
      int fid;
      Vector3f bary;
      if(add_normals){
        if(igl::unproject_onto_mesh(Vector2f(x,y), viewer.core(right_view).view, viewer.core(right_view).proj, viewer.core(right_view).viewport, Disp, SphereF, fid, bary)) {
          VectorXi face = SphereF.row(fid);
          VectorXd v = SphereV.row(face(0))*bary(0)+SphereV.row(face(1))*bary(1)+SphereV.row(face(2))*bary(2);
          v = v.normalized();
          if (gauss_data.style_R[state.group].size() > 0){
            auto &D = gauss_data.style_R[state.group];
            D.conservativeResize(D.size()+1);
            D(D.size()-1) = v(1);
          } else {
            auto &N = gauss_data.style_N[state.group];
            N.conservativeResize(N.rows()+1, N.cols());
            N.row(N.rows()-1) = v;
          }

          normalize_g(gauss_data);
          draw();
        }
      } else {
        // Move closest control point
        MatrixXf CP;
        MatrixXf what;
        if (gauss_data.style_R[state.group].size() > 0){
          MatrixXd rings = MatrixXd::Zero(gauss_data.style_R[state.group].size(), 3);
          for (size_t i = 0; i < gauss_data.style_R[state.group].size(); i++){
            rings(i,1) = gauss_data.style_R[state.group](i);
            rings(i,0) = sqrt(1-pow(gauss_data.style_R[state.group](i),2));
          }
          what = MatrixXf(rings.cast<float>());
        } else {
          what = MatrixXf(gauss_data.style_N[state.group].cast<float>());
        }
        igl::project(
          what,
          viewer.core(right_view).view,
          viewer.core(right_view).proj,
          viewer.core(right_view).viewport,
          CP);
        VectorXf D = (CP.rowwise()-last_mouse).rowwise().norm();
        sel = (D.minCoeff(&sel) < 30)?sel:-1;
        if(sel != -1) {
          last_mouse(2) = CP(sel,2);
          return true;
        }
      }
    }
    return false;
  };
  // move g-functions
  viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer &, int,int)->bool
  {
    if(sel!=-1){
      double x = viewer.current_mouse_x;
      double y = viewer.core().viewport(3) - viewer.current_mouse_y;
      RowVector3f drag_mouse(x, y, last_mouse(2));
      RowVector3f drag_scene,last_scene;
      igl::unproject(
        drag_mouse,
        viewer.core(right_view).view,
        viewer.core(right_view).proj,
        viewer.core(right_view).viewport,
        drag_scene);
      igl::unproject(
        last_mouse,
        viewer.core(right_view).view,
        viewer.core(right_view).proj,
        viewer.core(right_view).viewport,
        last_scene);
      if (gauss_data.style_R[state.group].size() > 0){
        gauss_data.style_R[state.group](sel) += (drag_scene-last_scene).cast<double>()(1);
        gauss_data.style_R[state.group](sel) = max(-1., min(1., gauss_data.style_R[state.group](sel)));
      } else {
        gauss_data.style_N[state.group].row(sel) += (drag_scene-last_scene).cast<double>();
        gauss_data.style_N[state.group].row(sel) = gauss_data.style_N[state.group].row(sel).normalized();
      }
      last_mouse = drag_mouse;
      normalize_g(gauss_data);
      return true;
    }
    return false;
  };
  viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer&, int, int)->bool
  {
    sel = -1;
    return false;
  };
  // ---------------------------------------------------------------------------
  // -------------------------- GUI Windows ------------------------------------
  // ---------------------------------------------------------------------------
  // draw additional windows
  char outputName[128] = "output";
  menu.callback_draw_viewer_window = []() {};
  menu.callback_draw_custom_window = [&]()
  {
    // Define next window position + size
    {
      ImGui::SetNextWindowPos(ImVec2(0.f * menu.menu_scaling(), 0), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(220, -1), ImGuiCond_FirstUseEver);
      ImGui::Begin(
        "Gauss Stylization", nullptr,
        ImGuiWindowFlags_NoSavedSettings
      );
    }
    {
      // How to use
      ImGui::Text("Instructions");
      ImGui::BulletText("[click]   place constrained points");
      ImGui::BulletText("[space]  start stylization");
      ImGui::BulletText("R           reset ");
      ImGui::BulletText("</>      decrease/increase lambda");
      ImGui::BulletText("Q/W     rotate x-axis");
      ImGui::BulletText("A/S       rotate y-axis");
      ImGui::BulletText("Y/X       rotate z-axis");
      // ImGui::BulletText("N          show/hide g-function");
      // ImGui::Text(" ");
      ImGui::Text("Info:");
      ImGui::Text("Number of Vertices: ");ImGui::SameLine();ImGui::Text("%lu", V.rows());
      ImGui::Text("Number of Faces: ");ImGui::SameLine();ImGui::Text("%lu", F.rows());
    }
    {
      ImGui::PushItemWidth(-80);


      if (ImGui::Combo("Input Mesh", (int*)&state.meshNr, fileNames)) {
        reset();
        loadMesh(state.meshNr);
        draw();
      }
      ImGui::SliderInt("ADMM Iterations", &state.iter, 1, 100);

      if (ImGui::DragScalar("lambda", ImGuiDataType_Double, &gauss_data.lambda, 2e-1, 0, 0, "%.1e")){
        cube_data.lambda = gauss_data.lambda;
      };
      static int e = 1;
      if (ImGui::RadioButton("Gauss stylization", &e, 1)){
        state.gauss = e == 1;
      } if (ImGui::RadioButton("Cubic stylization", &e, 0)){
        state.gauss = e == 1;
      };
    }

    {
      // output file name
      ImGui::InputText(".obj", outputName, IM_ARRAYSIZE(outputName));

      if (ImGui::Button("save output mesh", ImVec2(-1, 0))) {
        string outputFile = OUTPUT_PATH;
        outputFile.append("cubic_");
        outputFile.append(outputName);
        outputFile.append(".obj");

        igl::writeOBJ(outputFile, U, F);

        string inputFile = OUTPUT_PATH;
        inputFile.append("input_");
        inputFile.append(outputName);
        inputFile.append(".obj");
        igl::writeOBJ(inputFile, V, F);
      }

      if (ImGui::Button("save g func mesh", ImVec2(-1, 0))) {
        string outputFile = OUTPUT_PATH;
        outputFile.append("g_function");
        outputFile.append(outputName);
        outputFile.append(".obj");
        igl::writeOBJ(outputFile, Disp, SphereF);
      }
    }
    ImGui::End();
  };
  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing;
  groupmenu.callback_draw_viewer_window = []() {};
  groupmenu.callback_draw_custom_window = [&](){
    {
      ImGui::SetNextWindowSizeConstraints(ImVec2(300, -1),ImVec2(300, -1));
      ImGui::SetNextWindowPos(ImVec2(viewer.core().viewport(2), 0), ImGuiCond_Always,ImVec2(1,0));
      ImGui::SetNextWindowSize(ImVec2(300, -1), ImGuiCond_Always);
      ImGui::Begin(
        "group", nullptr,
        window_flags
      );
    }
    // How to use
    ImGui::Text("Style Groups:");
    auto txt = "";
    for (size_t k = 0; k < gauss_data.n_weights.size(); k++) {
      if (k == state.group){
        txt = "*Group ";
      } else {
        txt = "Group ";
      }
      ImVec4 color = ImVec4((float)cp1(k,0),(float)cp1(k,1),(float)cp1(k,2), 1.);
      ImGui::ColorEdit4("", (float*)&color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
      ImGui::SameLine();
      if (ImGui::Button((txt+std::to_string(k)).c_str(), ImVec2(180, 0))) {
        state.group = k;
        draw();
      }
      if (k != 0 || gauss_data.n_weights.size() > 1){
        ImGui::SameLine();
        if (ImGui::Button(("x##"+std::to_string(k)).c_str(), ImVec2(20, 0))) {
          if (state.group >= k){
            state.group = state.group-1;
          }
          remove_style(k);
          draw();
        }
      }
    }
    if (ImGui::Button("Add Group", ImVec2(-1, 0))) {
      add_style(solid_normals::cube(),Vempty);
      state.group = gauss_data.n_weights.size()-1;
    }

    ImGui::Separator();

    if (ImGui::Button("SELECT Region", ImVec2(-1, 0))){
      selection_plugin.mode=selection_plugin.RECTANGULAR_MARQUEE;
      draw();
    };

  };

  stylemenu.callback_draw_viewer_window = []() {};
  stylemenu.callback_draw_custom_window = [&](){
    {
      ImGui::SetNextWindowSizeConstraints(ImVec2(300, -1),ImVec2(300, -1));
      ImGui::SetNextWindowPos(ImVec2(viewer.core().viewport(2), 0), ImGuiCond_Always,ImVec2(0,0));
      ImGui::SetNextWindowSize(ImVec2(300, -1), ImGuiCond_Always);
      ImGui::Begin(
        "Styles", nullptr,
        window_flags
      );
    }
    {
      // How to use
      ImGui::Text("Instructions");
      ImGui::BulletText("[drag]    Move normals or rings");
      ImGui::BulletText("[drag]+D  delete current normal or ring");
      if (add_normals){
        ImGui::BulletText("[click]   Add normals or rings");
      }
      auto txt = "";
      if (gauss_data.r_weights[state.group].size() > 0){
        txt = "Add Rings";
      } else {
        txt = "Add Normals";
      }
      ImGui::Checkbox(txt, &add_normals);
      ImGui::Text("g(n) parameters:");
      if (ImGui::DragScalar("mu", ImGuiDataType_Double, &gauss_data.mu[state.group], 2e-1, 0, 0, "%.1e")){
        draw();
      }
      if (ImGui::DragScalar("sigma", ImGuiDataType_Double, &gauss_data.sigma[state.group], 2e-1, 0, 0, "%.1e")){
        normalize_g(gauss_data);
        draw();
      }
      if (gauss_data.r_weights[state.group].size() > 0){
        if (ImGui::DragScalar("caxiscontrib", ImGuiDataType_Double, &gauss_data.caxiscontrib[state.group], 2e-1, 0, 0, "%.1e")){
          draw();
        }
        ImGui::Text("Axis: ");
        ImGui::SameLine();
        if (ImGui::Button("Top")) {
          gauss_data.style_N[state.group] = solid_normals::axis().row(0);
          normalize_g(gauss_data);
          draw();
        }ImGui::SameLine();
        if (ImGui::Button("Bottom")) {
          gauss_data.style_N[state.group] = solid_normals::axis().row(1);
          normalize_g(gauss_data);
          draw();
        }ImGui::SameLine();
        if (ImGui::Button("Both")) {
          gauss_data.style_N[state.group] = solid_normals::axis();
          normalize_g(gauss_data);
          draw();
        }ImGui::SameLine();
        if (ImGui::Button("None")) {
          gauss_data.style_N[state.group] = MatrixXd(0,3);
          normalize_g(gauss_data);
          draw();
        }
      }
      ImGui::Text("Reset style to:");
      int style;
      if (ImGui::Combo("Style", &style, "Cube\0Pyramid\0Tetrahedron\0Octrahedron\0Plane\0Dodecahedron\0Icosahedron\0Rhomboid Prism\0Truncated pyramid\0Triangular prism\0Hex prism\0Hex pyramid\0\0")) {
        MatrixXd normals;
        normals = Normals[style]();

        // g needs to be normalized whenever the set of normals changes
        gauss_data.style_N[state.group] = normals;
        gauss_data.style_R[state.group] = Vempty;
        normalize_g(gauss_data);
        draw();
      }
      ImGui::Text("Semi-discrete:");
      if (ImGui::Button("Cylinder")) {
        // set stylization normals
        gauss_data.style_N[state.group] = solid_normals::axis();
        VectorXd cyl(1);
        cyl << 0;
        gauss_data.style_R[state.group] = cyl;
        normalize_g(gauss_data);
        draw();
      }ImGui::SameLine();
      if (ImGui::Button("Cone")) {
        gauss_data.style_N[state.group] = solid_normals::axis().row(1);
        VectorXd cyl(1);
        cyl << 0.6;
        gauss_data.style_R[state.group] = cyl;
        normalize_g(gauss_data);
        draw();
      }
      ImGui::Separator();
    }
  };

  selection_plugin.callback = [&](){
    screen_space_selection(U,F,tree,viewer.core().view,viewer.core().proj,viewer.core().viewport,selection_plugin.L,W,and_visible);
    update();
    selection_plugin.mode=selection_plugin.OFF;
    state.select_mode=false;
    draw();
  };

  // initialize the scene
  {
    viewer.data().set_mesh(V,F);
    viewer.data().show_lines = false;
    viewer.data().point_size = 5;
    viewer.core().is_animating = true;
    viewer.data().face_based = true;
    Vector4f backColor;
    backColor << 1., 1., 1., 1.;
    viewer.core().background_color = backColor;

    viewer.data(Sphere).set_mesh(SphereV*0.5, SphereF);
    viewer.data(Sphere).show_lines = false;
    viewer.core(right_view).is_animating = true;
    viewer.data(Sphere).set_colors(RowVector3d(1., 1., 1.));

    viewer.data(Sphere).show_lines = false;
    draw();
    selection_plugin.mode=selection_plugin.OFF;
    viewer.launch(true,false,"Gauss Stylization", WIDTH*2, HEIGHT);
  }

}
