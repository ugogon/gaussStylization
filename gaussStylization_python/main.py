import numpy as np
import matplotlib.pyplot as plt
# make sure __init__.py is present in the build folder
import build.gauss_stylization as gs


def log_energies(data, max_iterations = -1, ADMM_iterations=1, mesh="../meshes/bunny.obj"):
    res = []

    # load mesh
    U,V,F = gs.loadMesh(mesh)
    # reset data
    data.reset()
    gs.init_constraints(V,F,data)
    # do precomputation
    gs.gauss_style_precomputation(V,F,data)

    # calculate inital energies
    Ea = gs.arap_energy(U, data)
    normals = gs.calc_normals(U, F)
    Eo = gs.orginal_energy_without_arap(U, data, normals)
    Ec = gs.coupling_energy_without_arap(U, data, data.nf_stars, data.e_ij_stars)
    res.append([Ea,Eo,Ec])

    oldEo = Eo+1
    while abs(Eo-oldEo) > 0.001 and max_iterations != 0:
        oldEo = Eo
        # do one interation
        gs.gauss_style_single_iteration(V, U, F, data, ADMM_iterations)

        Ea = gs.arap_energy(U, data)
        normals = gs.calc_normals(U, F)
        Eo = gs.orginal_energy_without_arap(U, data, normals)
        Ec = gs.coupling_energy_without_arap(U, data, data.nf_stars, data.e_ij_stars)
        res.append([Ea,Eo,Ec])

        max_iterations -= 1
    return np.array(res)

# initialize data object
data = gs.data()
data.lamda = 4
# normal directions of g
N = np.array([[0, 0, 1],[0, 0, -1],[0, 1, 0],[0, -1, 0],[1, 0, 0],[-1, 0, 0]])
# rings of g
R = np.array([])
# add style with default parameters
gs.add_style(N,R,data)

# run 100 iterations max with 6 ADMM steps
ns1 = log_energies(data, max_iterations=100, ADMM_iterations=6, mesh="../meshes/bunny.obj")
plt.title("6 ADMM iterations on bunny.obj")
plt.ylabel("Energy (Eq. 4)")
plt.xlabel("Iteration")
plt.plot(ns1[:,0]+ns1[:,1])
plt.show()
