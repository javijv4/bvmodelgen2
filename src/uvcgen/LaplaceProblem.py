#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:24:12 2023

@author: Javiera Jilberto Vallejos
"""
import numpy as np
from dolfinx import fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType
from scipy.spatial import KDTree

# https://fenicsproject.discourse.group/t/conversion-from-array-to-petsc-and-from-petsc-to-array/6941
def petsc2array(v):
    s=v.getValues(range(0, v.getSize()[0]), range(0,  v.getSize()[1]))
    return s

def project(v, target_func, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = fem.form(ufl.inner(Pv, w) * dx)
    L = fem.form(ufl.inner(v, w) * dx)

    # Assemble linear system
    A = dolfinx.fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L)
    dolfinx.fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.x.petsc_vec)


class LaplaceProblem:
    def __init__(self, mesh, mt):
        self.V = fem.functionspace(mesh, ("CG", 1))
        self.Vg_lin = fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim, )))
        self.Vg = fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim, )))
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        self.tdim = mesh.topology.dim
        self.fdim = self.tdim - 1
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)

        self.mesh = mesh
        self.mt = mt

        # Problem
        self.f = fem.Constant(self.mesh, ScalarType(0))
        self.a = ufl.dot(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        self.L = self.f * self.v * ufl.dx

    def set_bc(self, bcs_marker):
        """
        Parameters
        ----------
        bcs_marker : dict
            keys are facet markers, values are bc values
        """
        bcs = []
        for btype in bcs_marker.keys():
            for marker in bcs_marker[btype]:
                if btype == 'function':
                    facets = self.mt.find(marker)
                    dofs = fem.locate_dofs_topological(self.V, self.fdim, facets)
                    uD = fem.Function(self.V)
                    uD.x.petsc_vec.array = bcs_marker[btype][marker]
                    bc = fem.dirichletbc(uD, dofs)
                elif btype == 'face':
                    facets = self.mt.find(marker)
                    dofs = fem.locate_dofs_topological(self.V, self.fdim, facets)
                    uD = fem.Constant(self.mesh, ScalarType(bcs_marker[btype][marker]))
                    bc = fem.dirichletbc(uD, dofs, self.V)
                elif btype == 'point':
                    point = np.array(marker)
                    func = lambda x: np.isclose(np.linalg.norm(x-point[:,None], axis=0), 0, rtol=1e-3, atol=1e-4)
                    dofs = fem.locate_dofs_geometrical(self.V, func)
                    uD = fem.Constant(self.mesh, ScalarType(bcs_marker[btype][marker]))
                    bc = fem.dirichletbc(uD, dofs, self.V)
                bcs.append(bc)
        return bcs

    def solve(self, bcs_marker):
        bcs = self.set_bc(bcs_marker)
        problem = dolfinx.fem.petsc.LinearProblem(self.a, self.L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        return uh

    def get_lap_gradient(self, lap):
        # Compute normalized gradient
        grad_lap = ufl.grad(lap)
        g = fem.Function(self.Vg)
        project(grad_lap, g)

        return g


    def get_linear_gradient(self, uh):
        grad = ufl.grad(uh)
        g = fem.Function(self.Vg_lin)
        project(grad, g)

        return g


    def solve_diffusion(self, bcs_marker):
        # Solve first laplace
        bcs = self.set_bc(bcs_marker)
        problem = dolfinx.fem.petsc.LinearProblem(self.a, self.L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        lap = problem.solve()

        # Compute gradient
        # g = self.get_lap_gradient(lap)
        g = ufl.grad(lap)
        gg = ufl.outer(g,g)
        D = gg + 1e-5*(ufl.Identity(3)-gg)  # 1e-5 seems to work good

        a = ufl.dot(ufl.grad(self.u), D*ufl.grad(self.v)) * ufl.dx
        bcs = self.set_bc(bcs_marker)
        problem = dolfinx.fem.petsc.LinearProblem(a, self.L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        return uh

    def get_array_gradient(self, array, linear=False):
        uh = fem.Function(self.V)
        uh.x.petsc_vec.array =  array
        if linear:
            g = self.get_linear_gradient(uh)
        else:
            g = self.get_lap_gradient(uh)

        return g

    def visualize(self, uh, clim=None):
        from dolfinx import plot
        import pyvista as pv
        from pyvistaqt import BackgroundPlotter
        u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(self.V)
        u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
        u_grid.point_data["u"] = uh.x.array.real
        u_grid.set_active_scalars("u")
        u_plotter = BackgroundPlotter()
        u_plotter.add_mesh(u_grid, show_edges=True, n_colors=10, clim=clim)
        u_plotter.view_xy()


class TrajectoryProblem:
    def __init__(self, mesh, mt):
        self.V = fem.functionspace(mesh, ("CG", 1),)
        self.V0 = fem.functionspace(mesh, ("DG", 0),)
        self.Vg = fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim, )))
        self.Vg1 = fem.functionspace(mesh, ("CG", 1, (mesh.geometry.dim, )))
        self.u = ufl.TrialFunction(self.V)
        self.v_lap = ufl.TestFunction(self.V)
        self.v_tp = ufl.TestFunction(self.V0)

        self.tdim = mesh.topology.dim
        self.fdim = self.tdim - 1
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)

        self.mesh = mesh
        self.mt = mt

        # Trajectory Problem
        self.tp_f = fem.Constant(self.mesh, ScalarType(1))
        self.tp_L = self.tp_f * self.v_tp * ufl.dx

        # Laplace Problem
        self.lap_f = fem.Constant(self.mesh, ScalarType(0))
        self.lap_a = ufl.dot(ufl.grad(self.u), ufl.grad(self.v_lap)) * ufl.dx
        self.lap_L = self.lap_f * self.v_lap * ufl.dx

        # Threshold for correcting gradients
        self.grad_thresh = 1e-2


    def set_bc(self, bcs_marker):
        """
        Parameters
        ----------
        bcs_marker : dict
            keys are facet markers, values are bc values
        """
        bcs = []
        for btype in bcs_marker.keys():
            for marker in bcs_marker[btype]:
                if btype == 'function':
                    facets = self.mt.find(marker)
                    dofs = fem.locate_dofs_topological(self.V, self.fdim, facets)
                    uD = fem.Function(self.V)
                    uD.x.petsc_vec.array =  bcs_marker[btype][marker]
                    bc = fem.dirichletbc(uD, dofs)
                elif btype == 'face':
                    facets = self.mt.find(marker)
                    dofs = fem.locate_dofs_topological(self.V, self.fdim, facets)
                    uD = fem.Constant(self.mesh, ScalarType(bcs_marker[btype][marker]))
                    bc = fem.dirichletbc(uD, dofs, self.V)
                elif btype == 'point':
                    point = np.array(marker)
                    func = lambda x: np.isclose(np.linalg.norm(x-point[:,None], axis=0), 0)
                    dofs = fem.locate_dofs_geometrical(self.V, func)
                    uD = fem.Constant(self.mesh, ScalarType(bcs_marker[btype][marker]))
                    bc = fem.dirichletbc(uD, dofs, self.V)
                bcs.append(bc)
        return bcs


    def solve_laplace(self, bcs_marker):
        bcs = self.set_bc(bcs_marker)
        problem = dolfinx.fem.petsc.LinearProblem(self.lap_a, self.lap_L, bcs=bcs,
                                          petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()

        return uh


    def solve_trajectory_problem(self, bcs_marker):
        bcs = []
        a = fem.form(self.tp_a)
        L = fem.form(self.tp_L)

        S = dolfinx.fem.petsc.assemble_matrix(a, bcs=bcs)
        S.assemble()
        b = dolfinx.fem.petsc.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.apply_lifting(b, [a], bcs=[bcs])
        dolfinx.fem.petsc.set_bc(b, bcs)

        A = S.transposeMatMult(S)
        A.assemble()
        x, y = A.createVecs()
        S.multTranspose(b, y)

        # Need to apply BCs to A
        bcs = self.set_bc(bcs_marker)

        for bc in bcs:
            dofs = bc.dof_indices()[0]
            A.zeroRowsColumns(dofs)
            y.array[dofs] = bc.value.value

        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOperators(A)
        solver.solve(y, x)

        return x


    def get_lap_gradient(self, lap):
        # Compute normalized gradient
        grad_lap = ufl.grad(lap)
        g = fem.Function(self.Vg)
        project(grad_lap, g)
        garr = g.x.petsc_vec.reshape([-1,3])
        norm_g = np.linalg.norm(garr, axis=1)

        # normalizing
        below_thresh = norm_g < self.grad_thresh
        norm_g[below_thresh] = 1
        t = garr/norm_g[:,None]

        # correcting
        if np.any(below_thresh):
            midpoints = self.Vg.tabulate_dof_coordinates()
            tree = KDTree(midpoints[~below_thresh])
            _, nn = tree.query(midpoints[below_thresh])
            t[below_thresh] = t[nn]

        g.x.petsc_vec.array =  t.flatten()

        return g


    def solve(self, bcs_marker):
        # First solve laplace
        lap = self.solve_laplace(bcs_marker)

        # Compute normalized gradient
        grad = ufl.grad(lap)
        norm = ufl.sqrt(grad[0]*grad[0]+grad[1]*grad[1]+grad[2]*grad[2])
        g = grad/norm
        g = self.get_lap_gradient(lap)

        # Set problem
        self.tp_a = ufl.dot(ufl.grad(self.u), g) * self.v_tp * ufl.dx

        # First problem
        keys = np.array(list(bcs_marker['face'].keys()))
        values = np.array(list(bcs_marker['face'].values()))
        mark1 = keys[values==0]
        bc1 = {'face': dict(zip(mark1, [0]*len(mark1)))}
        x1 = self.solve_trajectory_problem(bc1)

        # # Second problem
        self.tp_a = -ufl.dot(ufl.grad(self.u), g) * self.v_tp * ufl.dx
        mark2 = keys[values==1]
        bc2 = {'face': dict(zip(mark2, [0]*len(mark2)))}
        x2 = self.solve_trajectory_problem(bc2)

        uh = fem.Function(self.V)
        uh.x.petsc_vec.array =  x1/(x1 + x2)

        return uh


    def solve_with_vector(self, bcs_marker, vector):
        g = fem.Function(self.Vg)
        g.x.petsc_vec.array =  vector.flatten()

        # Set problem
        self.tp_a = ufl.dot(ufl.grad(self.u), g) * self.v_tp * ufl.dx

        # First problem
        keys = np.array(list(bcs_marker['face'].keys()))
        values = np.array(list(bcs_marker['face'].values()))
        mark1 = keys[values==0]
        bc1 = {'face': dict(zip(mark1, [0]*len(mark1)))}
        x1 = self.solve_trajectory_problem(bc1)

        # # Second problem
        self.tp_a = -ufl.dot(ufl.grad(self.u), g) * self.v_tp * ufl.dx
        mark2 = keys[values==1]
        bc2 = {'face': dict(zip(mark2, [0]*len(mark2)))}
        x2 = self.solve_trajectory_problem(bc2)

        uh = fem.Function(self.V)
        uh.x.petsc_vec.array =  x1/(x1 + x2)

        return uh

    def visualize(self, uh, clim=None):
        from dolfinx import plot
        import pyvista as pv
        from pyvistaqt import BackgroundPlotter
        u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(self.V)
        u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
        u_grid.point_data["u"] = uh.x.array.real
        u_grid.set_active_scalars("u")
        u_plotter = BackgroundPlotter()
        u_plotter.add_mesh(u_grid, show_edges=True, n_colors=10, clim=clim)
        u_plotter.view_xy()
