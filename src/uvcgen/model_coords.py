#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:32:08 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import dolfinxio as dxio
from uvcgen.LaplaceProblem import LaplaceProblem, TrajectoryProblem
import uvcgen.UVC as uc
import meshio as io
from dolfinx.log import set_log_level, LogLevel
from scipy.interpolate import interp1d

set_log_level(LogLevel.WARNING)

def run_coord(mesh_, bdata_, bcs_marker, return_grad=False, diffusion=False):

    mesh, mt = dxio.read_meshio_mesh(mesh_, bdata_)
    LapSolver = LaplaceProblem(mesh, mt)

    if diffusion:
        # lap = LapSolver.solve_diffusion(bcs_marker)
        lap = LapSolver.solve(bcs_marker)
    else:
        lap = LapSolver.solve(bcs_marker)

    corr, _ = dxio.find_vtu_dx_mapping(mesh)
    if return_grad:
        grad = LapSolver.get_linear_gradient(lap)
        dxio.visualize_function('grad.xdmf', grad)
        grad = grad.vector.array.reshape([-1, 3])[corr]
        lap = lap.vector.array[corr]
        return lap, grad

    return lap.vector.array[corr]


def run_trajectory_coord(mesh_, bdata_, bcs_marker, vector=None):

    mesh, mt = dxio.read_meshio_mesh(mesh_, bdata_)
    TPSolver = TrajectoryProblem(mesh, mt)

    if vector is not None:
        lap = TPSolver.solve_with_vector(bcs_marker, vector)
    else:
        lap = TPSolver.solve(bcs_marker)

    corr, _ = dxio.find_vtu_dx_mapping(mesh)
    lap = lap.vector.array[corr]

    return lap


class UVCGen:
    def __init__(self, uvc, mmg=False):
        self.xyz_og = uvc.xyz
        self.bndry = uvc.patches
        self.mmg = mmg

    def get_solver(self, method, mesh='bv'):
        if method == 'laplace':
            if mesh == 'bv':
                return self.LapSolver
            elif mesh == 'lv':
                return self.lv_LapSolver
            elif mesh == 'rv':
                return self.rv_LapSolver
            elif mesh == 'ot':
                return self.ot_LapSolver
        elif method == 'trajectory':
            if mesh == 'bv':
                return self.TPSolver
            elif mesh == 'lv':
                return self.lv_TPSolver
            elif mesh == 'rv':
                return self.rv_TPSolver
            elif mesh == 'ot':
                return self.ot_TPSolver
        else:
            raise ('Method unknown, please choose laplace or trajectory')

    def init_bv_mesh(self, uvc, og=False):
        if og:
            bdata = uvc.bv_bdata_og
        else:
            bdata = uvc.bv_bdata
        self.bv_mesh, self.bv_mt = dxio.read_meshio_mesh(
            uvc.bv_mesh, bdata)

        self.LapSolver = LaplaceProblem(self.bv_mesh, self.bv_mt)
        # self.TPSolver = TrajectoryProblem(self.bv_mesh, self.bv_mt)
        self.bv_corr, self.bv_icorr = dxio.find_vtu_dx_mapping(self.bv_mesh)

    def init_rv_mesh(self, uvc, og=False):
        if og:
            bdata = uvc.rv_bdata_og
        else:
            bdata = uvc.rv_bdata
        self.rv_mesh, self.rv_mt = dxio.read_meshio_mesh(
            uvc.rv_mesh, bdata)
        self.rv_LapSolver = LaplaceProblem(self.rv_mesh, self.rv_mt)
        # self.rv_TPSolver = TrajectoryProblem(self.rv_mesh, self.rv_mt)
        self.rv_corr, self.rv_icorr = dxio.find_vtu_dx_mapping(self.rv_mesh)

    def init_lv_mesh(self, uvc, og=False):
        if og:
            bdata = uvc.lv_bdata_og
        else:
            bdata = uvc.lv_bdata
        self.lv_mesh, self.lv_mt = dxio.read_meshio_mesh(
            uvc.lv_mesh, bdata)
        self.lv_LapSolver = LaplaceProblem(self.lv_mesh, self.lv_mt)
        # self.lv_TPSolver = TrajectoryProblem(self.lv_mesh, self.lv_mt)
        self.lv_corr, self.lv_icorr = dxio.find_vtu_dx_mapping(self.lv_mesh)

    def init_ot_mesh(self, uvc, og=False):
        if og:
            bdata = uvc.ot_bdata_og
        else:
            bdata = uvc.ot_bdata
        self.ot_mesh, self.ot_mt = dxio.read_meshio_mesh(
            uvc.ot_mesh, bdata)
        self.ot_LapSolver = LaplaceProblem(self.ot_mesh, self.ot_mt)
        # self.ot_TPSolver = TrajectoryProblem(self.ot_mesh, self.ot_mt)
        self.ot_corr, self.ot_icorr = dxio.find_vtu_dx_mapping(self.ot_mesh)


    def get_func_gradient(self, uvc, func, which, linear=False):
        if which == 'lv':
            self.init_lv_mesh(uvc)
            array = uvc.lv_mesh.point_data[func][self.lv_icorr]
            if linear:
                corr, _ = dxio.find_vtu_dx_mapping(self.lv_mesh)
            else:
                corr, _ = dxio.find_vtu_dx_mapping(self.lv_mesh, cells=True)
            g = self.lv_LapSolver.get_array_gradient(array, linear=linear)
            g = g.vector.array.reshape([-1,3])[corr]

        return g

    def run_septum(self, uvc):
        self.init_bv_mesh(uvc)

        # Run septum problem
        bcs_septum = {'face': {self.bndry['lv_endo']: 0,
                      self.bndry['mv']: 0,
                      self.bndry['av']: 0,
                      self.bndry['rv_septum']: 0.5,
                      self.bndry['rv_endo']: 1,
                      self.bndry['tv']: 1,
                      self.bndry['pv']: 1}}

        if uvc.split_epi:
            bcs_septum['face'][self.bndry['lv_epi']] = 0
            bcs_septum['face'][self.bndry['rv_epi']] = 1

        septum = self.LapSolver.solve(bcs_septum)
        septum = septum.vector.array[self.bv_corr]
        uvc.bv_mesh.point_data['septum'] = septum

        return septum

    def correct_septum(self, uvc, septum0):
        self.init_bv_mesh(uvc)
        uvc.define_septum_bc(septum0)
        self.init_bv_mesh(uvc)

        bcs_septum = {'face': {self.bndry['lv_endo']: 0,
                      12: 0,
                      self.bndry['mv']: 0,
                      self.bndry['av']: 0,
                      self.bndry['rv_septum']: 0.5,
                      self.bndry['rv_endo']: 1,
                      self.bndry['tv']: 1,
                      self.bndry['pv']: 1}}

        septum = self.LapSolver.solve(bcs_septum)
        uvc.bv_mesh.point_data['septum'] = septum

        return septum.vector.array[self.bv_corr]


    def run_longitudinal(self, uvc, method='laplace', which='all'):
        ret = []

        if which == 'all' or which =='bv':
            bcs_point = {}
            for i in range(3):
                bcs_point[tuple(uvc.bv_mesh.points[uvc.bv_sep_apex_nodes[i]])] = 0
            bcs_marker = {'point': bcs_point,
                          'face': {self.bndry['av']: 1.0, self.bndry['mv']: 1.0,
                                    self.bndry['pv']: 1.0, self.bndry['tv']: 1.0}}
            self.init_bv_mesh(uvc)
            bv_solver = self.get_solver(method, 'bv')
            # bv_long = bv_solver.solve_diffusion(bcs_marker)
            bv_long = bv_solver.solve(bcs_marker)
            bv_long = bv_long.vector.array[self.bv_corr]
            uvc.bv_mesh.point_data['long'] = bv_long
            ret += [bv_long]

        if which == 'all' or which =='lv':
            bcs_point = {}
            for i in range(3):
                bcs_point[tuple(uvc.bv_mesh.points[uvc.bv_sep_apex_nodes[i]])] = 0
            bcs_marker = {'point': bcs_point,
                          'face': {self.bndry['av']: 1.0, self.bndry['mv']: 1.0,
                                    self.bndry['pv']: 1.0, self.bndry['tv']: 1.0}}
            self.init_lv_mesh(uvc)
            lv_solver = self.get_solver(method, 'lv')
            # lv_long = lv_solver.solve_diffusion(bcs_marker)
            lv_long = lv_solver.solve(bcs_marker)
            lv_long = lv_long.vector.array[self.lv_corr]
            uvc.lv_mesh.point_data['long'] = lv_long
            ret += [lv_long]

        return ret

    def correct_longitudinal(self, uvc):
        mesh = self.lv_mmg_mesh
        mmg_bdata = self.lv_mmg_bdata
        bc = self.lv_mmg_bc

        sep_nodes = np.where(bc == 1)[0]
        mpi_nodes = np.where(bc == 2)[0]
        zero_nodes = np.where(bc == 10)[0]

        # Running longitudinal coordinate
        av_nodes = np.unique(uvc.lv_bdata[uvc.lv_bdata[:,-1]==self.bndry['av'],1:-1])
        mv_nodes = np.unique(uvc.lv_bdata[uvc.lv_bdata[:,-1]==self.bndry['mv'],1:-1])

        bcs_point = {}
        for i in range(len(zero_nodes)):
            bcs_point[tuple(mesh.points[zero_nodes[i]])] = 0

        bdata = self.point_to_face_marker(
            av_nodes, self.bndry['av'], new_faces=True, mesh=mesh)
        bdata = self.point_to_face_marker(
            mv_nodes, self.bndry['mv'], bdata, new_faces=True, mesh=mesh)

        long = run_coord(mesh, mmg_bdata, {'point': bcs_point, 'face': {self.bndry['av']: 1, self.bndry['mv']: 1}})
        long[long > 1.0] = 1.0  # In case of numerical error
        long[long < 0.0] = 0.0

        # Getting zero nodes
        rv_sep_nodes = np.unique(mmg_bdata[mmg_bdata[:,-1]==self.bndry['rv_septum'],1:-1])
        rv_lvrv_nodes = np.unique(mmg_bdata[mmg_bdata[:,-1]==self.bndry['rv_lv_junction'],1:-1])
        rv_sep_nodes = np.union1d(rv_sep_nodes, rv_lvrv_nodes)
        sep_nodes = np.union1d(sep_nodes, zero_nodes)
        long_nodes = np.intersect1d(rv_sep_nodes, sep_nodes)
        if uvc.split_epi:
            epi_nodes = np.unique(mmg_bdata[mmg_bdata[:,-1]==self.bndry['lv_epi'],1:-1])
        else:
            epi_nodes = np.unique(mmg_bdata[mmg_bdata[:,-1]==self.bndry['epi'],1:-1])
        epi_nodes = np.union1d(epi_nodes, rv_lvrv_nodes)
        lat_nodes = np.union1d(mpi_nodes, zero_nodes)
        long_nodes = np.intersect1d(epi_nodes, lat_nodes)

        # Compute cartesian distance from node to node and correct long
        order = np.argsort(long[long_nodes])
        points_long = mesh.points[long_nodes[order]]
        dist = np.linalg.norm(np.diff(points_long, axis=0), axis=1)
        dist = np.append(0, np.cumsum(dist))
        norm_dist = dist/dist[-1]

        norm_func = interp1d(long[long_nodes[order]], norm_dist, fill_value='extrapolate')
        norm_long = norm_func(long)

        uvc.lv_mesh.point_data['long'] = norm_long[:len(uvc.lv_mesh.points)]

        # Run rv
        # pass lv_long to rv mesh
        lv_long = np.zeros(len(uvc.bv_mesh.points))
        lv_long[uvc.map_lv_bv] = uvc.lv_mesh.point_data['long']
        uvc.bv_mesh.point_data['long_bc'] = lv_long
        rv_lv_long = np.zeros(len(uvc.rv_mesh.points))
        rv_lv_long[uvc.map_bv_rv[uvc.map_bv_rv>=0]] = lv_long[uvc.map_bv_rv>=0]

        mmg_rv_long = np.zeros(len(self.rv_mmg_mesh.points))
        mmg_rv_long[0:len(rv_lv_long)] = rv_lv_long
        self.rv_mmg_mesh.point_data['long_bc'] = mmg_rv_long
        uvc.rv_mesh.point_data['long_bc'] = rv_lv_long


        zero_nodes = np.where(self.rv_bc == 3)[0]
        mesh = dxio.read_meshio_mesh(self.rv_mmg_mesh)
        _, mmg_rv_icorr = dxio.find_vtu_dx_mapping(mesh)
        bcs_marker = {'face': {self.bndry['av']: 1.0, self.bndry['mv']: 1.0,
                                self.bndry['pv']: 1.0, self.bndry['tv']: 1.0},
                      'function': {self.bndry['rv_lv_junction']: mmg_rv_long[mmg_rv_icorr]}}

        long = run_coord(self.rv_mmg_mesh, self.rv_mmg_bdata, bcs_marker)
        self.rv_mmg_mesh.point_data['long'] = long
        # io.write('mmg_mesh.vtu', self.rv_mmg_mesh)

        # plt.figure(1,clear=True)
        # plt.plot(long[zero_nodes], self.rv_mmg_mesh.point_data['long_plane'][zero_nodes], '.')

        # rv_solver = self.get_solver('laplace', 'rv')
        # rv_long = rv_solver.solve(bcs_marker)
        # rv_long = rv_long.vector.array[self.rv_corr]

        # dxio.visualize_meshtags('mt.xdmf', self.rv_mesh, self.rv_mt)
        # uvc.rv_mesh.point_data['long'] = rv_long


    def run_fast_longitudinal(self, uvc, method='laplace'):
        bcs_point = {}
        for i in range(len(uvc.lv_zero_nodes)):
            bcs_point[tuple(uvc.lv_mesh.points[uvc.lv_zero_nodes[i]])] = 0
        bcs_marker = {'point': bcs_point,
                      'face': {self.bndry['av']: 1.0, self.bndry['mv']: 1.0,
                                self.bndry['pv']: 1.0, self.bndry['tv']: 1.0}}
        self.init_lv_mesh(uvc)
        lv_solver = self.get_solver(method, 'lv')
        lv_long = lv_solver.solve(bcs_marker)
        lv_long = lv_long.vector.array[self.lv_corr]
        uvc.lv_mesh.point_data['long'] = lv_long

        # Correct
        bdata = uvc.lv_bdata_og
        epi_nodes = np.unique(bdata[bdata[:,-1]==self.bndry['lv_epi'],1:-1])
        rvlv_ant_nodes = np.unique(bdata[bdata[:,-1]==self.bndry['rvlv_ant'],1:-1])
        rvlv_post_nodes = np.unique(bdata[bdata[:,-1]==self.bndry['rvlv_post'],1:-1])
        epi_nodes = np.union1d(epi_nodes, rvlv_ant_nodes)
        epi_nodes = np.union1d(epi_nodes, rvlv_post_nodes)
        lat_nodes = np.unique(bdata[bdata[:,-1]==self.bndry['lat0'],1:-1])
        lat_nodes = np.union1d(lat_nodes, uvc.lv_zero_nodes)

        long_nodes = np.intersect1d(epi_nodes, lat_nodes)

        # Compute cartesian distance from node to node and correct long
        order = np.argsort(lv_long[long_nodes])
        points_long = uvc.lv_mesh.points[long_nodes[order]]
        dist = np.linalg.norm(np.diff(points_long, axis=0), axis=1)
        dist = np.append(0, np.cumsum(dist))
        norm_dist = dist/dist[-1]

        norm_func = interp1d(lv_long[long_nodes[order]], norm_dist)
        norm_long = norm_func(lv_long)
        uvc.lv_mesh.point_data['long'] = norm_long[:len(uvc.lv_mesh.points)]

        return lv_long, None



    def run_transmural(self, uvc, method='laplace', which='all'):
        ret = []
        if which == 'all' or which =='lv':
            # Run LV transmural problem
            self.init_lv_mesh(uvc, og=True)
            lv_solver = self.get_solver(method, 'lv')
            bcs_lv = {'face': {self.bndry['lv_endo']: 0,
                      self.bndry['rv_septum']: 1,
                      self.bndry['rv_endo']: 1}}
            if 'rv_lv_junction' in self.bndry:
                bcs_lv['face'][self.bndry['rv_lv_junction']] = 1
            else:
                bcs_lv['face'][self.bndry['rvlv_ant']] = 1
                bcs_lv['face'][self.bndry['rvlv_post']] = 1

            if 'lv_apex' in self.bndry:
                bcs_lv['face'][self.bndry['lv_apex']] = 0
            if uvc.split_epi:
                bcs_lv['face'][self.bndry['lv_epi']] = 1
            else:
                bcs_lv['face'][self.bndry['epi']] = 1

            trans_lv = lv_solver.solve(bcs_lv)
            trans_lv = trans_lv.vector.array[self.lv_corr]
            uvc.lv_mesh.point_data['trans'] = trans_lv

            ret += [trans_lv]

        if which == 'all' or which =='rv':
            # Run RV transmural problem
            self.init_rv_mesh(uvc, og=True)
            rv_solver = self.get_solver(method, 'rv')
            bcs_rv = {'face': {self.bndry['rv_endo']: 0}}
            if 'rv_apex' in self.bndry:
                bcs_rv['face'][self.bndry['rv_apex']] = 0
            if uvc.split_epi:
                bcs_rv['face'][self.bndry['rv_epi']] = 1
            else:
                bcs_rv['face'][self.bndry['epi']] = 1
            trans_rv = rv_solver.solve(bcs_rv)
            trans_rv = trans_rv.vector.array[self.rv_corr]
            uvc.rv_mesh.point_data['trans'] = trans_rv

            ret += [trans_rv]
        return ret

    @staticmethod
    def point_to_face_marker(point_marker, new_value, bdata=None, new_faces=False, mesh=None):

        if not new_faces:
            mark = np.isin(bdata[:, 1:-1], point_marker)
            face_marker = np.where(np.all(mark, axis=1))
            if bdata is None:
                raise ('if not adding faces, you need to provide bdata')
            bdata[face_marker, -1] = new_value
        else:
            if mesh is None:
                raise ('if new faces you need to pass the mesh')

            ien = mesh.cells[0].data
            arr = np.array([[0,1,2],[1,2,3],[2,3,0],[0,3,1]])
            tet_elem = np.repeat(np.arange(len(ien)), 4)
            faces = np.vstack(ien[:,arr])

            mark = np.isin(faces, point_marker)
            bfaces = np.where(np.sum(mark, axis=1) == 3)[0]

            new_faces = faces[bfaces]

            if bdata is None:
                new_bdata = np.vstack([tet_elem[bfaces],
                                       new_faces.T,
                                       np.ones(len(bfaces), dtype=int)*new_value]).T
                bdata = new_bdata
            else:
                # If bdata exists I always want to keep the old bdata
                og_faces = bdata[:,1:-1]
                new_og_faces = np.vstack([og_faces, new_faces])
                sort_new_og_faces = np.sort(new_og_faces, axis=1)
                new_og_elems = np.append(bdata[:,0], tet_elem[bfaces])
                sort_new_og_faces = np.vstack([new_og_elems, sort_new_og_faces.T]).T
                un, inv = np.unique(sort_new_og_faces,
                                    axis=0, return_inverse=True)

                ind = inv[inv>len(bdata)-1]
                new_bdata = np.hstack([new_og_elems[ind][:, None], new_og_faces[ind],
                                        np.ones([len(ind), 1], dtype=int)*new_value])
                bdata = np.vstack([bdata, new_bdata])

        return bdata

    def run_lv_circumferential1(self, uvc, method):
        uvc.create_lv_circ_bc1()

        # First, we need to define the point bc as face bc
        bc_point = uvc.lv_mesh.point_data['bc1']
        ant_marker = np.where(bc_point == 1)[0]
        pos_marker = np.where(bc_point == -1)[0]

        self.bndry['rv_lv_ant'] = 21
        self.bndry['rv_lv_post'] = 22
        uvc.lv_bdata = self.point_to_face_marker(
            ant_marker, self.bndry['rv_lv_ant'], uvc.lv_bdata)
        uvc.lv_bdata = self.point_to_face_marker(
            pos_marker, self.bndry['rv_lv_post'], uvc.lv_bdata)

        self.init_rv_mesh(uvc)
        self.init_lv_mesh(uvc)

        # Run LV circumferential problem
        bcs_lv = {'face': {self.bndry['rv_lv_ant']: 1,
                  self.bndry['rv_lv_post']: -1}}
        lv_solver = self.get_solver(method, 'lv')
        lv_circ1 = lv_solver.solve(bcs_lv)
        lv_circ1 = lv_circ1.vector.array[self.lv_corr]
        uvc.lv_mesh.point_data['lv_circ1'] = lv_circ1

        return lv_circ1

    def run_lv_circumferential2(self, lv_circ1, uvc, method):
        if self.mmg:
            mesh, bc = uc.mmg_create_lv_circ_bc2(lv_circ1, uvc)
            sep_nodes = np.where(bc == 1)[0]
            mpi_nodes = np.where(bc == 2)[0]
            bdata = self.point_to_face_marker(
                sep_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                mpi_nodes, 2, bdata, new_faces=True, mesh=mesh)

            # Need to figure out a better way to map...
            circ = run_coord(mesh, bdata, {'face': {1: 0, 2: 1}})

            # mesh.point_data['circ'] = circ
            # io.write('check.vtu', mesh)
            lv_circ2 = circ[:len(uvc.lv_mesh.points)]


        else:
            uvc.create_lv_circ_bc2(lv_circ1)

            bc_point = uvc.lv_mesh.point_data['bc2']
            lat_marker_neg = uvc.lv_bc_mpi
            sep_marker_neg = uvc.lv_bc_0n
            lat_marker_pls = uvc.lv_bc_pi
            sep_marker_pls = uvc.lv_bc_0p

            self.bndry['lat_pi'] = 23
            self.bndry['lat_mpi'] = 24
            self.bndry['sep_0n'] = 25
            self.bndry['sep_0p'] = 26
            uvc.lv_bdata = self.point_to_face_marker(lat_marker_pls,
                                                     self.bndry['lat_pi'], uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(lat_marker_neg,
                                                     self.bndry['lat_mpi'], uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(sep_marker_pls,
                                                     self.bndry['sep_0p'], uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(sep_marker_neg,
                                                     self.bndry['sep_0n'], uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)

            self.init_rv_mesh(uvc)
            self.init_lv_mesh(uvc)

            bcs_lv = {'function': {self.bndry['lat_pi']: np.ones(len(bc_point)),
                      self.bndry['lat_mpi']: np.ones(len(bc_point)),
                      self.bndry['sep_0n']: np.zeros(len(bc_point))}}
            lv_solver = self.get_solver(method, 'lv')
            lv_circ2 = lv_solver.solve(bcs_lv)
            lv_circ2_1 = lv_circ2.vector.array[self.lv_corr]

            bcs_lv = {'function': {self.bndry['lat_pi']: np.ones(len(bc_point)),
                      self.bndry['lat_mpi']: np.ones(len(bc_point)),
                      self.bndry['sep_0p']: np.zeros(len(bc_point))}}
            lv_solver = self.get_solver(method, 'lv')
            lv_circ2 = lv_solver.solve(bcs_lv)
            lv_circ2_2 = lv_circ2.vector.array[self.lv_corr]

            lv_circ2 = (lv_circ2_1 + lv_circ2_2)/2
            lv_circ2 = uvc.correct_lv_circ2(lv_circ2)

        uvc.lv_mesh.point_data['lv_circ2'] = lv_circ2

        return lv_circ2

    def run_lv_circumferential3(self, lv_circ2, uvc, method):
        uvc.create_lv_circ_bc3(lv_circ2)
        bc_point = uvc.lv_mesh.point_data['bc3']
        sep_marker = np.where(bc_point == 1)[0]
        self.bndry['lv_septum'] = 27
        uvc.lv_bdata = self.point_to_face_marker(
            sep_marker, self.bndry['lv_septum'], uvc.lv_bdata)

        self.init_rv_mesh(uvc)
        self.init_lv_mesh(uvc)
        lv_solver = self.get_solver(method, 'lv')
        bcs_lv = {'face': {self.bndry['lv_septum']: 1,
                  self.bndry['rv_septum']: 1,
                  self.bndry['rv_endo']: 1}}

        if uvc.split_epi:
            bcs_lv['face'][self.bndry['lv_epi']] = -1
            bcs_lv['face'][self.bndry['rv_epi']] = -1
        else:
            bcs_lv['face'][self.bndry['epi']] = -1

        lv_circ3 = lv_solver.solve(bcs_lv)
        lv_circ3 = lv_circ3.vector.array[self.lv_corr]
        uvc.lv_mesh.point_data['lv_circ3'] = lv_circ3

        return lv_circ3

    def run_lv_circumferential4(self, lv_circ3, uvc, method, correct=True):
        if self.mmg:
            mesh, bc, mmg_bdata = uc.mmg_create_lv_circ_bc4(lv_circ3, uvc)
            sep_nodes = np.where(bc == 1)[0]
            mpi_nodes = np.where(bc == 2)[0]
            ant_nodes = np.where(bc == 4)[0]
            post_nodes = np.where(bc == 5)[0]
            zero_nodes = np.where(bc == 10)[0]

            bdata = self.point_to_face_marker(
                sep_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                mpi_nodes, 2, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                ant_nodes, 4, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                post_nodes, 5, bdata, new_faces=True, mesh=mesh)

            bcs_point = {}
            for i in range(len(zero_nodes)):
                bcs_point[tuple(mesh.points[zero_nodes[i]])] = 0

            bcs_lv = {'face': {1: 0, 2: 1, 4: 1/3, 5: 1/3},
                      'point': bcs_point}

            circ = run_coord(mesh, bdata, bcs_lv)
            lv_circ4 = np.abs(circ[:len(uvc.lv_mesh.points)])
            uvc.lv_mesh.point_data['circ_aux'] = lv_circ4
            lv_circ4 = uvc.correct_lv_circ4(lv_circ4)*np.pi

            # Save MMG mesh
            self.lv_mmg_mesh = mesh
            self.lv_mmg_bdata = mmg_bdata
            self.lv_mmg_bc = bc

        else:
            # Last problem
            uvc.create_lv_circ_bc4(lv_circ3)
            bc_point = uvc.lv_mesh.point_data['bc4_n']
            ant_marker = np.where(bc_point == 1)[0]
            pos_marker = np.where(bc_point == -1)[0]
            self.bndry['circ_ant'] = 28
            self.bndry['circ_post'] = 29
            uvc.lv_bdata = self.point_to_face_marker(ant_marker,  self.bndry['circ_ant'],
                                                     uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(pos_marker, self.bndry['circ_post'],
                                                     uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)

            self.init_lv_mesh(uvc)
            lv_solver = self.get_solver(method, 'lv')
            bcs_lv = {'face': {self.bndry['circ_ant']: 1,
                      self.bndry['circ_post']: -1}}
            lv_circ4 = lv_solver.solve(bcs_lv)
            lv_circ4_1 = lv_circ4.vector.array[self.lv_corr]

            bc_point = uvc.lv_mesh.point_data['bc4_p']
            ant_marker = np.where(bc_point == 1)[0]
            pos_marker = np.where(bc_point == -1)[0]
            self.bndry['circ_ant'] = 28
            self.bndry['circ_post'] = 29
            uvc.lv_bdata = self.point_to_face_marker(ant_marker, self.bndry['circ_ant'],
                                                     uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)
            uvc.lv_bdata = self.point_to_face_marker(pos_marker, self.bndry['circ_post'],
                                                     uvc.lv_bdata, new_faces=True, mesh=uvc.lv_mesh)

            self.init_lv_mesh(uvc)
            lv_solver = self.get_solver(method, 'lv')
            bcs_lv = {'face': {self.bndry['circ_ant']: 1,
                      self.bndry['circ_post']: -1}}
            lv_circ4 = lv_solver.solve(bcs_lv)
            lv_circ4_2 = lv_circ4.vector.array[self.lv_corr]

            lv_circ4 = (lv_circ4_1 + lv_circ4_2)/2
            uvc.create_lv_circ_bc5(lv_circ4)
            lv_circ4 = uvc.correct_lv_circ4(lv_circ4)*np.pi

        uvc.lv_mesh.point_data['lv_circ4'] = lv_circ4

        return lv_circ4

    def run_lv_circumferential5(self, lv_circ4, uvc, method):

        if self.mmg:

            circ_aux = np.zeros(len(uvc.lv_mesh.points))
            lv_circ5 = np.zeros(len(uvc.lv_mesh.points))
            sep_mesh, sep_bdata, sep_map, lat_mesh, lat_bdata, lat_map = uc.mmg_create_lv_circ_bc5(lv_circ4, uvc)

            if method == 'trajectory':
                self.get_local_vectors(uvc)
                eC = uvc.lv_mesh.cell_data['eC'][0]

                # sep_eC
                sep_eC = eC[sep_map[2]]
                sep_circ, corr = run_trajectory_coord(sep_mesh, sep_bdata, {'face': {1: 1, 2: 0}}, vector=sep_eC)
                sep_circ = sep_circ[corr]

                lat_eC = eC[lat_map[2]]
                lat_circ, corr = run_trajectory_coord(lat_mesh, lat_bdata, {'face': {1: 1, 2: 0}}, vector=lat_eC)
                lat_circ = lat_circ[corr]

            else:
                sep_circ = run_coord(sep_mesh, sep_bdata, {'face': {1: 1, 2: 0}})
                lat_circ = run_coord(lat_mesh, lat_bdata, {'face': {1: 1, 2: 0}})

            circ_aux[sep_map[0]] = sep_circ[sep_map[1]]
            circ_aux[lat_map[0]] = lat_circ[lat_map[1]]
            circ_aux = (circ_aux*2-1)
            uvc.lv_mesh.point_data['circ_aux'] = circ_aux

            sep_circ = (sep_circ*2-1)/3
            lat_circ_aux = lat_circ*2-1
            lat_circ = (2/3-np.abs(lat_circ_aux*2/3)) + 1/3
            lat_circ = lat_circ*np.sign(lat_circ_aux)
            lv_circ5[sep_map[0]] = sep_circ[sep_map[1]]
            lv_circ5[lat_map[0]] = lat_circ[lat_map[1]]

            lv_circ5 *= np.pi
            uvc.lv_mesh.point_data['lv_circ5'] = lv_circ5

        else:
            return lv_circ4

        return lv_circ5

    def run_rv_circumferential1(self, uvc, method):
        uvc.create_rv_circ_bc()

        # First, we need to define the point bc as face bc
        bc_point = uvc.rv_mesh.point_data['bc1']
        ant_marker = np.where(bc_point == 1)[0]
        pos_marker = np.where(bc_point == -1)[0]

        uvc.rv_bdata = self.point_to_face_marker(
            ant_marker, self.bndry['rv_lv_ant'], uvc.rv_bdata)
        uvc.rv_bdata = self.point_to_face_marker(
            pos_marker, self.bndry['rv_lv_post'], uvc.rv_bdata)

        self.init_rv_mesh(uvc)

        # Run RV circumferential problem
        rv_solver = self.get_solver(method, 'rv')
        bcs_rv = {'face': {self.bndry['rv_lv_ant']: 1,
                           self.bndry['rv_lv_post']: -1,
                           self.bndry['tv']: -1,
                           self.bndry['pv']: 1}}
        rv_circ1 = rv_solver.solve(bcs_rv)
        rv_circ1 = rv_circ1.vector.array[self.rv_corr]/3
        uvc.rv_mesh.point_data['rv_circ1'] = rv_circ1

        return rv_circ1

    def run_rv_circumferential2(self, rv_circ1, uvc, method):

        if self.mmg:
            mesh, bc, mmg_bdata = uc.mmg_create_rv_circ_bc2(rv_circ1, uvc)

            ant_nodes = np.where(bc == 1)[0]
            post_nodes = np.where(bc == 2)[0]
            zero_nodes = np.where(bc == 3)[0]
            bdata = self.point_to_face_marker(
                ant_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                post_nodes, 2, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                zero_nodes, 3, bdata, new_faces=True, mesh=mesh)

            # Need to figure out a better way to map...
            circ = run_coord(mesh, bdata, {'face': {1: -1, 2: 1, 3: 0}})
            rv_circ2 = circ[:len(uvc.rv_mesh.points)]
            uvc.rv_mesh.point_data['rv_circ2'] = rv_circ2

            self.rv_mmg_mesh = mesh
            self.rv_mmg_bdata = mmg_bdata
            self.rv_bc = bc


        else:
            raise('Not implemented')

        return rv_circ2

    def run_circumferential(self, uvc, method='laplace'):
        # LV side
        lv_circ1 = self.run_lv_circumferential1(uvc, 'laplace')

        # For the second problem we define surfaces at 0 and pi
        lv_circ2 = self.run_lv_circumferential2(lv_circ1, uvc, 'laplace')

        # # Third problem
        lv_circ3 = self.run_lv_circumferential3(lv_circ2, uvc, 'laplace')

        # # Fourth problem
        lv_circ = self.run_lv_circumferential4(lv_circ3, uvc, 'laplace')

        # # Fifth problem
        # lv_circ = self.run_lv_circumferential5(lv_circ4, uvc, method)
        uvc.lv_mesh.point_data['circ'] = lv_circ

        # # RV side
        rv_circ1 = self.run_rv_circumferential1(uvc, method)
        rv_circ = self.run_rv_circumferential2(rv_circ1, uvc, method)
        uvc.rv_mesh.point_data['circ'] = rv_circ

        return lv_circ, rv_circ



    def run_fast_circumferential(self, uvc, method='laplace'):
        self.init_lv_mesh(uvc)
        lv_solver = self.get_solver(method, 'lv')
        bcs_point = {}
        for i in range(len(uvc.lv_zero_nodes)):
            bcs_point[tuple(uvc.lv_mesh.points[uvc.lv_zero_nodes[i]])] = 0
        bcs_lv = {'face': {self.bndry['sep0']: 0,
                  self.bndry['lat0']: 1,
                  self.bndry['sep_ant']: 1/3,
                  self.bndry['sep_post']: 1/3},
                  'point': bcs_point,}

        lv_circ = lv_solver.solve(bcs_lv)
        lv_circ = lv_circ.vector.array[self.lv_corr]
        lv_circ = uvc.correct_lv_circ_by_subdomain(lv_circ)

        uvc.lv_mesh.point_data['circ'] = lv_circ*np.pi

        # TODO need to generate rv boundaries so it doesn't take the base elems.
        self.init_rv_mesh(uvc)
        rv_solver = self.get_solver(method, 'rv')
        bcs_rv = {'face': {self.bndry['rvlv_ant']: 1/3,
                  self.bndry['rvlv_post']: -1/3}}
        rv_circ = rv_solver.solve(bcs_rv)
        rv_circ = rv_circ.vector.array[self.rv_corr]
        uvc.rv_mesh.point_data['circ'] = rv_circ*np.pi

        return lv_circ, rv_circ

    def run_rv_lv_marker(self, uvc, method = 'laplace'):
        self.init_lv_mesh(uvc, og=True)
        lv_solver = self.get_solver(method, 'lv')

        bcs_lv = {'face': {self.bndry['lv_endo']: 0 }}
        if 'rv_lv_junction' in self.bndry:
            bcs_lv['face'][self.bndry['rv_lv_junction']] = 1
        else:
            bcs_lv['face'][self.bndry['rvlv_ant']] = 1
            bcs_lv['face'][self.bndry['rvlv_post']] = 1
        lv_rvlv = lv_solver.solve(bcs_lv)
        lv_rvlv = lv_rvlv.vector.array[self.lv_corr]
        uvc.lv_mesh.point_data['lv_rvlv'] = lv_rvlv

        return lv_rvlv


    def save_mmg_boundaries(self):
        mesh, bc = self.lv_mmg_mesh, self.lv_mmg_bc
        sep_nodes = np.where(bc == 1)[0]
        mpi_nodes = np.where(bc == 2)[0]
        ant_nodes = np.where(bc == 4)[0]
        post_nodes = np.where(bc == 5)[0]
        zero_nodes = np.where(bc == 10)[0]

        div_nodes = np.concatenate([sep_nodes, mpi_nodes, zero_nodes])

        bdata = self.point_to_face_marker(
            div_nodes, 1, new_faces=True, mesh=mesh)

        surf0 = io.Mesh(mesh.points, {'triangle': bdata[:,1:-1]})

        io.write('zero.stl', surf0)


        div_nodes = np.concatenate([ant_nodes, post_nodes, zero_nodes])
        bdata = self.point_to_face_marker(
            div_nodes, 1, new_faces=True, mesh=mesh)

        surf0 = io.Mesh(mesh.points, {'triangle': bdata[:,1:-1]})

        io.write('sep.stl', surf0)


    def run_ot_circumferential1(self, uvc, method='laplace'):
        self.init_ot_mesh(uvc)
        uvc.create_ot_circ_bc1()

        bcs_marker = {'function': {self.bndry['base']:
                                   uvc.ot_mesh.point_data['bc1'][self.ot_icorr]}}

        ot_solver = self.get_solver(method, 'ot')
        ot_circ = ot_solver.solve(bcs_marker)
        ot_circ = ot_circ.vector.array[self.ot_corr]
        uvc.ot_mesh.point_data['ot_circ1'] = ot_circ

        return ot_circ

    def run_ot_circumferential2(self, ot_circ1, uvc, method='laplace', correct=True):

        if self.mmg:
            mesh, bc = uc.mmg_create_ot_circ_bc2(ot_circ1, uvc)
            sep_nodes = np.where(bc == 1)[0]
            mpi_nodes = np.where(bc == 2)[0]
            pi_nodes = np.where(bc == 3)[0]
            tv_nodes = np.where(bc == 4)[0]
            base_nodes = np.where(bc == 5)[0]
            bdata = self.point_to_face_marker(
                sep_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                mpi_nodes, 2, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                pi_nodes, 3, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                tv_nodes, 4, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                base_nodes, self.bndry['base'], bdata, new_faces=True, mesh=mesh)

            # Need to figure out a better way to map...
            bc1 = np.zeros(len(mesh.points))
            bc1[:len(uvc.ot_mesh.points)] = uvc.ot_mesh.point_data['bc1']/np.pi
            mesh.point_data['bc1'] = bc1
            mesh_, mt = dxio.read_meshio_mesh(mesh, bdata)
            _, icorr = dxio.find_vtu_dx_mapping(mesh_)
            bc1 = bc1[icorr]

            circ = run_coord(mesh, bdata, {'face': {1: 0, 2: -1, 3: 1, 4: 0},
                                           'function': {self.bndry['base']:
                                                                      bc1}})
            ot_circ2i = circ[:len(uvc.ot_mesh.points)]
            uvc.ot_mesh.point_data['ot_circ2i'] = ot_circ2i

            mesh.point_data['ot_circ2i'] = circ

            mesh, bc = uc.mmg_create_ot_circ_bc3(ot_circ2i, uvc)
            sep_nodes = np.where(bc == 1)[0]
            mpi_nodes = np.where(bc == 2)[0]
            pi_nodes = np.where(bc == 3)[0]
            ant_nodes = np.where(bc == 4)[0]
            post_nodes = np.where(bc == 5)[0]
            base_nodes = np.where(bc == 6)[0]
            bdata = self.point_to_face_marker(
                sep_nodes, 1, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                mpi_nodes, 2, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                pi_nodes, 3, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                ant_nodes, 4, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                post_nodes, 5, bdata, new_faces=True, mesh=mesh)
            bdata = self.point_to_face_marker(
                base_nodes, self.bndry['base'], bdata, new_faces=True, mesh=mesh)

            # Need to figure out a better way to map...
            bc1 = np.zeros(len(mesh.points))
            bc1[:len(uvc.ot_mesh.points)] = uvc.ot_mesh.point_data['bc1']/np.pi
            mesh.point_data['bc1'] = bc1

            mesh_, mt = dxio.read_meshio_mesh(mesh, bdata)
            _, icorr = dxio.find_vtu_dx_mapping(mesh_)
            bc1 = bc1[icorr]
            circ = run_coord(mesh, bdata, {'face': {1: 0, 2: -1, 3: 1, 4: 1/3, 5: -1/3},
                                            'function': {self.bndry['base']: bc1}})
            ot_circ2 = circ[:len(uvc.lv_mesh.points)]*np.pi

        else:
            uvc.create_ot_circ_bc2(ot_circ1)

            bc_point = uvc.ot_mesh.point_data['bc2']
            lat_marker_neg = uvc.ot_bc_mpi
            lat_marker_pls = uvc.ot_bc_pi

            self.bndry['lat_pi'] = 23
            self.bndry['lat_mpi'] = 24
            uvc.ot_bdata = self.point_to_face_marker(lat_marker_pls, self.bndry['lat_pi'],
                                                     uvc.ot_bdata, new_faces=True, mesh=uvc.ot_mesh)
            uvc.ot_bdata = self.point_to_face_marker(lat_marker_neg, self.bndry['lat_mpi'],
                                                     uvc.ot_bdata, new_faces=True, mesh=uvc.ot_mesh)

            self.init_ot_mesh(uvc)

            bcs_ot = {'function': {self.bndry['lat_pi']: np.ones(len(bc_point))*np.pi,
                      self.bndry['lat_mpi']: np.ones(len(bc_point))*-np.pi,
                      self.bndry['base']: uvc.ot_mesh.point_data['bc1'][self.ot_icorr]}}

            ot_solver = self.get_solver(method, 'ot')
            ot_circ2 = ot_solver.solve(bcs_ot)
            ot_circ2 = ot_circ2.vector.array[self.ot_corr]
        uvc.ot_mesh.point_data['circ'] = ot_circ2

        return ot_circ2

    def run_ot_circumferential(self, uvc, method='laplace'):
        ot_circ1 = self.run_ot_circumferential1(uvc, method)
        ot_circ2 = self.run_ot_circumferential2(ot_circ1, uvc, method)

        return ot_circ2


    def get_local_vectors(self, uvc, linear=True):
        glong = self.get_func_gradient(uvc, 'long', 'lv', linear=linear)
        eL = glong/np.linalg.norm(glong, axis=1)[:,None]

        gtrans = self.get_func_gradient(uvc, 'trans', 'lv', linear=linear)
        eT = gtrans/np.linalg.norm(gtrans, axis=1)[:,None]

        eC = np.cross(eL, eT, axisa=1, axisb=1)
        eC = eC/np.linalg.norm(eC, axis=1)[:,None]

        if linear:
            uvc.lv_mesh.point_data['eL'] = eL
            uvc.lv_mesh.point_data['eC'] = eC
            uvc.lv_mesh.point_data['eT'] = eT
        else:
            uvc.lv_mesh.cell_data['eL'] = [eL]
            uvc.lv_mesh.cell_data['eC'] = [eC]
            uvc.lv_mesh.cell_data['eT'] = [eT]

