#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:32:08 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import dolfinxio as dxio
from uvcgen.LaplaceProblem import LaplaceProblem
import meshio as io
import cheartio as chio
from dolfinx.log import set_log_level, LogLevel

set_log_level(LogLevel.WARNING)


class FibGen:
    def __init__(self, mesh, bdata, boundaries, apex_id, params, use_mmg=False):
        self.mesh = mesh
        self.xyz = mesh.points
        self.bdata = bdata
        self.bndry = boundaries
        self.use_mmg = use_mmg

        self.init_bv_mesh(mesh, bdata)
        self.init_apex_bc(apex_id)

        self.params = {}
        for key in params.keys():
            self.params[key] = np.deg2rad(params[key])

    def init_bv_mesh(self, mesh, bdata):
        self.bv_mesh, self.bv_mt = dxio.read_meshio_mesh(mesh, bdata)

        self.LapSolver = LaplaceProblem(self.bv_mesh, self.bv_mt)
        self.bv_corr, self.bv_icorr = dxio.find_vtu_dx_mapping(self.bv_mesh)

    def init_apex_bc(self, apex_id):
        self.bc_apex= {tuple(self.xyz[apex_id]): 1.0}


    def run_laplace_problems(self):
        lap_fields = {}
        grad_fields = {}

        print('Running ventricular split')
        lap_fields['ven_trans'], grad_fields['ven_trans'] = self.run_ven_trans()
        print('Running longitudinal problems')
        lap_fields['lv_av_long'], grad_fields['lv_av_long'] = self.run_lv_av()
        lap_fields['lv_mv_long'], grad_fields['lv_mv_long'] = self.run_lv_mv()
        # lap_fields['lv_valves_long'], grad_fields['lv_valves_long']= self.run_lv_valves()
        lap_fields['rv_pv_long'], grad_fields['rv_pv_long'] = self.run_rv_pv()
        lap_fields['rv_tv_long'], grad_fields['rv_tv_long'] = self.run_rv_tv()
        print('Running weight problems')
        lap_fields['lv_weight'], grad_fields['lv_weight'] = self.run_lv_weight()
        lap_fields['rv_weight'], grad_fields['rv_weight'] = self.run_rv_weight()
        lap_fields['rv_op_weight'], grad_fields['rv_op_weight'] = self.run_rv_op_weight()
        print('Running transmural problems')
        lap_fields['lv_trans'], grad_fields['lv_trans'] = self.run_lv_trans()
        lap_fields['rv_trans'], grad_fields['rv_trans'] = self.run_rv_trans()
        lap_fields['epi_trans'], grad_fields['epi_trans'] = self.run_epi_trans()
        lap_fields['septum'], grad_fields['septum'] = self.run_septum()
        print('Done!')

        self.lap = lap_fields
        self.grad = grad_fields

        return lap_fields, grad_fields


    def run_laplace(self, bcs_marker, return_gradient=True):
        lap = self.LapSolver.solve(bcs_marker)
        if return_gradient:
            glap = self.LapSolver.get_linear_gradient(lap)

        lap = lap.x.petsc_vec.array[self.bv_corr]
        if return_gradient:
            glap = glap.x.petsc_vec.array.reshape([-1,3])[self.bv_corr]
            glap = glap/np.linalg.norm(glap, axis=1)[:,None]
            return lap, glap
        else:
            return lap

    def run_lv_av(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['av']: 0.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_lv_mv(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['mv']: 0.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_lv_valves(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['av']: 0.0,
                               self.bndry['mv']: 0.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_rv_pv(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['pv']: 0.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_rv_tv(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['tv']: 0.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_rv_valves(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['pv']: 0.0,
                               self.bndry['tv']: 0.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_lv_weight(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['av']: 0.0,
                               self.bndry['mv']: 1.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_rv_weight(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['pv']: 0.0,
                               self.bndry['tv']: 1.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_rv_op_weight(self, gradient=True):
        bcs_marker = {'point': self.bc_apex,
                      'face': {self.bndry['pv']: 1.0,
                               self.bndry['tv']: 0.0,
                               self.bndry['mv']: 1.0,
                               self.bndry['av']: 1.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_epi_trans(self, gradient=True):
        bcs_marker = {'face': {self.bndry['lv_epi']: 1.0,
                               self.bndry['rv_epi']: 1.0,
                               self.bndry['lv_endo']: 0.0,
                               self.bndry['rv_endo']: 0.0,
                               self.bndry['rv_septum']: 0.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_lv_trans(self, gradient=True):
        bcs_marker = {'face': {self.bndry['lv_epi']: 0.0,
                               self.bndry['rv_epi']: 0.0,
                               self.bndry['rv_endo']: 0.0,
                               self.bndry['rv_septum']: 0.0,
                               self.bndry['lv_endo']: 1.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_rv_trans(self, gradient=True):
        bcs_marker = {'face': {self.bndry['lv_epi']: 0.0,
                               self.bndry['rv_epi']: 0.0,
                               self.bndry['lv_endo']: 0.0,
                               self.bndry['rv_endo']: 1.0,
                               self.bndry['rv_septum']: 1.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_septum(self, gradient=True):
        bcs_marker = {'face': {self.bndry['rv_septum']: 0.5,
                               self.bndry['lv_endo']: 0.0,
                               self.bndry['rv_endo']: 1.0,
                               self.bndry['rv_septum']: 1.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def run_ven_trans(self, gradient=True):
        bcs_marker = {'face': {self.bndry['lv_endo']: 1.0,
                               self.bndry['rv_endo']: 0.0,
                               self.bndry['rv_septum']: 0.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap
    
    def run_lv_rv(self, gradient=True):
        bcs_marker = {'function': {self.bndry['lv_epi']: self.lap['ven_trans'][self.bv_icorr],
                                   self.bndry['rv_epi']: self.lap['ven_trans'][self.bv_icorr]},
                        'face': {self.bndry['rv_septum']: 1.0,
                                 self.bndry['lv_endo']: 1.0}}

        lap = self.run_laplace(bcs_marker, return_gradient=gradient)
        return lap

    def compute_basis_vectors(self):
        lap = self.lap
        grad = self.grad

        # LV
        # longitudinal
        lv_glong = grad['lv_mv_long']*lap['lv_weight'][:,None] + grad['lv_av_long']*(1 - lap['lv_weight'][:,None])
        eL_lv = lv_glong/np.linalg.norm(lv_glong, axis=1)[:,None]

        # transmural
        lv_gtrans = grad['lv_trans'] - (eL_lv*grad['lv_trans'])*eL_lv
        eT_lv = lv_gtrans/np.linalg.norm(lv_gtrans, axis=1)[:,None]

        # circumferential
        eC_lv = np.cross(eL_lv, eT_lv, axisa=1, axisb=1)
        eC_lv = eC_lv/np.linalg.norm(eC_lv, axis=1)[:,None]

        # Ensuring orthogonality
        eT_lv = np.cross(eC_lv, eL_lv, axisa=1, axisb=1)
        eT_lv = eT_lv/np.linalg.norm(eT_lv, axis=1)[:,None]

        # RV
        # longitudinal
        rv_glong = grad['rv_tv_long']*lap['rv_weight'][:,None]  + grad['rv_pv_long']*(1 - lap['rv_weight'][:,None] )
        eL_rv = rv_glong/np.linalg.norm(rv_glong, axis=1)[:,None]

        # transmural
        rv_gtrans = grad['rv_trans'] - (eL_rv*grad['rv_trans'])*eL_rv
        eT_rv = rv_gtrans/np.linalg.norm(rv_gtrans, axis=1)[:,None]

        # circumferential
        eC_rv = np.cross(eL_rv, eT_rv, axisa=1, axisb=1)
        eC_rv = eC_rv/np.linalg.norm(eC_rv, axis=1)[:,None]

        # Ensuring orthogonality
        eT_rv = np.cross(eC_rv, eL_rv, axisa=1, axisb=1)
        eT_rv = eT_rv/np.linalg.norm(eT_rv, axis=1)[:,None]

        # Write out global circumferential vector
        eC = eC_rv*(1-lap['ven_trans'][:,None]) + eC_lv*lap['ven_trans'][:,None]
        eC = eC/np.linalg.norm(eC, axis=1)[:,None]

        self.basis = {'eC_lv': eC_lv,
                        'eT_lv': eT_lv,
                        'eL_lv': eL_lv,
                        'eC_rv': eC_rv,
                        'eT_rv': eT_rv,
                        'eL_rv': eL_rv,
                        'eC': eC}

        return self.basis

    @staticmethod
    def redistribute_weight(weight, up, low, strategy='centre'):
        new_weight = weight.copy()

        if strategy == 'flip':
            # Shift all weights
            new_mean = 1 - np.mean(weight)
            shift = new_mean - np.mean(weight)
            new_weight = new_weight + shift

            # Cut off values outside of range 0 - 1
            new_weight[new_weight > 1] = 1
            new_weight[new_weight < 0] = 0

            # Redistribute new tail
            new_weight = (new_weight - np.min(new_weight)) / (np.max(new_weight) - np.min(new_weight))
            tmp = new_weight.copy()

            if shift > 0:
                tmp[tmp >= new_mean] = np.nan
                tmp = (tmp - np.nanmin(tmp)) / (new_mean - np.nanmin(tmp))
            elif shift < 0:
                tmp[tmp <= new_mean] = np.nan
                tmp = (tmp - new_mean) / (np.nanmax(tmp) - new_mean)

            tmp[np.isnan(tmp)] = new_weight[np.isnan(tmp)]
            new_weight = tmp

        else:  # cut off tails so that weights are centered
            # Find upper and lower limits
            upper_lim = np.quantile(weight, up)
            lower_lim = np.quantile(weight, low)

            # Set upper and lower values to limits
            new_weight[new_weight > upper_lim] = upper_lim
            new_weight[new_weight < lower_lim] = lower_lim

            # Redistribute/normalize values
            new_weight = (new_weight - np.min(new_weight)) / (np.max(new_weight) - np.min(new_weight))

        return new_weight


    def compute_alpha_beta_angles(self):
        lap = self.lap

        # LV
        alpha_lv_endo_long = self.params['AENDOLV'] * lap['lv_weight'] + self.params['AOTENDOLV'] * (1 - lap['lv_weight'])  # Endo
        new_long_weight = self.redistribute_weight(lap['lv_mv_long'], 0.7, 0.01)
        alpha_lv_epi_long = self.params['AEPILV'] * new_long_weight + self.params['AOTEPILV'] * (1 - new_long_weight)

        alpha_wall_lv = alpha_lv_endo_long * (1 - lap['epi_trans']) + alpha_lv_epi_long * lap['epi_trans']
        beta_wall_lv = (self.params['BENDOLV'] * (1 - lap['epi_trans']) + self.params['BEPILV'] * lap['epi_trans']) * lap['lv_weight']

        # RV
        new_long_weight_rv = self.redistribute_weight(lap['rv_op_weight'], 0.2, 0.001)
        alpha_rv_endo_long = self.params['AENDORV'] * new_long_weight_rv + self.params['ATRIENDO'] * (1 - new_long_weight_rv)
        alpha_rv_epi_long = self.params['AEPIRV'] * lap['rv_weight'] + self.params['AOTEPIRV'] * (1 - lap['lv_weight'])

        alpha_wall_rv = alpha_rv_endo_long * (1 - lap['epi_trans']) + alpha_rv_epi_long * lap['epi_trans']
        beta_wall_rv = (self.params['BENDORV'] * (1 - lap['epi_trans']) + self.params['BEPIRV'] * lap['epi_trans']) * lap['rv_weight']

        # Septum
        sep = np.abs(lap['ven_trans'] - 0.5)
        sep = (sep - np.min(sep)) / (np.max(sep) - np.min(sep))
        alpha_septum = (alpha_lv_endo_long * sep * lap['lv_trans']) + (alpha_rv_endo_long * sep * lap['rv_trans'])
        beta_septum = (self.params['BENDOLV'] * lap['lv_trans'] * lap['lv_weight']) + (self.params['BENDORV'] * lap['rv_trans'] * lap['lv_weight'])

        self.angles = {'alpha_lv_endo_long': alpha_lv_endo_long,
                'alpha_lv_epi_long': alpha_lv_epi_long,
                'alpha_wall_lv': alpha_wall_lv,
                'beta_wall_lv': beta_wall_lv,
                'alpha_rv_endo_long': alpha_rv_endo_long,
                'alpha_rv_epi_long': alpha_rv_epi_long,
                'alpha_wall_rv': alpha_wall_rv,
                'beta_wall_rv': beta_wall_rv,
                'alpha_septum': alpha_septum,
                'beta_septum': beta_septum
                }

        return self.angles


    @staticmethod
    def rotate_basis(eC, eL, eT, alpha, beta):
        eC = eC/np.linalg.norm(eC, axis=1)[:,None]
        eT = eT/np.linalg.norm(eT, axis=1)[:,None]
        eL = eL/np.linalg.norm(eL, axis=1)[:,None]

        # Matrix of directional vectors
        Q = np.stack([eC, eL, eT], axis=-1)
        Q = np.transpose(Q, (2, 1, 0))

        # Create rotation matrix - from Doste code
        axis = eT
        R = np.array([[np.cos(alpha) + (axis[:, 0]**2)*(1 - np.cos(alpha)), axis[:,0] * axis[:,1]*(1 - np.cos(alpha)) - axis[:,2]*np.sin(alpha), axis[:,0]*axis[:,2]*(1 - np.cos(alpha)) + axis[:,1]*np.sin(alpha)],
                             [axis[:,1]*axis[:,0]*(1 - np.cos(alpha)) + axis[:,2]*np.sin(alpha), np.cos(alpha) + (axis[:,1]**2)*(1 - np.cos(alpha)), axis[:,1]*axis[:, 2]*(1 - np.cos(alpha)) - axis[:, 0]*np.sin(alpha)],
                             [axis[:,2]*axis[:,0]*(1 - np.cos(alpha)) - axis[:,1]*np.sin(alpha), axis[:,2]*axis[:,1]*(1 - np.cos(alpha)) + axis[:, 0]*np.sin(alpha), np.cos(alpha)+(axis[:, 2]**2)*(1 - np.cos(alpha))]])

        # Rotate the circumferential direction around the transmural direction
        QX = np.zeros_like(R)
        for i in range(len(eC)):
            QX[:, :, i] = np.matmul(Q[:, :, i], R[:, :, i])

        # Second rotation (beta) about QX
        axis2 = QX[1, :, :].T
        R2 = np.array([
            [np.cos(beta) + (axis2[:,0]**2)*(1 - np.cos(beta)), axis2[:,0]*axis2[:, 1]*(1 - np.cos(beta)) - axis2[:,2] * np.sin(beta), axis2[:,0] * axis2[:,2] * (1 - np.cos(beta)) + axis2[:,1] * np.sin(beta)],
            [axis2[:,1] * axis2[:,0]*(1 - np.cos(beta)) + axis2[:,2]*np.sin(beta), np.cos(beta) + (axis2[:,1]**2)*(1 - np.cos(beta)), axis2[:,1] * axis2[:,2] * (1 - np.cos(beta)) - axis2[:,0] * np.sin(beta)],
            [axis2[:,2] * axis2[:,0]*(1 - np.cos(beta)) - axis2[:,1]*np.sin(beta), axis2[:, 2] * axis2[:,1] * (1 - np.cos(beta)) + axis2[:,0] * np.sin(beta), np.cos(beta) + (axis2[:,2]**2) * (1 - np.cos(beta))]
        ])

        QX2 = np.zeros_like(R)
        for i in range(len(eC)):
            QX2[:, :, i] = np.matmul(QX[:, :, i], R2[:, :, i])

        return QX2

    def compute_local_basis(self):
        basis = self.basis
        angles = self.angles
        Qlv_septum = self.rotate_basis(basis['eC_lv'], basis['eL_lv'], basis['eT_lv'], angles['alpha_septum'], angles['beta_septum'])
        Qrv_septum = self.rotate_basis(basis['eC_rv'], basis['eL_rv'], basis['eT_rv'], angles['alpha_septum'], angles['beta_septum'])
        Qlv_epi = self.rotate_basis(basis['eC_lv'], basis['eL_lv'], basis['eT_lv'], angles['alpha_wall_lv'], angles['beta_wall_lv'])
        Qrv_epi = self.rotate_basis(basis['eC_rv'], basis['eL_rv'], basis['eT_rv'], angles['alpha_wall_rv'], angles['beta_wall_rv'])

        self.local_basis = {'Qlv_septum': Qlv_septum,
                       'Qrv_septum': Qrv_septum,
                       'Qlv_epi': Qlv_epi,
                       'Qrv_epi': Qrv_epi,
                       }

        return self.local_basis


    @staticmethod
    def bislerp(Q1, Q2, interp_func):
        from scipy.spatial.transform import Rotation, Slerp
        # Initialize variable
        Q = np.zeros_like(Q1)

        # Round values of interpolation function to ensure that none are above 1.0
        # or less than 0.0
        interp_func[interp_func > 1.0] = 1.0
        interp_func[interp_func < 0.0] = 0.0

        for i in range(Q1.shape[2]):
            # Convert to quaternions
            r1 = Rotation.from_matrix(Q1[:,:,i])
            qA = r1.as_quat()

            r2 = Rotation.from_matrix(Q2[:, :, i])
            qB = r2.as_quat()

            rot = Rotation.from_quat(np.array([qA, qB]))

            # Spherical interpolation and convert back to rotation matrix
            slerp = Slerp(np.array([0, 1]), rot)
            interpolated_quaternion = slerp(interp_func[i])
            Q[:,:,i] = interpolated_quaternion.as_matrix()

        return Q


    def generate_discontinuous_mesh(self, ven_trans):
        xyz = self.mesh.points
        ien = self.mesh.cells[0].data

        ven_trans_elem = np.mean(ven_trans[ien], axis=1)
        pos_elems = np.where(ven_trans_elem >= 0.5)[0]
        neg_elems = np.where(ven_trans_elem < 0.5)[0]
        pos_nodes = np.unique(ien[pos_elems])
        neg_nodes = np.unique(ien[neg_elems])
        inter_nodes = np.intersect1d(pos_nodes, neg_nodes)

        inter_elems = np.where(np.any(np.isin(ien, inter_nodes), axis=1))[0]
        pos_elems = np.intersect1d(inter_elems, pos_elems)
        neg_elems = np.intersect1d(inter_elems, neg_elems)

        new_xyz = np.vstack([xyz, xyz[inter_nodes]])
        map_new_nodes = np.arange(len(new_xyz))
        map_new_nodes[inter_nodes] = np.arange(len(inter_nodes)) + len(xyz)
        new_ien = np.copy(ien)
        new_ien[neg_elems] = map_new_nodes[new_ien[neg_elems]]

        new_mesh = io.Mesh(new_xyz, {'tetra': new_ien})

        map_to_mesh = np.arange(len(new_xyz))
        map_to_mesh[len(xyz):] = inter_nodes
        new_ven_trans = -np.ones(len(new_xyz))
        new_ven_trans[pos_nodes] = 1
        # new_mesh.point_data['ven_trans'] = new_ven_trans

        # io.write('check.vtu', new_mesh)

        return new_mesh, map_to_mesh, new_ven_trans
    

    def interpolate_local_basis(self):
        disc_mesh, disc_map, div_func = self.generate_discontinuous_mesh(self.lap['ven_trans'])
        self.fib_mesh = disc_mesh
        self.disc_map = disc_map

        epi_trans = self.lap['epi_trans'][disc_map]

        Qrv_septum = self.local_basis['Qrv_septum'][:,:,disc_map]
        Qlv_septum = self.local_basis['Qlv_septum'][:,:,disc_map]
        Qrv_epi = self.local_basis['Qrv_epi'][:,:,disc_map]
        Qlv_epi = self.local_basis['Qlv_epi'][:,:,disc_map]

        Qs = Qrv_septum
        Qs[:,:,div_func > 0.5] = Qlv_septum[:,:,div_func > 0.5]

        Qepi = self.bislerp(Qrv_epi, Qlv_epi, div_func)
        Q = self.bislerp(Qs, Qepi, epi_trans)

        return Q

    def get_linear_fibers(self, method='bislerp'):
        # Map to continuous mesh
        map_disc = chio.map_between_meshes_disc(self.mesh, self.fib_mesh)
        if method == 'bislerp':
            _, disc_map, _ = self.generate_discontinuous_mesh(self.lap['ven_trans'])

            epi_trans = self.lap['epi_trans'][disc_map]
            ven_trans = self.lap['ven_trans'][disc_map]

            Qrv_septum = self.local_basis['Qrv_septum'][:,:,disc_map]
            Qlv_septum = self.local_basis['Qlv_septum'][:,:,disc_map]
            Qrv_epi = self.local_basis['Qrv_epi'][:,:,disc_map]
            Qlv_epi = self.local_basis['Qlv_epi'][:,:,disc_map]

            Qs = Qrv_septum
            Qs[:,:,ven_trans > 0.5] = Qlv_septum[:,:,ven_trans > 0.5]

            Qepi = self.bislerp(Qrv_epi, Qlv_epi, ven_trans)
            Q = self.bislerp(Qs, Qepi, epi_trans)
            f, n, s = Q

            self.f_linear = f.T[map_disc]
            self.s_linear = s.T[map_disc]
            self.n_linear = n.T[map_disc]

        elif method == 'interpolate':
            self.f_linear = self.f[map_disc]
            self.s_linear = self.s[map_disc]
            self.n_linear = self.n[map_disc]

        return self.f_linear, self.s_linear, self.n_linear

    def get_fibers(self):
        print('Computing basis vectors')
        self.compute_basis_vectors()

        print('Computing angles')
        self.compute_alpha_beta_angles()

        print('Computing local basis')
        self.compute_local_basis()

        print('Interpolating basis')
        Q = self.interpolate_local_basis()

        print('Done!')
        f, n, s = Q
        self.f = f.T
        self.n = n.T
        self.s = s.T

        return f.T, s.T, n.T



    def get_fiber_angle(self):
        alpha_wall_lv = self.angles['alpha_wall_lv'][self.disc_map]
        alpha_wall_rv = self.angles['alpha_wall_rv'][self.disc_map]
        alpha_septum = self.angles['alpha_septum'] [self.disc_map]
        ven_trans = self.lap['ven_trans'][self.disc_map]
        epi_trans = self.lap['epi_trans'][self.disc_map]
        alpha_wall = alpha_wall_lv * ven_trans + alpha_wall_rv * (1 - ven_trans)
        alpha = alpha_wall * epi_trans + alpha_septum * (1 - epi_trans)
        fib_angle = np.rad2deg(alpha)
        return fib_angle


    def get_point_data(self):
        data = {}
       
        data.update(self.lap)
        keys = list(data.keys())
        for key in keys:
            data['g_' + key] = self.grad[key]
        data.update(self.basis)
        data.update(self.angles)

        return data


    def get_fiber_data(self):
        data = {}
    
        data['fib_angle'] = self.get_fiber_angle()
        data['f'] = self.f
        data['n'] = self.n
        data['s'] = self.s

        return data


    def write_intermediate_dfile(self, out_fldr, which='all'):
        if which == 'lap' or which == 'all':
            for key in self.lap.keys():
                chio.write_dfile(out_fldr + key + '.FE', self.lap[key])
        elif which == 'grad' or which == 'all':
            for key in self.grad.keys():
                chio.write_dfile(out_fldr + 'g_' + key + '.FE', self.grad[key])
        elif which == 'basis' or which == 'all':
            for key in self.basis.keys():
                chio.write_dfile(out_fldr + key + '.FE', self.basis[key])
        elif which == 'angles' or which == 'all':
            for key in self.angles.keys():
                chio.write_dfile(out_fldr + key + '.FE', self.basis[key])