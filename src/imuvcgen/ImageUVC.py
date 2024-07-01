#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 18:32:43 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
from niftiutils import readFromNIFTI
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator
import imuvcgen.laplace_functions as lf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from imuvcgen.utils import fit_elipse_2d
import plot_functions as pf
import meshio as io

inlet = 1
outlet = 2
walls = 3
exterior = -1
interior = 0

def define_laplace_labels(mask, maskin, maskout):
    # define wall
    maskwall = mask - binary_erosion(mask)

    # Delete everything non inlet or outlet
    maskwall = maskwall * (mask == 1) * (1 - np.minimum(1,maskin + maskout))

    labels = interior * (mask == 1) * (1 - maskin) * (1 - maskout) * (1 - maskwall) + exterior * (mask == 0)
    labels = labels + inlet * (maskin == 1) * (1 - maskwall)
    labels = labels + outlet * (maskout == 1) * (1 - maskwall)
    labels = labels + walls * maskwall

    return labels

def mask_to_points(mask):
    return np.vstack(np.where(mask==1)).T


def find_division_mask(var, value, mask):
    # Find division points
    pos = ((var > value)*mask).astype(int)
    neg = ((var <= value)*mask).astype(int)

    pos_dl = binary_dilation(pos)
    neg_dl = binary_dilation(neg)

    div = (pos_dl==1)*(neg_dl==1)*mask

    # split lat mask
    pos_div = (div*pos).astype(int)
    neg_div = (div*neg).astype(int)

    return div, pos_div, neg_div

class UVC:
    def __init__(self, sa_cmr, translations=None):
        # Get affine (affines are always grabbed from images)
        self.affine = sa_cmr.affine
        seg_data = sa_cmr.lge_data
        sa_labels = sa_cmr.lge_labels

        # SA segmentation stuff
        self.img_long_axis_vector = np.array([0,0,1])   # Assume it is always aligned with z
        self.pixdim = sa_cmr.zooms
        lv_mask = (seg_data==sa_labels['lv_wall']) + (seg_data==sa_labels['lv_fib'])
        rv_mask = (seg_data==sa_labels['rv_bp'])
        self.lv_mask = lv_mask.astype(float)
        self.rv_mask = rv_mask.astype(float)
        self.lv_fib_img = (seg_data==sa_labels['lv_fib']).astype(float)
        self.lv_fib = self.lv_fib_img[lv_mask==1]


        # Get lv coords
        ijk = np.vstack(np.where(self.lv_mask)).T
        ijk = self.add_translations(ijk, translations)
        xyz = nib.affines.apply_affine(self.affine, ijk)
        self.lv_xyz = xyz

        # Get rv coords
        rv_ijk = np.vstack(np.where(self.rv_mask )).T
        rv_ijk = self.add_translations(rv_ijk, translations)
        self.rv_xyz = nib.affines.apply_affine(self.affine, rv_ijk)


        # Define coordinates
        x = np.arange(self.lv_mask.shape[0])
        y = np.arange(self.lv_mask.shape[1])
        X, Y = np.meshgrid(x,y)
        coords = np.zeros([self.lv_mask.shape[0], self.lv_mask.shape[1], 2])
        coords[:,:,0] = X.T
        coords[:,:,1] = Y.T
        self.coords = coords.reshape([-1,2])

        # LA landmark stuff
        self.translations = translations
        # self.process_la_images(la_cmrs, la_labels)


    def add_translations(self, ijk, trans):
        if len(trans) > 0:
            ijk = ijk.astype(float)
            for i in range(len(trans)):
                ijk[ijk[:,2]==i,0:2] += trans[i]
        return ijk



    def process_la_images(self, la_landmarks, la_labels):
        # Finding points from landmarks
        pts = {}
        for key in la_labels.keys():
            pts[key] = []


        for view in la_landmarks.keys():
            seg_lnd_data, affine, _ = readFromNIFTI(la_landmarks[view], 0)

            for key in la_labels:
                ind = np.where(np.isclose(seg_lnd_data,la_labels[key]))
                ind = np.vstack(ind).T.astype(float)

                ind = self.add_translations(ind, self.translations[view])

                pts[key].append(nib.affines.apply_affine(affine, ind))

        for key in la_labels.keys():
            pts[key] = np.vstack(pts[key])

        self.la_points = pts

        # Getting centroids
        centroids = {}
        for key in la_labels:
            if len(self.la_points[key]) > 3:
                centroids[key] = self.get_valve_centroid(self.la_points[key])
            else:
                centroids[key] = np.mean(self.la_points[key], axis=0)
        self.la_landmarks = centroids
        self.mv_centroid = centroids['mv']
        self.apex_lv_epi = centroids['epi_apex']
        self.apex_lv_endo = centroids['endo_apex']

        # Find long axis vector
        long_axis_vector = centroids['mv'] - centroids['epi_apex']
        self.long_length = np.linalg.norm(long_axis_vector)
        self.la_long_axis_vector = long_axis_vector/self.long_length



    @staticmethod
    def get_valve_centroid(points):
        # Fit plane to points
        centroid = np.mean(points, axis=0)
        svd = np.linalg.svd(points - centroid)
        normal = svd[2][-1]
        normal = normal/np.linalg.norm(normal)

        # Project points to plane
        vector = points - centroid
        vector_plane = vector - (vector@normal)[:,None]*normal[None]

        # X axis will be aligned with first vector
        x_vector = vector_plane[0]
        x_vector = x_vector/np.linalg.norm(x_vector)
        y_vector = np.cross(normal, x_vector)
        x_coord = vector_plane@x_vector
        y_coord = vector_plane@y_vector
        xy = np.vstack([x_coord, y_coord]).T
        center, axis_l,rotation = fit_elipse_2d(xy[:,:2])

        valve_centroid = centroid + x_vector*center[0] + y_vector*center[1]

        return valve_centroid


    def compute_slice_centroids(self):
        self.lv_centroids = np.zeros([self.lv_mask.shape[-1],2])
        for z in range(self.rv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue
            lv_coords = mask_to_points(self.lv_mask[:,:,z])

            self.lv_centroids[z] = np.mean(lv_coords, axis=0)


    def get_trans_bndry(self):
        lv_bndry_endo = np.zeros_like(self.lv_mask)
        lv_bndry_epi = np.zeros_like(self.lv_mask)
        for z in range(self.lv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue
            # Get LV boundary
            lv_bndry = self.lv_mask[:,:,z] - binary_erosion(self.lv_mask[:,:,z])

            # Get BP mask
            bp_mask = binary_fill_holes(self.lv_mask[:,:,z]) - self.lv_mask[:,:,z]
            bp_dilated = binary_dilation(bp_mask)

            # Split endo and epi bndy
            lv_bndry_endo[:,:,z] = (bp_dilated==1)*(lv_bndry==1)
            lv_bndry_endo[:,:,z] = (bp_dilated==1)*(lv_bndry==1)
            lv_bndry_epi[:,:,z] = lv_bndry - lv_bndry_endo[:,:,z]

        self.lv_bndry_endo = lv_bndry_endo
        self.lv_bndry_epi = lv_bndry_epi


    def solve_trans_laplace(self):
        trans = np.zeros_like(self.lv_mask)
        for z in range(self.lv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue

            labels = define_laplace_labels(self.lv_mask[:,:,z],
                                           self.lv_bndry_endo[:,:,z],
                                           self.lv_bndry_epi[:,:,z])
            n, ind = lf.uvc_get_index_mapping(labels)
            A = lf.uvc_get_ablock(self.lv_mask[:,:,z], ind, labels, n, self.pixdim)

            # Dirichlet conditions
            RHS = np.zeros(A.shape[0])
            RHS[ind[labels==outlet]] = 1

            trans[:,:,z] = lf.solve_laplace(A, RHS, self.lv_mask[:,:,z])

        self.trans = trans


    def get_circ0_bndry(self):
        self.mid_sep = np.zeros([self.lv_mask.shape[-1],2])
        self.sep_vector = np.zeros([self.lv_mask.shape[-1],2])

        rv_insert_ant = np.zeros_like(self.lv_mask)
        rv_insert_post = np.zeros_like(self.lv_mask)
        self.rv_insert_post = np.zeros([self.lv_mask.shape[-1],2], dtype=int)
        self.rv_insert_ant = np.zeros([self.lv_mask.shape[-1],2], dtype=int)
        for z in range(self.rv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue
            rv_bndry = binary_dilation(self.rv_mask[:,:,z]) - self.rv_mask[:,:,z]
            lv_bndry = self.lv_mask[:,:,z] - binary_erosion(self.lv_mask[:,:,z])
            sep_bndry = rv_bndry*lv_bndry
            rv_bndry = rv_bndry - sep_bndry

            # Find insertion points
            rv_inserts = np.zeros([2,2])
            lv_pts = mask_to_points(lv_bndry==1)
            rv_pts = mask_to_points(rv_bndry==1)

            tree = KDTree(rv_pts)
            dist, _ = tree.query(lv_pts)
            rv_inserts[0] = lv_pts[np.argmin(dist)]

            dist_insert1 = np.linalg.norm(lv_pts - rv_inserts[0], axis=1)
            lv_pts_aux = lv_pts[dist_insert1 > np.quantile(dist_insert1, 0.2)]

            dist, _ = tree.query(lv_pts_aux)
            rv_inserts[1] = lv_pts_aux[np.argmin(dist)]

            # TODO need to figure out which one is anterior/posterior
            # find septum vector
            self.mid_sep[z] = (rv_inserts[0] + rv_inserts[1])/2
            sep_vector = self.mid_sep[z] - self.lv_centroids[z]
            self.sep_vector[z] = sep_vector/np.linalg.norm(sep_vector)

            ant_post_vector = np.cross(self.sep_vector[z], self.img_long_axis_vector)
            dist = ant_post_vector[0:2]@(rv_inserts-self.lv_centroids[z]).T
            order = np.argsort(dist)
            # First in inferior
            self.rv_insert_post[z] = rv_inserts[order[0]]
            self.rv_insert_ant[z] = rv_inserts[order[1]]


            rv_insert_ant[self.rv_insert_ant[z,0],self.rv_insert_ant[z,1],z] = 1
            rv_insert_post[self.rv_insert_post[z,0],self.rv_insert_post[z,1],z] = 1

        self.rv_insert_ant_mask = rv_insert_ant
        self.rv_insert_post_mask = rv_insert_post

        # Save position of rv insterts in xyz
        slices = np.arange(len(self.rv_insert_ant))[:,None]
        insert_ijk = np.vstack([np.hstack([self.rv_insert_ant, slices]),
                               np.hstack([self.rv_insert_post, slices])])
        insert_ijk = insert_ijk[~np.all(insert_ijk[:,0:2]==0, axis=1)]
        insert_ijk = self.add_translations(insert_ijk, self.translations)
        self.insert_xyz = nib.affines.apply_affine(self.affine, insert_ijk)


    def solve_circ0_laplace(self):
        circ0 = np.zeros_like(self.lv_mask)
        for z in range(self.lv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue
            labels = define_laplace_labels(self.lv_mask[:,:,z],
                                           self.rv_insert_ant_mask[:,:,z],
                                           self.rv_insert_post_mask[:,:,z])
            n, ind = lf.uvc_get_index_mapping(labels)
            A = lf.uvc_get_ablock(self.lv_mask[:,:,z], ind, labels, n, self.pixdim)

            # Dirichlet conditions
            RHS = np.zeros(A.shape[0])
            RHS[ind[labels==inlet]] = 1
            RHS[ind[labels==outlet]] = -1
            circ0[:,:,z] = lf.solve_laplace(A, RHS, self.lv_mask[:,:,z]==1)

        self.circ0 = circ0


    def get_circ1_bndry(self):
        self.pi_mask = np.zeros_like(self.lv_mask)
        self.mpi_mask = np.zeros_like(self.lv_mask)
        self.sep_mask = np.zeros_like(self.lv_mask)
        self.lat_mask = np.zeros_like(self.lv_mask)
        for z in range(self.lv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue

            # Find division points
            div, pos_div, neg_div = find_division_mask(self.circ0[:,:,z], 0,
                                                       self.lv_mask[:,:,z])

            # split mask in lat and sep
            dist = self.sep_vector[z]@(self.coords - self.lv_centroids[z]).T
            dist = dist.reshape(div.shape)
            lat_mask = div*(dist<0)
            sep_mask = div*(dist>=0)

            # split lat mask
            self.pi_mask[:,:,z] = (lat_mask*pos_div).astype(int)
            self.mpi_mask[:,:,z] = (lat_mask*neg_div).astype(int)
            self.sep_mask[:,:,z] = sep_mask.astype(int)
            self.lat_mask[:,:,z] = lat_mask.astype(int)




    def solve_circ1_laplace(self):
        circ = np.zeros_like(self.lv_mask)
        self.sep_vals = []
        for z in range(self.lv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0:
                self.sep_vals.append([])
                continue
            # Get values for septum
            circ0 = self.circ0[:,:,z]
            sep_vals = circ0[self.sep_mask[:,:,z]==1]*np.pi/2
            self.sep_vals.append(sep_vals)

            # Usual stuff
            labels = define_laplace_labels(self.lv_mask[:,:,z],
                                           self.sep_mask[:,:,z],
                                           self.lat_mask[:,:,z])
            n, ind = lf.uvc_get_index_mapping(labels)
            A = lf.uvc_get_ablock(self.lv_mask[:,:,z], ind, labels, n, self.pixdim)

            # Dirichlet conditions
            RHS = np.zeros(A.shape[0])
            RHS[ind[self.pi_mask[:,:,z]==1]] = np.pi
            RHS[ind[self.sep_mask[:,:,z]==1]] = sep_vals
            RHS[ind[self.mpi_mask[:,:,z]==1]] = -np.pi

            circ[:,:,z] = lf.solve_laplace(A, RHS, self.lv_mask[:,:,z])

        self.circ1 = circ


    def get_circ_bndry(self):       # TODO I should probably do it as in the paper
        self.rvi_ant_mask = np.zeros_like(self.lv_mask)
        self.rvi_post_mask = np.zeros_like(self.lv_mask)
        for z in range(self.lv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue
            circ = self.circ1[:,:,z]
            rv_insert_ant = self.rv_insert_ant[z]
            rv_insert_post = self.rv_insert_post[z]

            # Getting the circ value at the inserts
            rvi_circ_ant = circ[rv_insert_ant[0], rv_insert_ant[1]]
            rvi_circ_post = circ[rv_insert_post[0], rv_insert_post[1]]


            _, ant_div, _ = find_division_mask(circ, rvi_circ_ant, self.lv_mask[:,:,z])
            _, _, post_div = find_division_mask(circ, rvi_circ_post, self.lv_mask[:,:,z])
            ant_div[self.lat_mask[:,:,z]==1] = 0
            post_div[self.lat_mask[:,:,z]==1] = 0

            self.rvi_ant_mask[:,:,z] = ant_div
            self.rvi_post_mask[:,:,z] = post_div


    def solve_circ_laplace(self):
        circ = np.zeros_like(self.lv_mask)
        for z in range(self.lv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue
            out_mask = self.lat_mask[:,:,z] + self.rvi_ant_mask[:,:,z] + self.rvi_post_mask[:,:,z]

            # Usual stuff
            labels = define_laplace_labels(self.lv_mask[:,:,z],
                                           self.sep_mask[:,:,z],
                                           out_mask)
            n, ind = lf.uvc_get_index_mapping(labels)
            A = lf.uvc_get_ablock(self.lv_mask[:,:,z], ind, labels, n, self.pixdim)

            # Dirichlet conditions
            RHS = np.zeros(A.shape[0])
            RHS[ind[self.pi_mask[:,:,z]==1]] = np.pi
            RHS[ind[self.rvi_ant_mask[:,:,z]==1]] = np.pi/3
            RHS[ind[self.sep_mask[:,:,z]==1]] = self.sep_vals[z]
            RHS[ind[self.rvi_post_mask[:,:,z]==1]] = -np.pi/3
            RHS[ind[self.mpi_mask[:,:,z]==1]] = -np.pi

            circ[:,:,z] = lf.solve_laplace(A, RHS, self.lv_mask[:,:,z])

        self.circ = circ


    def correct_circ(self):
        def get_cartesian_dist(mask, circ):
            # Get endo border
            mask_ij = np.vstack(np.where(mask==1)).T
            mask_circ = circ[mask==1]
            order = np.argsort(mask_circ)
            mask_circ = mask_circ[order]
            mask_ij = mask_ij[order]

            # Get cartesian distances
            dist = np.cumsum(np.linalg.norm(np.diff(mask_ij, axis=0), axis=1))
            dist = np.append(0, dist)
            dist = dist/np.max(dist)

            return (dist*2-1), mask_circ


        for z in range(self.lv_mask.shape[-1]):
            if np.sum(self.lv_mask[:,:,z]) == 0: continue
            circ = np.copy(self.circ[:,:,z])
            lat_mask = np.zeros_like(self.lv_mask[:,:,z], dtype=bool)
            lat_mask[self.lv_mask[:,:,z]==1] = np.abs(circ[self.lv_mask[:,:,z]==1]) > np.pi/3
            sep_mask = np.zeros_like(self.lv_mask[:,:,z], dtype=bool)
            sep_mask[self.lv_mask[:,:,z]==1] = np.abs(circ[self.lv_mask[:,:,z]==1]) <= np.pi/3

            correct_circ = np.zeros_like(circ)

            # Getting lines
            endo_mask = self.lv_bndry_endo[:,:,z]
            epi_mask = self.lv_bndry_epi[:,:,z]

            # Find division points
            div, pos_div, neg_div = find_division_mask(self.trans[:,:,z], 0.5,
                                                        self.lv_mask[:,:,z])

            # Sep side
            endo_dist, endo_circ = get_cartesian_dist(endo_mask*sep_mask, circ)
            epi_dist, epi_circ = get_cartesian_dist(epi_mask*sep_mask, circ)
            pos_dist, pos_circ = get_cartesian_dist(pos_div*sep_mask, circ)
            neg_dist, neg_circ = get_cartesian_dist(neg_div*sep_mask, circ)

            func_endo = interp1d(endo_circ, endo_dist, fill_value="extrapolate")
            func_epi = interp1d(epi_circ, epi_dist, fill_value="extrapolate")
            func_pos = interp1d(pos_circ, pos_dist, fill_value="extrapolate")
            func_neg = interp1d(neg_circ, neg_dist, fill_value="extrapolate")

            theta = np.linspace(-np.pi/3, np.pi/3, 361, endpoint=True)
            mean_dist = (func_endo(theta) + func_epi(theta) + func_pos(theta) + func_neg(theta))/4
            func_mean = interp1d(theta, mean_dist)

            post_val = func_mean(-np.pi/3)
            ant_val = func_mean(np.pi/3)

            vals = np.copy(mean_dist)
            vals[theta <= 0] = (vals[theta <= 0])/(post_val)*-1
            vals[theta >= 0] = (vals[theta >= 0])/(ant_val)

            correct_func = interp1d(theta, vals*np.pi/3)
            correct_circ[sep_mask] = correct_func(circ[sep_mask])

            # Lat side
            # invert circ
            circ[circ<=0] = -(circ[circ<=0] + np.pi)
            circ[circ>0] = -(circ[circ>0] - np.pi)

            endo_dist, endo_circ = get_cartesian_dist(endo_mask*lat_mask, circ)
            epi_dist, epi_circ = get_cartesian_dist(epi_mask*lat_mask, circ)
            pos_dist, pos_circ = get_cartesian_dist(pos_div*lat_mask, circ)
            neg_dist, neg_circ = get_cartesian_dist(neg_div*lat_mask, circ)

            func_endo = interp1d(endo_circ, endo_dist, fill_value="extrapolate")
            func_epi = interp1d(epi_circ, epi_dist, fill_value="extrapolate")
            func_pos = interp1d(pos_circ, pos_dist, fill_value="extrapolate")
            func_neg = interp1d(neg_circ, neg_dist, fill_value="extrapolate")

            theta = np.linspace(-np.pi/3*2, np.pi/3*2, 361, endpoint=True)
            mean_dist = (func_endo(theta) + func_epi(theta) + func_pos(theta) + func_neg(theta))/4
            func_mean = interp1d(theta, mean_dist)

            post_val = func_mean(-np.pi*2/3)
            ant_val = func_mean(np.pi*2/3)

            vals = np.copy(mean_dist)
            vals[theta <= 0] = (vals[theta <= 0])/(post_val)*-1
            vals[theta >= 0] = (vals[theta >= 0])/(ant_val)

            correct_func = interp1d(theta, vals*np.pi/3*2)
            correct_circ[lat_mask] = correct_func(circ[lat_mask])
            correct_circ[lat_mask*(correct_circ<=0)] = -(correct_circ[lat_mask*(correct_circ<=0)] + np.pi)
            correct_circ[lat_mask*(correct_circ>0)] = -(correct_circ[lat_mask*(correct_circ>0)] - np.pi)

            self.circ[:,:,z] = correct_circ
            # Visualize
            # circ = np.copy(self.circ[:,:,z])
            # def get_circ_segs(circ):
            #     segs = (circ>2*np.pi/3).astype(int)+(circ>np.pi/3).astype(int) + \
            #                 (circ>0).astype(int) + (circ>-2*np.pi/3).astype(int) + \
            #                 (circ>-np.pi/3).astype(int) + (circ>-np.pi).astype(int)
            #     segs[self.lv_mask[:,:,z]==0] = 0
            #     return segs

            # if z==3:
            #     plt.figure(42, clear=True)
            #     segs = get_circ_segs(circ)
            #     plt.imshow(segs)


    def get_long(self):
        origin = nib.affines.apply_affine(self.affine, np.array([0,0,0]))
        slice_vector_z = nib.affines.apply_affine(self.affine, np.array([0,0,1])) - origin
        sa_long_axis_vector = -slice_vector_z/np.linalg.norm(slice_vector_z)
        self.sa_long_axis_vector = sa_long_axis_vector
        img_apex_point = self.la_landmarks['epi_apex']

        # Slice points
        v_apex_origin = img_apex_point - origin
        v_apex_origin = v_apex_origin - np.dot(sa_long_axis_vector, v_apex_origin)*sa_long_axis_vector
        img_origin = origin + v_apex_origin
        img_slice_points = img_origin - sa_long_axis_vector*np.arange(self.lv_mask.shape[2])[:,None]*self.pixdim[2]

        # Project slices to LA vector position
        vecs = img_slice_points - img_apex_point
        d = (sa_long_axis_vector@vecs.T)/np.dot(sa_long_axis_vector, self.la_long_axis_vector)
        self.slice_long = d/self.long_length

        self.long = np.zeros_like(self.circ)
        long_xyz = self.la_long_axis_vector@(self.lv_xyz - self.apex_lv_endo).T
        long_xyz = long_xyz/self.long_length
        self.long[self.lv_mask==1] = long_xyz

        self.long_plane = np.zeros_like(self.circ)
        for i in range(len(self.slice_long)):
            self.long_plane[:,:,i] = self.slice_long[i]


    def get_long_from_mesh(self, la_landmarks):
        origin = nib.affines.apply_affine(self.affine, np.array([0,0,0]))
        slice_vector_z = nib.affines.apply_affine(self.affine, np.array([0,0,1])) - origin
        sa_long_axis_vector = -slice_vector_z/np.linalg.norm(slice_vector_z)
        self.sa_long_axis_vector = sa_long_axis_vector

        mv = la_landmarks[1]
        apex = la_landmarks[0]
        la_long_axis_vector = mv - apex
        la_long_axis_length = np.linalg.norm(la_long_axis_vector)
        la_long_axis_vector = la_long_axis_vector/la_long_axis_length

        self.long = np.zeros_like(self.circ)
        long_xyz = la_long_axis_vector@(self.lv_xyz - apex).T
        long_xyz = long_xyz/la_long_axis_length
        self.long[self.lv_mask==1] = long_xyz

        self.long_length = la_long_axis_length
        self.la_long_axis_vector = la_long_axis_vector

        self.la_landmarks = {'mv': mv, 'epi_apex': apex}

    def fib_interpolator(self):

        C=self.circ[self.lv_mask==1]
        T=self.trans[self.lv_mask==1]
        L=self.long[self.lv_mask==1]
        self.lv_ctl = np.vstack([C,T,L]).T

        # Get LV ctl coordinates
        slice_c = []
        slice_t = []
        slice_l = []
        slice_fib = []
        for i in range(self.lv_mask.shape[2]):
            if np.all(self.lv_mask[:,:,i]==0): continue
            circ = self.circ[:,:,i]
            trans = self.trans[:,:,i]
            long = self.long[:,:,i]
            fib = self.lv_fib_img[:,:,i]
            slice_c.append(circ[self.lv_mask[:,:,i]==1])
            slice_t.append(trans[self.lv_mask[:,:,i]==1])
            slice_l.append(long[self.lv_mask[:,:,i]==1])
            slice_fib.append(fib[self.lv_mask[:,:,i]==1])

        # Append a slice on top and bottom
        long_dist = self.pixdim[2]/self.long_length
        slice_0_long = np.mean(slice_l[0])
        slice_l_long = np.mean(slice_l[-1])
        if slice_0_long > slice_l_long:
            slice_l = [slice_l[0]+long_dist] + slice_l + [slice_l[-1]-long_dist]
        else:
            slice_l = [slice_l[0]-long_dist] + slice_l + [slice_l[-1]+long_dist]

        slice_c = [slice_c[0]] + slice_c + [slice_c[-1]]
        slice_t = [slice_t[0]] + slice_t + [slice_t[-1]]
        slice_fib = [slice_fib[0]*0] + slice_fib + [slice_fib[-1]*0]

        C = np.concatenate(slice_c)
        T = np.concatenate(slice_t)
        L = np.concatenate(slice_l)
        F = np.concatenate(slice_fib)

        lv_ctl = np.vstack([C,T,L]).T
        lv_ctl[:,0] *= (1+1e-5)
        lv_ctl[:,1] *= (1+1e-5)
        arr = np.array([[-np.pi, 0, 0],
                        [np.pi, 0, 0],
                        [-np.pi, 1, 0],
                        [np.pi, 1, 0],
                        [-np.pi, 0, 1],
                        [np.pi, 0, 1],
                        [-np.pi, 1, 1],
                        [np.pi, 1, 1]])
        lv_ctl = np.vstack([lv_ctl, arr])

        lv_fib = np.append(F, np.zeros(8))

        # Extend circumferential
        lv_ctl_plus = lv_ctl.copy()
        lv_ctl_plus[:,0] += 2*np.pi

        lv_ctl_minus = lv_ctl.copy()
        lv_ctl_minus[:,0] -= 2*np.pi

        lv_ctl = np.vstack([lv_ctl, lv_ctl_plus, lv_ctl_minus])
        lv_fib = np.concatenate([lv_fib, lv_fib, lv_fib])

        # Restrict to -3pi/2, 3pi/2
        mask = (lv_ctl[:,0]>-np.pi*3/2)*(lv_ctl[:,0]<np.pi*3/2)
        lv_ctl = lv_ctl[mask]
        lv_fib = lv_fib[mask]

        self.fib_func = LinearNDInterpolator(lv_ctl, lv_fib, fill_value=0.0)


    def plot_ctl_in_xyz(self, plotly=True):
        cmaps = ['IceFire', 'Inferno', 'RdBu']
        if plotly:
            for i in range(3):
                fig = pf.show_point_cloud(self.lv_xyz, size = 3, color=self.lv_ctl[:,i], cmap=cmaps[i])
                fig = pf.show_point_cloud(self.insert_xyz, fig=fig, size = 6, color='black')
                # fig = pf.show_point_cloud(self.la_landmarks['endo_apex'], fig=fig, size = 6, color='cyan')
                fig = pf.show_point_cloud(self.la_landmarks['epi_apex'], fig=fig, size = 6, color='cyan')
                fig = pf.show_point_cloud(self.la_landmarks['mv'], fig=fig, size = 6, color='magenta')
                fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
                fig.update_scenes(aspectmode='data')
                fig.show()
        else:
            for i in range(3):
                fig = plt.figure(i+1, clear=True)
                ax = fig.add_subplot(projection='3d')
                ax.scatter(self.lv_xyz[:,0],self.lv_xyz[:,1],self.lv_xyz[:,2], c=self.lv_ctl[:,i])
                ax.scatter(*self.la_points['endo_apex'].T, marker='.', color='C0')
                ax.scatter(*self.la_points['epi_apex'].T, marker='.', color='C1')
                ax.scatter(*self.la_points['mv'].T, marker='.', color='C2')
                ax.scatter(*self.la_landmarks['endo_apex'].T, marker='o', color='C0')
                ax.scatter(*self.la_landmarks['epi_apex'].T, marker='o', color='C1')
                ax.scatter(*self.la_landmarks['mv'].T, marker='o', color='C2')
                ax.scatter(*self.insert_xyz.T, marker='o', color='k', s=60)
                ax.set_aspect('equal')


    def plot_xyz_in_ctl(self):

        for i in range(3):
            fig = plt.figure(i+1, clear=True)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.lv_ctl[:,0],self.lv_ctl[:,1],self.lv_ctl[:,2], c=self.lv_xyz[:,i])
            ax.set_aspect('equal')


    def plot_fib_in_xyz(self, la_slices=[], plot="plotly"):
        fib = self.lv_fib

        if len(la_slices) > 0:
            la_xyz = []
            la_fib = []
            for la_slice in la_slices:
                la_xyz.append(la_slice.get_xyz_trans('lv'))
                la_fib.append(2-la_slice.lge_data)

        if plot=="plotly":
            import plot_functions as pf
            fig = pf.show_point_cloud(self.lv_xyz, size = 3, color=fib, cmap='Picnic',
                                      cmin=0, cmax=1, label='sa')
            if len(la_slices) > 0:
                for i in range(len(la_xyz)):
                    fig = pf.show_point_cloud(la_xyz[i], size = 3, color=la_fib[i], fig=fig,
                                              cmap='Picnic', cmin=0, cmax=1, label=la_slices[i].cmr.view)

            fig = pf.show_point_cloud(self.insert_xyz, fig=fig, size = 6, color='black')
            fig = pf.show_point_cloud(self.la_landmarks['endo_apex'], fig=fig, size = 6, color='cyan')
            fig = pf.show_point_cloud(self.la_landmarks['epi_apex'], fig=fig, size = 6, color='cyan')
            fig = pf.show_point_cloud(self.la_landmarks['mv'], fig=fig, size = 6, color='magenta')
            fig = pf.show_point_cloud(np.vstack(list(self.la_points.values())), fig=fig, size = 6, color='black')
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
            fig.update_scenes(aspectmode='data')
            fig.show()

        elif plot=="paraview":
            points = self.lv_xyz
            if len(la_slices) > 0:
                for i in range(len(la_xyz)):
                    points = np.vstack([points, la_xyz[i]])
                    fib = np.append(fib, la_fib[i])
            mesh = io.Mesh(points, {'vertex': np.arange(len(points))[:,None]}, point_data={'fibrosis': fib})
            return mesh

        else:
            fig = plt.figure(1, clear=True)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.lv_xyz[:,0],self.lv_xyz[:,1],self.lv_xyz[:,2], c=fib, cmap='rainbow')
            ax.set_aspect('equal')

    def plot_output_paraview(self, la_slices=[]):
        points = self.lv_xyz
        fib = self.lv_fib

        if len(la_slices) > 0:
            la_xyz = []
            la_fib = []
            for la_slice in la_slices:
                la_xyz.append(la_slice.get_xyz_trans('lv'))
                la_fib.append(2-la_slice.lge_data)

        if len(la_slices) > 0:
            for i in range(len(la_xyz)):
                points = np.vstack([points, la_xyz[i]])
                fib = np.append(fib, la_fib[i])

        circ = np.zeros(len(points))
        circ[:len(self.lv_xyz)] = self.lv_ctl[:,0]
        trans = np.zeros(len(points))
        trans[:len(self.lv_xyz)] = self.lv_ctl[:,1]
        long = np.zeros(len(points))
        long[:len(self.lv_xyz)] = self.lv_ctl[:,2]
        mesh = io.Mesh(points, {'vertex': np.arange(len(points))[:,None]},
                       point_data={'fibrosis': fib,
                                   'circ': circ,
                                   'trans': trans,
                                   'long': long})
        return mesh

    def plot_fib_in_ctl(self):
        fib = self.lv_fib
        fig = plt.figure(2, clear=True)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.lv_ctl[:,0],self.lv_ctl[:,1],self.lv_ctl[:,2], c=fib, cmap='rainbow')
        ax.set_aspect('equal')


    def save_data(self, fname):

        data = {}
        data['fib_func'] = self.fib_func
        sep_vector = np.mean(self.sep_vector[np.all(self.sep_vector!=0, axis=1)], axis=0)
        sep_vector = sep_vector/np.linalg.norm(sep_vector)
        data['septum_vector'] = sep_vector
        data['la_long_axis_vector'] = self.la_long_axis_vector
        data['sa_long_axis_vector'] = self.sa_long_axis_vector
        data['la_long_axis_length'] = self.long_length
        data['mv_centroid'] = self.la_landmarks['mv']
        data['apex_lv_epi'] = self.la_landmarks['epi_apex']
        data['lv_ctl'] = self.lv_ctl
        data['lv_xyz'] = self.lv_xyz
        data['rv_xyz'] = self.rv_xyz
        data['lv_fib'] = self.lv_fib
        data['insert_points'] = self.insert_xyz

        np.save(fname, data, allow_pickle=True)



import nibabel as nib
def save_coord_nii(fname, coord, affine=np.eye(4), header=None):
    img = nib.Nifti1Image(coord, affine, header)
    nib.save(img, fname)



