#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:02:21 2023

@author: Javiera Jilberto Vallejos
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
import nibabel as nib
import csv

from niftiutils import readFromNIFTI
import masks2contours.utils as ut
from masks2contours.SelectFromCollection import SelectFromCollection


def correct_labels(seg, labels):
    new_seg = np.copy(seg)
    for i, which in enumerate(['lvbp', 'lv', 'rv']):
        vals = labels[which]
        if type(vals) == list:
            for v in vals:
                new_seg[seg == v] = i+1
        else:
            new_seg[seg == vals] = i+1

    return new_seg

class CMRImage:
    def __init__(self, view, img_file, affine_file=None, valve_file=None, labels=None, frame=0):
        self.view = view

        data, transform, pixspacing, _ = readFromNIFTI(img_file, frame)
        self.data = data
        self.pixdim = pixspacing[0]
        self.zooms = pixspacing
        self.fname = img_file

        if affine_file is not None:
            _, transform, _, _ = readFromNIFTI(affine_file, frame)   # use the transform from the image

        self.affine = transform

        if valve_file is not None:
            self.valves, _, _ = readFromNIFTI(valve_file, frame)
        else:
            self.valves = None

        if labels is None:
            self.labels = {'lvbp': 1.0, 'lv': 2.0, 'rv': 3.0}
        else:
            self.labels = labels


    def extract_slices(self, transfile=None, defseptum=False):
        # get normal
        arr = np.array([[0,0,0],[0,0,1]])
        points = nib.affines.apply_affine(self.affine, arr)
        normal = points[1] - points[0]
        normal = normal/np.linalg.norm(normal)

        if transfile is not None:
            translations = np.load(transfile)

        slices = []
        for n in range(self.data.shape[2]):
            data_slice = self.data[:,:,n]
            if np.all(data_slice==0): continue   # avoid saving slices with no data
            origin = nib.affines.apply_affine(self.affine, np.array([0,0,n]))
            slc = CMRSlice(data_slice, origin, normal, n, self, defseptum=defseptum)
            if transfile is not None:
                slc.accumulated_translation += translations[n]

            if slc.valid:
                slices += [slc]

        return slices

    def inverse_transform(self, point):
        A, t = nib.affines.to_matvec(self.affine)
        ijk = np.linalg.solve(A, point-t)
        ijk = np.floor(ijk)
        return ijk


class LGEImage:
    def __init__(self, view, img_file, lge_labels, affine_file=None):
        self.view = view

        data, transform, pixspacing, _ = readFromNIFTI(img_file, 0)
        self.lge_data = data
        self.pixdim = pixspacing[0]
        self.zooms = pixspacing
        self.fname = img_file

        if affine_file is not None:
            _, transform, _, _ = readFromNIFTI(affine_file, 0)   # use the transform from the image

        self.affine = transform
        self.lge_labels = lge_labels
        self.labels = {'lvbp': 1.0, 'lv': 2.0, 'rv': 3.0}

        # Checks
        self.check_lge_data()

        self.from_lge_to_cmr_labels()


    def check_lge_data(self):
        data_vals = np.round(np.unique(self.lge_data))
        if len(data_vals) == 1:
            if data_vals[0] == 0:
                raise Exception('The segmentation of {} is empty'.format(self.view))
        label_vals = np.append(0,list(self.lge_labels.values()))
        isin = ~np.isin(data_vals, label_vals)
        if np.any(isin):
            pos = np.concatenate(np.where(self.lge_data == data_vals[isin][0]))
            raise Exception('Segmentation of {} has a label with value {} in position ({:d},{:d},{:d}) that does not correspond with the LGE labels'.format(self.view, data_vals[isin][0], *pos, ))

    def from_lge_to_cmr_labels(self):
        lv = np.isclose(self.lge_data, self.lge_labels['lv_wall']) + np.isclose(self.lge_data, self.lge_labels['lv_fib'])
        rv = np.isclose(self.lge_data, self.lge_labels['rv_bp'])

        lvbp = np.zeros_like(lv)
        for i in range(lv.shape[2]):
            slc = lv[:,:,i]
            if np.all(slc==0): continue
            if 'sa' in self.view:
                seed = np.copy(slc)
                seed[1:-1, 1:-1] = slc.max()
                lvbp[:,:,i] = morphology.reconstruction(seed, slc, method='erosion') - slc
            else:
                hull = morphology.convex_hull_image(slc)
                bp = (hull.astype(int) - slc.astype(int)).astype(bool)

                # Need to make sure to only keep largest object
                labelled = measure.label(bp)
                rp = measure.regionprops(labelled)

                # get size of largest cluster
                size = np.sort([j.area for j in rp])
                size = np.mean(size[-2:])

                # remove everything smaller than largest
                bp = morphology.remove_small_objects(bp, min_size=size)

                lvbp[:,:,i] = bp

        self.data = lvbp.astype(int) + lv.astype(int)*2 + rv.astype(int)*3
        self.lv_mask = lv

    def extract_slices(self, translations=None, transfile=None, defseptum=False):
        # get normal
        arr = np.array([[0,0,0],[0,0,1]])
        points = nib.affines.apply_affine(self.affine, arr)
        normal = points[1] - points[0]
        normal = normal/np.linalg.norm(normal)

        if transfile is not None:
            translations = np.load(transfile)

        slices = []
        if len(self.data.shape) == 2:
            data_slice = self.data
            if not np.all(data_slice==0):   # avoid saving slices with no data
                origin = nib.affines.apply_affine(self.affine, np.array([0,0,0]))
                slc = CMRSlice(data_slice, origin, normal, 0, self, defseptum=defseptum)
                if translations is not None:
                    slc.accumulated_translation += translations[0]

                if slc.valid:
                    slices += [slc]
        else:
            for n in range(self.data.shape[2]):
                data_slice = self.data[:,:,n]
                if np.all(data_slice==0): continue   # avoid saving slices with no data
                origin = nib.affines.apply_affine(self.affine, np.array([0,0,n]))
                slc = CMRSlice(data_slice, origin, normal, n, self,
                               lge_data = self.lge_data[:,:,n][self.lv_mask[:,:,n]], defseptum=defseptum)
                if (translations is not None):
                    slc.accumulated_translation += translations[n]

                if slc.valid:
                    slices += [slc]

        return slices


class CMRSlice:
    def __init__(self, image, origin, normal, slice_number, cmr, lge_data=None, defseptum=False):
        self.data = np.round(image)
        self.origin = origin
        self.normal = normal
        self.slice_number = slice_number
        self.cmr = cmr

        # getting pixel coords
        self.valid = self.get_boundaries(define_septum=defseptum)

        i = np.arange(image.shape[0])
        j = np.arange(image.shape[1])
        i, j = np.meshgrid(i,j)
        ij = np.vstack([i.flatten(),j.flatten()]).T
        self.ijk = np.hstack([ij, np.full((len(ij),1), slice_number)]).astype(float)

        # Getting bv pixel coordinates
        lv_ij = np.vstack(np.where(np.isclose(image, self.cmr.labels['lv']))).T
        self.lv_ijk = np.hstack([lv_ij, np.full((len(lv_ij),1), slice_number)]).astype(float)

        bv_ij = np.vstack(np.where(image>1.0)).T
        self.bv_ijk = np.hstack([bv_ij, np.full((len(bv_ij),1), slice_number)]).astype(float)

        all_ij = np.vstack(np.where(image>0.0)).T
        self.all_ijk = np.hstack([all_ij, np.full((len(all_ij),1), slice_number)]).astype(float)

        # Utils for optimization
        self.accumulated_translation = np.zeros(2)  # Only needed if aligning slices
        self.accumulated_matrix = np.eye(2)

        # If lge, we also store lge data
        if lge_data is not None:
            self.lge_data = lge_data

    def get_boundaries(self, define_septum=False):
        seg = self.data

        LVendo = np.isclose(seg, self.cmr.labels['lvbp'])
        LVepi = np.isclose(seg, self.cmr.labels['lv'])
        if not np.all(~LVepi):
            LVepi += LVendo
        RVendo = np.isclose(seg, self.cmr.labels['rv'])

        # Get contours
        LVendoCS = getContoursFromMask(LVendo, irregMaxSize = 20)
        LVepiCS = getContoursFromMask(LVepi, irregMaxSize = 20)
        RVendoCS = getContoursFromMask(RVendo, irregMaxSize = 20)

        is_2chr = False
        if (len(LVendoCS) == 0) and (len(LVepiCS) == 0) and (len(RVendoCS) > 0):    # 2CHr, only RV present
            is_2chr = True

        # Check that LVepi and LVendo do not share any points (in SA)
        if 'sa' in self.cmr.view:
            [dup, _, _] = ut.sharedRows(LVepiCS, LVendoCS)
            if len(dup) > 0:  # If they share rows, the slice is not valid
                return False

        # Differentiate contours for RV free wall and RV septum.
        [RVseptCS, ia, ib] = ut.sharedRows(LVepiCS, RVendoCS)
        RVFW_CS = ut.deleteHelper(RVendoCS, ib, axis = 0) # Delete the rows whose indices are in ib.

        # Remove RV septum points from the LV epi contour.
        if define_septum:
            LVepiCS = ut.deleteHelper(LVepiCS, ia, axis = 0)  # In LVepiCS, delete the rows with index ia.

        LVendoIsEmpty = LVendoCS is None or np.max(LVendoCS.shape) <= 2
        LVepiIsEmpty = LVepiCS is None or np.max(LVepiCS.shape) <= 2
        RVendoIsEmpty = RVendoCS is None or RVendoCS.size == 0

        if not RVendoIsEmpty:
            self.has_rv = True
        else:
            self.has_rv = False

        # If doing long axis, remove line segments at base which are common to LVendoCS and LVepiCS.
        if 'la' in self.cmr.view:
            if os.path.exists(os.path.join(os.path.dirname(self.cmr.fname), self.cmr.view + '_delete_points.npz')):
                print('Loading saved points to delete in contours. If you want to redo the selection, delete the file ' + self.cmr.view + '_delete_points.npz.')
                delete_points = np.load(os.path.join(os.path.dirname(self.cmr.fname), self.cmr.view + '_delete_points.npz'))
                delete_points = np.stack([delete_points['lvendo'], delete_points['lvepi'], delete_points['rvfw']])
            else:
                delete_points = -np.ones([3, self.cmr.data.shape[2], 200], dtype=int)
            if not is_2chr:
                [_, ia, ib] = ut.sharedRows(LVendoCS, LVepiCS)
                LVendoCS = ut.deleteHelper(LVendoCS, ia, axis = 0)
                LVepiCS = ut.deleteHelper(LVepiCS, ib, axis = 0)

                # Delete LV epi contours at the base
                if len(LVepiCS) > 0:
                    if np.any(delete_points[1, self.slice_number] >= 0):
                        delete = delete_points[1, self.slice_number]
                        delete = delete[delete >= 0]
                        LVepiCS = ut.deleteHelper(LVepiCS, delete, axis = 0)
                    else:
                        _, ax = plt.subplots(1,1)
                        ax.scatter(LVendoCS[:, 0], LVendoCS[:, 1], s=10, c='C0')
                        pts = ax.scatter(LVepiCS[:, 0], LVepiCS[:, 1], s=10, c='C1')
                        if self.has_rv:
                            ax.scatter(RVseptCS[:, 0], RVseptCS[:, 1], s=10, c='C2')
                            ax.scatter(RVFW_CS[:, 0], RVFW_CS[:, 1], s=10, c='C3')
                        selector = SelectFromCollection(ax, pts, 'LV epi')
                        plt.show()
                        while selector.run:
                            plt.pause(.5)
                            pass

                        LVepiCS = ut.deleteHelper(LVepiCS, selector.ind, axis = 0)
                        delete_points[1, self.slice_number, :len(selector.ind)] = selector.ind


                # Delete LV epi contours at the base
                if len(LVendoCS) > 0:
                    if np.any(delete_points[0,self.slice_number] >= 0):
                        delete = delete_points[0, self.slice_number]
                        delete = delete[delete >= 0]
                        LVendoCS = ut.deleteHelper(LVendoCS, delete, axis = 0)
                    else:
                        delete = np.zeros(200)
                        _, ax = plt.subplots(1,1)
                        pts = ax.scatter(LVendoCS[:, 0], LVendoCS[:, 1], s=10, c='C0')
                        try:
                            ax.scatter(LVepiCS[:, 0], LVepiCS[:, 1], s=10, c='C1')
                        except:
                            pass
                        if self.has_rv:
                            try:
                                ax.scatter(RVseptCS[:, 0], RVseptCS[:, 1], s=10, c='C2')
                            except:
                                pass
                            ax.scatter(RVFW_CS[:, 0], RVFW_CS[:, 1], s=10, c='C3')
                        selector = SelectFromCollection(ax, pts, 'LV endo')
                        plt.show()
                        while selector.run:
                            plt.pause(.5)
                            pass
                        LVendoCS = ut.deleteHelper(LVendoCS, selector.ind, axis = 0)
                        delete_points[0, self.slice_number, :len(selector.ind)] = selector.ind

            if self.has_rv:
                # Delete RV free wall contours at the base
                if np.any(delete_points[2,self.slice_number] >= 0):
                    delete = delete_points[2, self.slice_number]
                    delete = delete[delete >= 0]
                    RVFW_CS = ut.deleteHelper(RVFW_CS, delete, axis = 0)
                else:
                    _, ax = plt.subplots(1,1)
                    if not is_2chr:
                        ax.scatter(LVendoCS[:, 0], LVendoCS[:, 1], s=10, c='C0')
                        if len(LVepiCS) > 0:
                            ax.scatter(LVepiCS[:, 0], LVepiCS[:, 1], s=10, c='C1')

                        if len(RVseptCS) > 0:
                            ax.scatter(RVseptCS[:, 0], RVseptCS[:, 1], s=10, c='C2')
                    pts = ax.scatter(RVFW_CS[:, 0], RVFW_CS[:, 1], s=10, c='C3')
                    selector = SelectFromCollection(ax, pts, 'RV free wall')
                    plt.show()
                    while selector.run:
                        plt.pause(.5)
                        pass
                    RVFW_CS = ut.deleteHelper(RVFW_CS, selector.ind, axis = 0)
                    delete_points[2, self.slice_number, :len(selector.ind)] = selector.ind

            cmr_folder = os.path.dirname(self.cmr.fname)
            np.savez(os.path.join(cmr_folder, self.cmr.view + '_delete_points.npz'), lvendo=delete_points[0],
                     lvepi=delete_points[1], rvfw=delete_points[2])

        if not LVendoIsEmpty:
            self.lvendo_ijk = np.hstack([LVendoCS, np.full((len(LVendoCS),1), self.slice_number)]).astype(float)
        else:
            self.lvendo_ijk = np.array([])
        if not LVepiIsEmpty:
            self.lvepi_ijk = np.hstack([LVepiCS, np.full((len(LVepiCS),1), self.slice_number)]).astype(float)
        else:
            self.lvepi_ijk = np.array([])
        if not RVendoIsEmpty:
            self.rvendo_ijk = np.hstack([RVFW_CS, np.full((len(RVFW_CS),1), self.slice_number)]).astype(float)
            if not is_2chr:
                if len(RVseptCS) > 0:
                    self.rvsep_ijk = np.hstack([RVseptCS, np.full((len(RVseptCS),1), self.slice_number)]).astype(float)
        else:
            self.rvendo_ijk = np.array([])

        if LVepiIsEmpty:
            print('WARNING: No LV epi segmentation in {}, slice {}'.format(self.cmr.view.upper(), (self.slice_number+1)))
        if LVendoIsEmpty:
            return False
        else:
            return True


    def get_xyz_trans(self, which, translation=np.zeros(2), use_cum_trans=True):
        # Get working ijk
        if which == 'lv':
            working_ijk = np.copy(self.lv_ijk)
        elif which == 'lvendo':
            working_ijk = np.copy(self.lvendo_ijk)
        elif which == 'lvepi':
            working_ijk = np.copy(self.lvepi_ijk)
        elif which == 'lvepisep':
            if self.has_rv:
                working_ijk = [np.copy(self.lvepi_ijk), np.copy(self.rvsep_ijk)]
                working_ijk = np.vstack(working_ijk)
            else:
                working_ijk = np.copy(self.lvepi_ijk)
        elif which == 'rvendo':
            if self.has_rv:
                working_ijk = np.copy(self.rvendo_ijk)
            else:
                return np.array([])
        elif which == 'rvsep':
            if self.has_rv:
                working_ijk = np.copy(self.rvsep_ijk)
            else:
                return np.array([])
        elif which == 'bv':
            working_ijk = np.copy(self.bv_ijk)
        elif which == 'all':
            working_ijk = np.copy(self.all_ijk)

        # Define translation
        if use_cum_trans:
            t = translation + self.accumulated_translation
        else:
            t = translation

        # apply translation
        working_ijk[:,0:2] += t

        return nib.affines.apply_affine(self.cmr.affine, working_ijk)


    def get_xyz_affine(self, which, affine=np.array([0.,0.,0.,0.,0.,0.]), use_cum_trans=True, use_cum_affine=True):
        """ affine is an array of length 6, first 4 are the matrix and the last two the translation"""
        # Get working ijk
        if which == 'lv':
            working_ijk = np.copy(self.lv_ijk)
        elif which == 'lvendo':
            working_ijk = np.copy(self.lvendo_ijk)
        elif which == 'lvepi':
            working_ijk = np.copy(self.lvepi_ijk)
        elif which == 'lvepisep':
            working_ijk = [np.copy(self.lvepi_ijk), np.copy(self.rvsep_ijk)]
            working_ijk = np.vstack(working_ijk)
        elif which == 'rvendo':
            working_ijk = np.copy(self.rvendo_ijk)
        elif which == 'bv':
            working_ijk = np.copy(self.bv_ijk)
        elif which == 'all':
            working_ijk = np.copy(self.all_ijk)
        elif which == 'contours':
            lvendo = np.copy(self.lvendo_ijk)
            lvepi = np.copy(self.lvepi_ijk)
            if self.has_rv:
                rv = np.copy(self.rvendo_ijk)
                working_ijk = np.vstack([lvendo,lvepi,rv])
            else:
                working_ijk = np.vstack([lvendo,lvepi])

        # Define translation
        if use_cum_trans:
            t = affine[4:] + self.accumulated_translation
        else:
            t = affine[4:]
        if use_cum_affine:
            M = (np.eye(2) + affine[0:4].reshape([2,2]))@self.accumulated_matrix
        else:
            M = np.eye(2) + affine[0:4].reshape([2,2])

        # Apply affine transform in centered coordinates to avoid weird deformations
        centroid = np.mean(working_ijk[:,0:2], axis=0)
        t += centroid - M@centroid

        # apply transform
        working_ijk[:,0:2] = working_ijk[:,0:2]@M.T + t

        return nib.affines.apply_affine(self.cmr.affine, working_ijk)


    def tocontours(self, downsample):
        contour_list = ['lvendo', 'lvepi']
        if self.has_rv:
            contour_list += ['rvsep', 'rvendo']

        contours = []
        contours_added = []
        for name in contour_list:
            try:
                contour_points = self.get_xyz_trans(which=name)
            except:
                continue
            contour_points = contour_points[0:(contour_points.size - 1):downsample, :]

            contour = CMRContour(contour_points, name, self.slice_number, self.cmr.view, self.normal)
            contours.append(contour)
            contours_added.append(name)

            if self.has_rv and name == 'rvendo':   # Save rv inserts
                tmpRV_insertIndices = ut.getRVinsertIndices(contour_points)
                if len(tmpRV_insertIndices) == 0: continue
                rv_insert_points = contour_points[tmpRV_insertIndices]

                if ('la' in self.cmr.view) and ('rvsep' in contours_added):
                    rv_insert_points = ut.getLAinsert(rv_insert_points, contours[-2].points)
                contour = CMRContour(rv_insert_points, 'rvinsert', self.slice_number, self.cmr.view, self.normal)
                contours.append(contour)

        return contours



class CMRContour:
    def __init__(self, points, ctype, slice_number, view, normal=None, weight=1):
        self.points = points
        self.slice = slice_number
        self.view = view
        self.ctype = ctype
        self.weight = weight
        self.normal = normal

    def get_cname(self):
        if self.ctype == 'rvinsert':
            return 'RV_INSERT'
        elif self.ctype == 'apexepi':
            return 'APEX_EPI_POINT'
        elif self.ctype == 'apexendo':
            return 'APEX_ENDO_POINT'
        elif self.ctype == 'rvapex':
            return 'APEX_RV_POINT'
        elif self.ctype == 'mv':
            return 'MITRAL_VALVE'
        elif self.ctype == 'tv':
            return 'TRICUSPID_VALVE'
        elif self.ctype == 'av':
            return 'AORTA_VALVE'

        # Get View
        if 'la' in self.view:
            name = 'LAX'
        elif 'sa' in self.view:
            name = 'SAX'
        else:
            raise "Unknown view"

        if self.ctype == 'lvendo':
            return name + '_LV_ENDOCARDIAL'
        elif self.ctype == 'lvepi':
            return name + '_LV_EPICARDIAL'
        elif self.ctype == 'rvendo':
            return name + '_RV_FREEWALL'
        elif self.ctype == 'rvsep':
            return name + '_RV_SEPTUM'
        elif self.ctype == 'rvinsert':
            return name + '_RV_INSERT'



def computeRVinstersWeight(RVinserts):
    RVinsertsWeights = []  # This will store errors.
    auxRVinserts = np.vstack(RVinserts).reshape([2,-1,3])
    numSlices = auxRVinserts.shape[1]
    for i in range(numSlices):
        _inserts1 = np.squeeze(auxRVinserts[i, :, :])
        _inserts2 = np.linspace(0, numSlices - 1, numSlices) #do we need to use 0, slices - 1 for first two args of linspace?
        inserts = np.vstack((_inserts1, _inserts2))
        inserts = inserts.transpose()
        inserts = inserts[np.all(inserts[:, 0:2], 1), :] # Get rid of rows with zeros in the first three columns.

        points = inserts[:, 0:3]
        indices = inserts[:, 3].astype(int)

        # Sort RV insert points by error, and save the normalized error in RVinsertsWeights.
        err = ut.fitLine3D(points)
        RVinsertsWeights[i, indices] = np.abs(err)/np.max(np.abs(err))


    for i in range(0, 2):
        # In the MATLAB script, the following step essentially amounted to horizontally stacking _inserts1^T and _inserts2^T. (^T is matrix transpose).
        # Since transposing a 1 x n ndarray produces a 1 x n ndarray, rather than a n x 1 ndarray (this happens because numpy
        # tries to make matrix transposition more efficent, since tranposition is a very predictable shuffle of old data)
        # we do a different but equivalent operation here. That is, we take the transpose of the ndarray obtained by stacking
        # _inserts2 below _inserts1.
        _inserts1 = np.squeeze(RVinserts[i, :, :])
        _inserts2 = np.linspace(0, numSlices - 1, numSlices) #do we need to use 0, slices - 1 for first two args of linspace?
        inserts = np.vstack((_inserts1, _inserts2))
        inserts = inserts.transpose()
        inserts = inserts[np.all(inserts[:, 0:2], 1), :] # Get rid of rows with zeros in the first three columns.

        points = inserts[:, 0:3]
        indices = inserts[:, 3].astype(int)

        # Sort RV insert points by error, and save the normalized error in RVinsertsWeights.
        err = ut.fitLine3D(points)
        RVinsertsWeights[i, indices] = np.abs(err)/np.max(np.abs(err))

    return RVinsertsWeights


def getContoursFromMask(maskSlice, irregMaxSize):
    '''
    maskSlice is a 2D ndarray, i.e it is a m x n ndarray for some m, n. This function returns a m x 2 ndarray, where
    each row in the array represents a point in the contour around maskSlice.
    '''

    # First, clean the mask.

    # Remove irregularites with fewer than irregMaxSize pixels.Note, the "min_size" parameter to this function is
    # incorrectly named, and should really be called "max_size".
    maskSlice = morphology.remove_small_objects(np.squeeze(maskSlice), min_size=irregMaxSize, connectivity=2)  # Might have to come back and see if connectivity = 2 was the right choice

    # Fill in the holes left by the removal (code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html).
    seed = np.copy(maskSlice)
    seed[1:-1, 1:-1] = maskSlice.max()
    maskSlice = np.squeeze(morphology.reconstruction(seed, maskSlice, method='erosion'))

    # The mask is now cleaned; we can use measure.find contours() now.

    # It is very important that .5 is used for the "level" parameter.
    #
    # From https://scikit-image.org/docs/0.7.0/api/skimage.measure.find_contours.html:
    # "... to find reasonable contours, it is best to find contours midway between the expected “light” and “dark” values.
    # In particular, given a binarized array, do not choose to find contours at the low or high value of the array. This
    # will often yield degenerate contours, especially around structures that are a single array element wide. Instead
    # choose a middle value, as above."
    lst = measure.find_contours(maskSlice, level = .5)

    # lst is a list of m_ x 2 ndarrays; the np.stack with axis = 0 used below converts
    # this list of ndarrays into a single ndarray by concatenating rows.

    # This "else" in the below is the reason this function is necessary; np.stack() does not accept an empty list.
    return np.vstack(lst) if not len(lst) == 0 else np.array([])


def cleanContours(contours, downsample):
    ''' Helper function that returns an m1 x 2 ndarray, for some m1, that is the result of cleaning up "contours".
    "contours" is an m2 x 2 ndarray for some m1.
    '''
    if contours.shape[0] == 0:
        return np.array([])

    # Remove any points which lie outside of image. Note: simply calling np.nonzero() doesn't do the trick; np.nonzero()
    # returns a tuple, and we want the first element of that tuple.
    indicesToRemove = np.nonzero(contours[:, 0] < 0)[0]
    contours = ut.deleteHelper(contours, indicesToRemove, axis = 0)

    # Downsample.
    contours = contours[0:(contours.size - 1):downsample, :]

    # Remove points in the contours that are too far from the majority of the contour. (This probably happens due to holes in segmentation).
    return ut.removeFarPoints(contours)


def add_valves(contours, all_valves, cmrs, translations={}):
    # To define valve points and apex
    valvelabels = np.array(['mv','mv','mv','mv','mv','mv','av','av','tv','tv'], dtype=object)

    for view in all_valves.keys():
        valves = all_valves[view]
        if len(valves.shape) == 2:        # Making it 3D if 2D
            valves = valves[:,:,None]

        mask = valves>0
        pts = np.vstack(np.where(mask)).T
        slices = pts[:,2]
        if len(translations) > 0:
            pts = pts.astype(float)
            pts[:,0:2] += translations[view][slices]
        label = valves[mask].astype(int)
        which = valvelabels[label-1]

        xyz = nib.affines.apply_affine(cmrs[view].affine, pts)

        for i, v in enumerate(which):
            ctr = CMRContour(xyz[i], v, slices[i], view)
            contours += [ctr]

def find_valves_from_la(contours):
    lv_contours = []
    rvfw_contours = []
    rvsep_contours = []
    for ctr in contours:
        if 'la' in ctr.view:
            if 'lvendo' in ctr.ctype:
                lv_contours.append(ctr)
            elif 'rvendo' in ctr.ctype:
                rvfw_contours.append(ctr)
            elif 'rvsep' in ctr.ctype:
                rvsep_contours.append(ctr)

    # Compute the distance for each LA contour in the SA normal direction
    for la in lv_contours:
        points = la.points
        points = np.append(points, points[0, None], axis=0)
        dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
        ind = np.argmax(dist)
        if ind == len(la.points) - 1:
            mv_points = np.vstack([la.points[ind], la.points[0]])
        else:
            mv_points = np.vstack([la.points[ind], la.points[ind+1]])

        # Add contour
        ctr = CMRContour(mv_points, 'mv', la.slice, la.view)
        contours += [ctr]

        # Get an apex estimate
        mv_centroid = np.mean(mv_points, axis=0)
        mv_dist = np.linalg.norm(points - mv_centroid, axis=1)
        apex_point = points[np.argmax(mv_dist)]
        la_vector = mv_centroid - apex_point
        la_vector = la_vector/np.linalg.norm(la_vector)

    # Do the same for the RV
    if len(rvsep_contours) == 0:
        for la in rvfw_contours:
            points = la.points
            points = np.append(points, points[0, None], axis=0)
            dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
            ind = np.argmax(dist)
            if ind == len(la.points) - 1:
                tv_points = np.vstack([la.points[ind], la.points[0]])
            else:
                tv_points = np.vstack([la.points[ind], la.points[ind+1]])

    else:
        for la in rvfw_contours:
            points = la.points
            dist = np.linalg.norm(points-apex_point, axis=1)
            tv_point1 = points[np.argmax(dist)]

        for la in rvsep_contours:
            points = la.points
            dist = np.linalg.norm(points-apex_point, axis=1)
            tv_point2 = points[np.argmax(dist)]

        tv_points = np.vstack([tv_point1, tv_point2])
    ctr = CMRContour(tv_points, 'tv', la.slice, la.view)
    contours += [ctr]

    # fig = plt.figure(2, clear=True)
    # ax = fig.add_subplot(111, projection='3d')

    # for ctr in lv_contours:
    #     points = ctr.points
    #     ax.plot(points[:, 0], points[:, 1], points[:, 2])
    # for ctr in rvfw_contours:
    #     points = ctr.points
    #     ax.plot(points[:, 0], points[:, 1], points[:, 2])

    # ax.plot(mv_points[:, 0], mv_points[:, 1], mv_points[:, 2], 'ro')
    # ax.plot(tv_points[:, 0], tv_points[:, 1], tv_points[:, 2], 'ro')
    # ax.plot(apex_point[0], apex_point[1], apex_point[2], 'ro')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()


def modify_weights_by_la(contours):
    nslices = len(contours)
    ind = np.arange(nslices)/(nslices-1)

    tfunc1 = np.pi/3*(1-4*ind)
    tfunc2 = 4*np.pi/3*(ind-3/4)
    weight = np.cos(tfunc1)*(ind<1/4) + 1*((ind>1/4)*(ind<3/4)) + np.cos(tfunc2)*(ind>3/4)

    for i, ctr in enumerate(contours):
        ctr.weight = weight[i]


def modify_sa_weights(contours):
    # Find sa contours
    endo_sa_contours = []
    epi_sa_contours = []
    for ctr in contours:
        if 'sa' in ctr.view:
            if 'lvendo' in ctr.ctype:
                endo_sa_contours.append(ctr)
            elif 'lvepi' in ctr.ctype:
                epi_sa_contours.append(ctr)

    modify_weights_by_la(endo_sa_contours)
    modify_weights_by_la(epi_sa_contours)


def add_apex(contours, cmrs, adjust_weights=False):
    # ENDO: Split la and sa contours
    sa_contours = []
    la_contours = []
    mv_points = []
    for ctr in contours:
        if 'lvendo' in ctr.ctype:
            if 'la' in ctr.view:
                la_contours.append(ctr)
            else:
                sa_contours.append(ctr)
        elif 'mv' in ctr.ctype:
            mv_points.append(ctr.points)
    if len(la_contours) == 0:
        return
    mv_points = np.vstack(mv_points)

    # Need to find if the slices are ordered from apex to base or base to apex
    mv_centroid = np.mean(mv_points, axis=0)
    mv_ij = cmrs['sa'].inverse_transform(mv_centroid)
    nslices = cmrs['sa'].data.shape[-1]
    if mv_ij[2] > nslices/2:
        apex_slice1 = 0
        apex_slice2 = 1
    else:
        apex_slice1 = -1
        apex_slice2 = -2

    # Last SA contour is always the most apical one
    sa1_centroid = np.mean(sa_contours[apex_slice1].points, axis=0)
    sa2_centroid = np.mean(sa_contours[apex_slice2].points, axis=0)

    # Find long_axis vector
    la_vector = sa2_centroid - sa1_centroid
    la_vector = la_vector/np.linalg.norm(la_vector)

    # Find lowest point in each la contour
    aux = []
    for ctr in la_contours:
        dist = ctr.points@la_vector
        ind = np.argmin(dist)
        aux.append(np.append(ctr.points[ind], dist[ind]))
    aux = np.vstack(aux)
    points = aux[:,0:3]
    dist = aux[:,3]
    endo_la_apex = points[np.argmin(dist)]

    vector = sa1_centroid-endo_la_apex
    endo_apex = sa1_centroid - (la_vector*np.dot(vector, la_vector))

    ctr = CMRContour(endo_apex, 'apexendo', 0, 'la')
    contours += [ctr]


    # EPI: Split la and sa contours
    sa_contours = []
    la_contours = []
    for ctr in contours:
        if 'lvepi' in ctr.ctype:
            if 'la' in ctr.view:
                la_contours.append(ctr)
            else:
                sa_contours.append(ctr)

    if len(la_contours) == 0:
        return

    # Find lowest point in each la contour
    aux = []
    for ctr in la_contours:
        dist = ctr.points@la_vector
        ind = np.argmin(dist)
        aux.append(np.append(ctr.points[ind], dist[ind]))
    aux = np.vstack(aux)
    points = aux[:,0:3]
    dist = aux[:,3]
    epi_la_apex = points[np.argmin(dist)]

    dist_epi_endo = (endo_la_apex-epi_la_apex)@la_vector
    apex = endo_apex - (la_vector*dist_epi_endo)

    ctr = CMRContour(apex, 'apexepi', 0, 'la')
    contours += [ctr]


def add_rv_apex(contours, cmrs):
    # Split la and sa contours
    safw_contours = []
    lafw_contours = []
    sasep_contours = []
    lasep_contours = []
    lv_epi_contours = []
    tv_points = []
    lv_apex = []
    for ctr in contours:
        if 'rvendo' in ctr.ctype:
            if 'la' in ctr.view:
                lafw_contours.append(ctr)
            else:
                safw_contours.append(ctr)
        elif 'rvsep' in ctr.ctype:
            if 'la' in ctr.view:
                lasep_contours.append(ctr)
            else:
                sasep_contours.append(ctr)
        elif 'lvepi' in ctr.ctype:
            if 'la' in ctr.view:
                lv_epi_contours.append(ctr)
            else:
                lv_epi_contours.append(ctr)
        elif 'mv' in ctr.ctype:
            tv_points.append(ctr.points)
        elif 'apexendo' in ctr.ctype:
            lv_apex.append(ctr.points)
    if len(lafw_contours) == 0:
        return
    tv_points = np.vstack(tv_points)
    lv_apex = np.vstack(lv_apex)

    # Merge FW and SEP contours
    rv_contours = []
    for i in range(len(safw_contours)):
        if i > len(sasep_contours)-1:
            rv_contours.append(safw_contours[i])
        else:
            lim_point1 = sasep_contours[i].points[0]
            dist1 = np.linalg.norm(lim_point1-safw_contours[i].points, axis=1)
            insert1 = np.argmin(dist1)
            lim_point2 = sasep_contours[i].points[-1]
            dist2 = np.linalg.norm(lim_point2-safw_contours[i].points, axis=1)
            insert2 = np.argmin(dist2)
            if insert1 > insert2:
                inserts = insert2, insert1
                sep_points = sasep_contours[i].points[::-1]
            else:
                inserts = insert1, insert2
                sep_points = sasep_contours[i].points

            fw_points = np.vstack([safw_contours[i].points[inserts[1]:], safw_contours[i].points[0:inserts[0]+1]])
            points = np.vstack([sep_points, fw_points])
            rv_contours.append(CMRContour(points, 'rvendo', safw_contours[i].slice, safw_contours[i].view, safw_contours[i].normal))

    # Need to find if the slices are ordered from apex to base or base to apex
    tv_centroid = np.mean(tv_points, axis=0)
    tv_ij = cmrs['sa'].inverse_transform(tv_centroid)
    nslices = cmrs['sa'].data.shape[-1]
    if tv_ij[2] > nslices/2:
        apex_slice0 = 0
        apex_slice1 = 1
        apex_slice2 = 2
    else:
        apex_slice0 = -1
        apex_slice1 = -2
        apex_slice2 = -3


    # Last SA contour is always the most apical one
    sa0_centroid = np.mean(rv_contours[apex_slice0].points, axis=0)
    sa1_centroid = np.mean(rv_contours[apex_slice1].points, axis=0)
    sa2_centroid = np.mean(rv_contours[apex_slice2].points, axis=0)

    # Find the area of these slices
    sa1_area = ut.calculate_area_of_polygon_3d(rv_contours[apex_slice1].points, normal=rv_contours[apex_slice1].normal)
    sa2_area = ut.calculate_area_of_polygon_3d(rv_contours[apex_slice2].points, normal=rv_contours[apex_slice2].normal)

    # Long_axis vector
    la_vector = rv_contours[apex_slice1].normal
    # Check that la_vector points toward the base
    if np.dot(sa2_centroid-sa1_centroid, la_vector) < 0:
        la_vector = -la_vector
    sa1_z = np.dot(sa1_centroid-lv_apex, la_vector)
    sa2_z = np.dot(sa2_centroid-lv_apex, la_vector)


    # Extrapolate to find when the area becomes 0
    rv_apex_z = sa1_z + (0 - sa1_area)/(sa2_area - sa1_area)*(sa2_z - sa1_z)
    dist_z = np.abs(rv_apex_z-sa1_z)

    sa_la_vector = sa2_centroid - sa1_centroid
    sa_la_vector = sa_la_vector/np.linalg.norm(sa_la_vector)
    aux = -dist_z/(np.dot(sa_la_vector, la_vector))
    rv_apex = sa0_centroid + aux*sa_la_vector/2

    ctr = CMRContour(rv_apex, 'rvapex', 0, 'la')
    contours += [ctr]



def writeResults(fname, contours, frame=0):

    # Set up file writers.
    try:
        file = open(fname, "w", newline = "", encoding = "utf-8")
    except Exception as e:
        print(e)
        exit()

    # x    y    z    contour type    slice    weight    time frame
    def writePoint(point, ctype, slicenum, weight = 1):
        writer.writerow(["{:0.6f}".format(point[0]), "{:0.6f}".format(point[1]), "{:0.6f}".format(point[2]),
                         ctype, "{:d}".format(slicenum + 1), "{:0.4f}".format(weight), "{:d}".format(frame)])

    def writeContour(ctr):
        points = ctr.points
        if len(points.shape) == 2:
            for k in range(0, len(points)):
                writePoint(points[k], ctr.get_cname(), ctr.slice, weight=ctr.weight)
        else:
            writePoint(points, ctr.get_cname(), ctr.slice, ctr.weight)

    writer = csv.writer(file, delimiter = "\t")
    writer.writerow(["x", "y", "z", "contour type", "slice", "weight", "time frame"])

    for ctr in contours:
        writeContour(ctr)


