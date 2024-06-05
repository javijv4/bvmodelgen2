#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:02:21 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
from skimage import morphology, measure

def clean_mask(mask, irregMaxSize=20):
    cleanmask = morphology.remove_small_objects(np.squeeze(mask), min_size=irregMaxSize, connectivity=2)

    # Fill in the holes left by the removal (code from https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html).
    seed = np.copy(cleanmask)
    seed[1:-1, 1:-1] = cleanmask.max()
    cleanmask = np.squeeze(morphology.reconstruction(seed, cleanmask, method='erosion'))

    return cleanmask


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


def check_seg_valid(view, data, labels, autoclean=True):
    new_data = np.copy(data)
    valid = np.ones(new_data.shape[2], dtype=bool)
    save = True

    lv_area = np.zeros(data.shape[2])
    rv_area = np.zeros(data.shape[2])
    lvbp_area = np.zeros(data.shape[2])
    for i in range(data.shape[2]):
        slc = data[:,:,i]

        # Get segmentations
        lv = np.isclose(slc, labels['lv'])
        rv = np.isclose(slc, labels['rv'])
        lvbp = np.isclose(slc, labels['lvbp'])
        mask = lv+rv+lvbp

        if np.max(mask) == 0:
            valid[i] = False
            continue


        # Print warning if segmentations are empty
        if np.all(lv == 0):
            print('WARNING: No LV segmentation in {}, slice {}'.format(view.upper(), (i+1)))
        if np.all(rv == 0):
            if view != 'la_2ch':
                print('WARNING: No RV segmentation in {}, slice {}'.format(view.upper(), (i+1)))
        if np.all(lvbp == 0):
            print('WARNING: No LVBP segmentation in {}, slice {}'.format(view.upper(), (i+1)))


        rv = clean_mask(rv)
        lvbp = clean_mask(lvbp)

        if 'sa' in view:
            lv = clean_mask(lv) - lvbp
        else:
            lv = clean_mask(lv)
        cleanmask = clean_mask(mask)
        mask = lv+rv+lvbp

        if np.max(mask) == 0:
            valid[i] = False
            print('Something is wrong in {}, slice {}'.format(view.upper(), (i+1)))
            continue

        # First check: Are there holes?
        if np.max(cleanmask-mask) > 0:
            save = False
            valid[i] = False
            print('There are holes in {}, slice {}'.format(view.upper(), (i+1)))
        if 'sa' in view:
            if np.max(lv) == 0: continue

            # Second check: Is the LV somewhat round? (Only SA)
            props = measure.regionprops(lv.astype(int))[0]
            if props.eccentricity > 0.5:
                valid[i] = False
                print('Eccentricity > 0.5 in {}, slice {}'.format(view.upper(), (i+1)))

            # Third check: is the LV wall close?
            lst = measure.find_contours(lv, level = .5)
            if len(lst) != 2:
                save = False
                valid[i] = False
                print('LV wall is not closed in {}, slice {}'.format(view.upper(), (i+1)))

            # Check that the rv is one region
            label = measure.label(rv)
            props = measure.regionprops(label)
            if len(props) > 1:
                area = np.array([p.area for p in props])
                order = np.argsort(area)[::-1]
                area = area[order]
                if area[1]/area[0] > 0.01:
                    if autoclean:
                        rv[:] = 0
                    print('RV is not one region in {}, slice {}, deleting...'.format(view.upper(), (i+1)))
                else:
                    rv = label == 1


        # TODO Add check for closed LA LV wall
        if not valid[i] and autoclean:
            new_data[:,:,i] = np.zeros_like(slc)
            save=True
        else:
            # calculating area
            lv_area[i] = np.sum(lv)
            rv_area[i] = np.sum(rv)
            lvbp_area[i] = np.sum(lvbp)
            new_data[:,:,i] = lvbp + 2*lv + 3*rv

    # Check for abrupt changes in area the rv
    if autoclean:
        nslices = np.sum(rv_area>0)
        diff_area = np.diff(rv_area[rv_area>0])
        change_area = np.zeros(len(rv_area))
        change_area[rv_area>0] = np.append(np.abs(diff_area/rv_area[rv_area>0][:-1]), 0.)
        if np.sum(diff_area>0) > nslices/2:  # lower slices are apex
            big_changes = np.where(change_area > 0.5)[0]
            for ind in big_changes:
                if ind > len(rv_area)/2:
                    print('Abrupt area change found in slice {}, deleting...'.format(ind))
                    new_data[:,:,ind] = 0

        else:       # lower slices are base
            big_changes = np.where(change_area > 2)[0]
            for ind in big_changes:
                if ind < len(rv_area)/2:
                    print('Abrupt area change found in slice {}, deleting...'.format(ind))
                    new_data[new_data[:,:,ind]==3,ind] = 0




    if np.all(~valid): save=False

    return new_data, save