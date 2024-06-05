#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:02:21 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
from skimage import morphology

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
