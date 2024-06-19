#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:14:38 2023

@author: jjv
"""

import sys
from masks2contours.m2c import LGEImage, writeResults, find_apex_mv_estimate, remove_base_nodes
from masks2contours import slicealign
import plot_functions as pf
import numpy as np

# Inputs
if len(sys.argv) > 1:
    patient = sys.argv[1]
else:
    patient = 'VB-1'
fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/LGE/'

seg_files = {'sa': fldr + 'SA_seg',
            'la_2ch': fldr + 'LA_2CH_seg',
            'la_3ch': fldr + 'LA_3CH_seg',
            'la_4ch': fldr + 'LA_4CH_seg',
            }

labels = {'lv_wall': 2, 'lv_fib': 1, 'rv_bp': 3}

# First, we align everything
# Extract all the info required
cmrs = {}
slices = []
views = []
for view in seg_files.keys():
    if seg_files[view] is None: continue
    try:
        cmr = LGEImage(view, seg_files[view], labels)
    except:
        continue
    print('Loading {} nifti'.format(view))
    views.append(view)
    slices += cmr.extract_slices()
    cmrs[view] = cmr

# Align slices
fig = pf.plot_slices(slices)
pf.save_figure(fldr + 'pre_align.html', fig)

print('Loading translations from file...')
translations = {}
found = 0
for view in  views:
    try:
        translations[view] = np.load(fldr + view.upper() + '_seg_translation.npy')
        found += 1
    except:
        print('Translation file for ' + view + ' not found.')
        continue

if found != len(views):
    # If there is more than 1 LA view we can precompute a better initial guess.
    if len(seg_files) > 2:
        slicealign.find_SA_initial_guess(slices)
    slicealign.optimize_stack_translation3(slices, nit=100)
    translations = slicealign.save_translations(fldr, cmrs, slices)

# Generate contours. Because for aligning the septum is not defined, we need to redo
# the reading steps.
cmrs = {}
slices = []
for view in seg_files.keys():
    if seg_files[view] is None: continue
    try:
        cmr = LGEImage(view, seg_files[view], labels)
    except:
        continue
    slices += cmr.extract_slices(translations=translations[view], defseptum=True)
    cmrs[view] = cmr

downsample = 3
contours = []
for slc in slices:
    ctrs = slc.tocontours(downsample)
    contours += ctrs

# Delete base
apex, mv = find_apex_mv_estimate(contours)
remove_base_nodes(contours, apex, mv)

# Save results
writeResults(fldr + 'contours.txt', contours)

# Visualize
fig = pf.plot_contours(contours, background=True)
pf.save_figure(fldr + 'contours.html', fig)

import meshio as io
vertex_contours = pf.contours2vertex(contours)
io.write(fldr + 'contours.vtu', vertex_contours)
