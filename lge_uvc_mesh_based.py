#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:24:20 2024

@author: Javiera Jilberto Vallejos
"""

import sys
import numpy as np
import meshio as io
import cheartio as chio
from imuvcgen.ImageUVC import UVC
from masks2contours import slicealign, m2c

if len(sys.argv) > 1:
    patient = sys.argv[1]
else:
    patient = 'TS-9'
lge_path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/LGE/'
mesh_path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/mesh/'

uvc_mesh = io.read(mesh_path + 'lv_mesh.vtu')
bdata = chio.read_bfile(mesh_path + 'bv_model')

rigid_transform = np.load(lge_path + 'rigid_mesh2lge.npz')
lge_translations = np.load(lge_path + 'SA_seg_translation.npy')

seg_files = {'sa': lge_path + 'SA_seg',
            'la_4ch': lge_path + 'LA_4CH_seg',
            }

trans_files = { 'sa': lge_path + 'SA_seg_translation.npy',
                'la_4ch': lge_path + 'LA_4CH_seg_translation.npy',
                }

labels = {'lv_wall': 2, 'lv_fib': 1, 'rv_bp': 3}

# Rotate mesh
uvc_mesh.points = uvc_mesh.points@rigid_transform['rotation'].T + rigid_transform['translation']
la_landmarks = rigid_transform['la_landmarks']

# Extract all the info required
cmrs = {}
slices = []

cmrs = {}
slices = []
for view in seg_files.keys():
    if seg_files[view] is None: continue
    cmr = m2c.LGEImage(view, seg_files[view], labels)
    slices += cmr.extract_slices(transfile=trans_files[view])
    cmrs[view] = cmr

sa_cmr = cmrs['sa']

uvc_img = UVC(sa_cmr, lge_translations)
uvc_img.compute_slice_centroids()
uvc_img.get_trans_bndry()
uvc_img.solve_trans_laplace()

uvc_img.get_circ0_bndry()
uvc_img.solve_circ0_laplace()

uvc_img.get_circ1_bndry()
uvc_img.solve_circ1_laplace()

uvc_img.get_circ_bndry()
uvc_img.solve_circ_laplace()
uvc_img.correct_circ()

uvc_img.get_long_from_mesh(la_landmarks)

# Fibrosis mapping
uvc_img.fib_interpolator()
uvc_img.save_data(lge_path + 'lge_info.npy')

la_slices = []
for slc in slices:
    if 'la' in slc.cmr.view:
        la_slices.append(slc)
mesh = uvc_img.plot_output_paraview(la_slices)
io.write(lge_path + 'fibrosis.vtu', mesh)
