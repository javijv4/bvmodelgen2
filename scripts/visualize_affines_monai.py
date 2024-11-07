#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/10/21 11:14:52

@author: Javiera Jilberto Vallejos 
'''

import os
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from plot_functions import show_point_cloud
from scipy.spatial.transform import Rotation as R 
import monai.transforms as mt

fldr = '/home/jilberto/Downloads/LGE_Javi_check/LGE/'

seg_files = {'sa': fldr + 'sa_seg',
            'la_2ch': fldr + 'LA_2CH_seg',
            'la_3ch': fldr + 'LA_3CH_seg',
            'la_4ch': fldr + 'LA_4CH_seg',
            }

fig = go.Figure()

for key, file in seg_files.items():
    img = mt.LoadImage(image_only=True)(file + '.nii.gz')
    data = img.numpy().astype(float)
    affine = img.affine
    points_ijk = np.vstack(np.where(data)).T
    points_xyz = nib.affines.apply_affine(affine, points_ijk)
    show_point_cloud(points_xyz, fig=fig, opacity=0.5, size=5, label=key)

fig.show()

# file = seg_files['sa']
# img = nib.load(file + '.nii.gz')
# data = img.get_fdata().astype(float)
# affine = Eidolon_affine(img)
# points_ijk = np.vstack(np.where(data)).T
# points_xyz = nib.affines.apply_affine(affine, points_ijk)
# show_point_cloud(points_xyz, fig=fig, opacity=0.5, size=5, label=os.path.basename(file).split('.')[0])

# file = seg_files['la_2ch']
# img = nib.load(file + '.nii.gz')
# data = img.get_fdata().astype(float)
# affine = Eidolon_affine(img)
# points_ijk = np.vstack(np.where(data)).T
# points_xyz = nib.affines.apply_affine(affine, points_ijk)
# show_point_cloud(points_xyz, fig=fig, opacity=0.5, size=5, label=os.path.basename(file).split('.')[0])

# fig.show()


