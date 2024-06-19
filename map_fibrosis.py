#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:30:00 2023

@author: jjv
"""

import sys
import os
import numpy as np
import meshio as io
import cheartio as chio
from mapfibrosis.mapper import map_fibrosis

if len(sys.argv) > 1:
    patient = sys.argv[1]
else:
    patient = 'VB-1'
lge_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/LGE/'
mesh_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/mesh/'
output_fldr = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/data/'

if not os.path.exists(output_fldr): os.mkdir(output_fldr)

mesh = io.read(mesh_fldr + 'bv_mesh.vtu')
mesh_info = np.load(mesh_fldr + 'info.npy', allow_pickle=True).item()
img_info = np.load(lge_fldr + 'lge_info.npy', allow_pickle=True).item()

fib_func = img_info['fib_func']
bv_fibrosis, long = map_fibrosis(fib_func, mesh, mesh_info)

# Save results
chio.write_dfile(output_fldr + 'fibrosis.FE', bv_fibrosis)

# For visualization
mesh.point_data['fibrosis'] = bv_fibrosis
io.write(mesh_fldr + 'bv_mesh.vtu', mesh)

# To check, we also overwrite the fit_mesh.vtu
rigid_transform = np.load(lge_fldr + 'rigid_mesh2lge.npz')
fit_mesh = io.Mesh(mesh.points, mesh.cells)
fit_mesh.point_data['fibrosis'] = bv_fibrosis
fit_mesh.point_data['circ'] = mesh.point_data['circ']
fit_mesh.point_data['trans'] = mesh.point_data['trans']
fit_mesh.point_data['long'] = long
fit_mesh.cell_data['rvlv'] = mesh.cell_data['rvlv']
fit_mesh.points = fit_mesh.points@rigid_transform['rotation'].T + rigid_transform['translation']

io.write(lge_fldr + 'fit_mesh.vtu', fit_mesh)

