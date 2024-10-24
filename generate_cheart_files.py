#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 07:42:57 2023

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
import meshio as io
import cheartio as chio

frame = 0
patient = 'ZS-11'
simmodeler_path = '/Users/jjv/Downloads/'
mesh_path = '/Users/jjv/Downloads/'
model_name = 'bv_model'          # output name for CH files
vol_mesh_name = simmodeler_path + 'volume'
surf_mesh_name = simmodeler_path + 'surface'

swap = True

swap_labels = {5: 1,   # endo lv
                6: 2,   # endo rv
                7: 3,   # epi lv
                4: 4,   # epi rv
                9: 5,   # rv septum
                10: 6,   # av
                8: 7,   # pv
                3: 8,   # tv
                2: 9,   # mv
                1: 10}   # junction

if not os.path.exists(mesh_path): os.mkdir(mesh_path)

# Reading .inp
vol_mesh = io.read(vol_mesh_name + '.inp')
surf_mesh = io.read(surf_mesh_name + '.inp')

# Join RV and LV regions
# The LV should be the region with more elements
sizes = [len(vol_mesh.cells[0]), len(vol_mesh.cells[1])]
lv_region = np.argmax(sizes)
rv_region = 1-lv_region
cells = np.vstack([vol_mesh.cells[lv_region].data, vol_mesh.cells[rv_region].data])
region = np.zeros(len(cells), dtype=int)
region[0:sizes[lv_region]] = 1

# Generate boundary file
bdata, _ = chio.create_bfile(surf_mesh, cells, inner_faces = True)

# Write CH files
chio.write_mesh(mesh_path + model_name, vol_mesh.points, cells)
chio.write_dfile(mesh_path + 'region.FE', region)

# Write .vtu
mesh = io.Mesh(vol_mesh.points, {'tetra': cells}, cell_data = {'region': [region]})
io.write(mesh_path + model_name + '.vtu', mesh)

bmesh = io.Mesh(vol_mesh.points, {'triangle': bdata[:,1:-1]},
                cell_data = {'patches': [bdata[:,-1]]})
io.write(mesh_path + model_name + '_boundary.vtu', bmesh)

if swap:
    new_labels = np.zeros(len(bdata))
    labels = bdata[:,-1]

    for i, s in enumerate(swap_labels.keys()):
        new_labels[labels == s] = i+1

    bdata[:,-1] = new_labels

    chio.write_bfile(mesh_path + model_name, bdata)

    bmesh = io.Mesh(vol_mesh.points, {'triangle': bdata[:,1:-1]},
                    cell_data = {'patches': [bdata[:,-1]]})
    io.write(mesh_path + model_name + '_boundary.vtu', bmesh)
