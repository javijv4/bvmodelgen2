#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 08:17:37 2023

@author: Javiera Jilberto Vallejos
"""
import sys
import numpy as np
import meshio as io
import cheartio as chio
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import KDTree

if len(sys.argv) > 1:
    patient = sys.argv[1]
else:
    patient = 'BI-18'
mesh_folder = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/{}/mesh/'.format(patient)

endo_patch = 1

long_lims = [0.2,0.9]

bv_mesh = chio.read_mesh(mesh_folder + 'bv_model', meshio=True)
bv_bdata = chio.read_bfile(mesh_folder + 'bv_model')
uvc_mesh = io.read(mesh_folder + 'lv_mesh.vtu')
xyz = uvc_mesh.points

trans = uvc_mesh.point_data['trans']
circ = uvc_mesh.point_data['circ']
long = uvc_mesh.point_data['long']

endo_long_mask = (long > long_lims[0])*(long < long_lims[1])
epi_long_mask = (long > long_lims[0]*0.9)*(long < long_lims[1]*1.1)

endo_nodes = np.where((trans == 0)*endo_long_mask)[0]
epi_nodes = np.where((trans == 1)*epi_long_mask)[0]

epi_circ = circ[epi_nodes]
epi_long = long[epi_nodes]
epi_ext_circ = np.concatenate([epi_circ-2*np.pi, epi_circ, epi_circ+2*np.pi])
epi_ext_long = np.concatenate([epi_long, epi_long, epi_long])   

epi_cl = np.vstack([epi_ext_circ, epi_ext_long]).T
epi_xyz_ext = np.vstack([xyz[epi_nodes], xyz[epi_nodes], xyz[epi_nodes]])
epi_func = LinearNDInterpolator(epi_cl, epi_xyz_ext)

endo_cl = np.vstack([circ[endo_nodes], long[endo_nodes]]).T
endo_func = LinearNDInterpolator(endo_cl, xyz[endo_nodes])

# Evaluate thickness at endo nodes
endo_xyz = xyz[endo_nodes]
epi_xyz = epi_func(endo_cl)
thickness = np.linalg.norm(endo_xyz - epi_xyz, axis=1)


# Grab the endo surface from the bv mesh
bv_endo_faces = bv_bdata[bv_bdata[:,-1]==endo_patch,1:-1]
bv_endo_nodes = np.unique(bv_endo_faces)

# Find correspondance from bv_endo_nodes to endo_nodes
tree = KDTree(bv_mesh.points)
_, corr = tree.query(xyz[endo_nodes])

surf_mesh = io.Mesh(bv_mesh.points, {'triangle': bv_endo_faces})
surf_mesh.point_data['thickness'] = np.zeros(len(bv_mesh.points))
surf_mesh.point_data['thickness'][corr] = thickness

io.write(mesh_folder + 'wall_thickness.vtu', surf_mesh)

