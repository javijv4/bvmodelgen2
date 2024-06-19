#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:16:45 2024

@author: Javiera Jilberto Vallejos
"""

import os
import sys
import numpy as np
from bvfitting import BiventricularModel, GPDataSet, ContourType
import meshio as io

if len(sys.argv) > 1:
    patient = sys.argv[1]
else:
    patient = 'KR-13'
lge_path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/LGE/'
geo_path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/Images/test_frame0/'

saving_path = lge_path      # this is where the results are saved
weight_GP = 1
low_smoothing_weight = 2
transmural_weight = 2


contours_to_plot = [ContourType.LAX_RA,
                    ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                    ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                    ContourType.SAX_LV_ENDOCARDIAL,
                    ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                    ContourType.APEX_ENDO_POINT, ContourType.APEX_EPI_POINT,
                    ContourType.MITRAL_VALVE, ContourType.TRICUSPID_VALVE,
                    ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL,
                    ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                    ContourType.LAX_RV_EPICARDIAL, ContourType.SAX_RV_OUTLET,
                    ContourType.PULMONARY_VALVE, ContourType.AORTA_VALVE,
                    ContourType.AORTA_PHANTOM, ContourType.MITRAL_PHANTOM,
                    ContourType.TRICUSPID_PHANTOM,
                    ]

# Load contour points from LGE image
filename = os.path.join(lge_path, 'contours.txt')
dataset = GPDataSet(filename)

# Set LV points to higher weight
dataset.weights[
    dataset.contour_type == ContourType.SAX_LV_ENDOCARDIAL] \
    = 3
dataset.weights[
    dataset.contour_type == ContourType.SAX_LV_EPICARDIAL] \
    = 3
dataset.weights[
    dataset.contour_type == ContourType.SAX_RV_SEPTUM] \
    = 3
dataset.weights[
    dataset.contour_type == ContourType.LAX_LV_ENDOCARDIAL] \
    = 50
dataset.weights[
    dataset.contour_type == ContourType.LAX_LV_EPICARDIAL] \
    = 50
dataset.weights[
    dataset.contour_type == ContourType.LAX_RV_SEPTUM] \
    = 50
# dataset.weights[
#     dataset.contour_type == ContourType.LAX_RV_FREEWALL] \
#     = 10



# Loads biventricular control_mesh
model_path = "src/bvfitting/template" # folder of the control mesh
bvmodel = BiventricularModel(model_path, filemod='_mod')

# Load control points fitted to the Standar MRI geometry
control_points = np.load(geo_path + 'control_points.npy')
bvmodel.update_control_mesh(control_points)
og_control_points = control_points

# Calculate rigid fit
# https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
rotations = []
translations = []
for i in range(100):
    [index, weights, distance_prior, projected_points_basis_coeff] = bvmodel.compute_data_xi(weight_GP, dataset)
    model_position = np.linalg.multi_dot([projected_points_basis_coeff, bvmodel.control_mesh])

    # vertex = io.Mesh(model_position, {'vertex': np.arange(len(model_position))[:,None]})
    # io.write('check.vtu', vertex)

    data_position = dataset.points_coordinates[index]
    # vertex = io.Mesh(data_position, {'vertex': np.arange(len(data_position))[:,None]})
    # io.write('check2.vtu', vertex)

    # Step 1. Centroids
    model_centroid = np.average(model_position, axis=0, weights=weights)
    data_centroid = np.average(data_position, axis=0, weights=weights)

    # Step 2. Centered vectors
    model_centered = model_position - model_centroid
    data_centered = data_position - data_centroid

    # Step 3. Covariance matrix
    X = model_centered.T
    Y = data_centered.T
    W = np.diag(weights)
    S = X@W@Y.T

    # Step 4. SVD
    U, O, Vt = np.linalg.svd(S)
    V = Vt.T

    detVUt = np.linalg.det(V@U.T)
    D = np.eye(len(O))
    D[-1,-1] = detVUt

    R = V@D@U.T

    # Step 5. Translation
    trans = data_centroid - R@model_centroid

    new_points = model_position@R.T + trans

    rot_control_points = bvmodel.control_mesh@R.T + trans

    rotations.append(R)
    translations.append(trans)

    displacement = rot_control_points - bvmodel.control_mesh
    bvmodel.update_control_mesh(np.add(bvmodel.control_mesh, displacement))

    disp_norm = np.linalg.norm(displacement)
    print('It {:d}, displacement norm {:2.3e}'.format(i, disp_norm))
    if disp_norm < 1e-2:
        break

# Calculate total rotations/translations
R = np.eye(3)
t = np.zeros(3)
for i, j in enumerate(reversed(range(len(rotations)))):
    t += R@translations[j]
    R = R@rotations[j]

# Save
la_landmarks = bvmodel.get_long_axis_landmarks()
np.savez(lge_path + 'rigid_mesh2lge.npz', rotation=R, translation=t, la_landmarks=la_landmarks)

# Save to visualize
xyz = bvmodel.et_pos
ien = bvmodel.et_indices
mesh = io.Mesh(xyz, {'triangle': ien})
io.write(lge_path + 'fit_mesh.vtu', mesh)