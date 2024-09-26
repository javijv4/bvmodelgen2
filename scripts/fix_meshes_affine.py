#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/08/22 13:22:45

@author: Javiera Jilberto Vallejos 
'''
import numpy as np
from niftiutils import readFromNIFTI
import meshio as io
import cheartio as chio
import nibabel as nib

patient = 'KL-4'
patients = ['AB-17', 'AS-10', 'AV-19', 'BI-18', 'CA-15', 'CW-21', 'DM-23', 
            'JL-3', 'JN-8', 'KL-4', 'KL-5', 'KR-13', 'MB-16', 'SL-16', 'TS-9', 'VB-1', 'ZS-11']

# for patient in patients:
#     path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/Images/'

#     _, affine, _, _ = readFromNIFTI(path + 'SA_seg', 0, correct_ras=False)
#     _, hat_affine, _, _ = readFromNIFTI(path + 'SA_seg', 0, correct_ras=True)

#     A, t = nib.affines.to_matvec(affine)
#     A_hat, t_hat = nib.affines.to_matvec(hat_affine)

#     mesh = chio.read_mesh(path + '../mesh/bv_model', meshio=True)
#     xyz = mesh.points

#     mat = A_hat@np.linalg.inv(A)

#     new_xyz = (mat@(xyz-t).T).T + t_hat
#     mesh.points = new_xyz

#     chio.write_mesh(path + '../mesh/bv_model', new_xyz, mesh.cells[0].data)



for patient in patients:
    path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/LGE/'

    _, affine, _, _ = readFromNIFTI(path + 'SA_seg', 0, correct_ras=False)
    _, hat_affine, _, _ = readFromNIFTI(path + 'SA_seg', 0, correct_ras=True)

    A, t = nib.affines.to_matvec(affine)
    A_hat, t_hat = nib.affines.to_matvec(hat_affine)

    mesh = io.read(path + 'fit_mesh.vtu')
    xyz = mesh.points

    mat = A_hat@np.linalg.inv(A)

    new_xyz = (mat@(xyz-t).T).T + t_hat
    mesh.points = new_xyz

    io.write(path + 'fit_mesh.vtu', mesh)