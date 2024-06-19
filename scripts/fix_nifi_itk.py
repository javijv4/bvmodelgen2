#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:43:35 2020

@author: javijv4
"""

import nibabel as nib
import numpy as np

def affine_pixdim(affine):
    vector = affine@np.array([0,0,1,0])-affine@np.array([0,0,0,0])
    pixdim = np.linalg.norm(vector[0:3])
    return pixdim

def get_correct_affine(img):
    zooms = affine.header.get_zooms()[0:3]

    if np.isclose(affine_pixdim(img.affine), zooms[2]): return img.affine
    elif np.isclose(affine_pixdim(img.get_sform()), zooms[2]): return img.get_sform()
    elif np.isclose(affine_pixdim(img.get_qform()), zooms[2]): return img.get_qform()

affine_file = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/MB-16/LGE/SA'
img_file = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/MB-16/LGE/SA_seg_itk'
affine_ext = '.nii'
img_ext = '.nii.gz'

affine = nib.load(affine_file + affine_ext)
aff = get_correct_affine(affine)

img = nib.load(img_file + img_ext)
seg_data = img.get_fdata()
new_img = nib.Nifti1Image(seg_data, aff)
nib.save(new_img, img_file + '_fix' + img_ext)
