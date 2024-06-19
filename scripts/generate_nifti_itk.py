#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:43:35 2020

@author: javijv4
"""

import nibabel as nib
import numpy as np

img_file = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/MB-16/LGE/SA'
ext = '.nii'

img = nib.load(img_file + ext)
data = img.get_fdata()
new_img = nib.Nifti1Image(data, np.eye(4))
nib.save(new_img, img_file + '_itk' +  ext)