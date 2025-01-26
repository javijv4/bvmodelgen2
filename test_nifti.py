#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/01/26 16:36:14

@author: Javiera Jilberto Vallejos 
'''
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from niftiutils import readFromNIFTI1, readFromNIFTI2
from masksutils import check_seg_valid
from masks2contours.m2c import getContoursFromMask
import masks2contours.utils as ut

path = '/home/jilberto/University of Michigan Dropbox/Javiera Jilberto Vallejos/Projects/Desmoplakin/Models/DSPPatients/AB-17/Images/'
seg_path = path + 'SA_seg'
labels = {'lv': 2., 'rv': 1., 'lvbp': 3.}
output_fldr = 'check/'
view = 'sa'

# Read data
data, affine, _, header = readFromNIFTI2(seg_path, 0, correct_ras=False, is_seg=True)

# Check if the segmentation is valid
new_data, isvalid = check_seg_valid(view, data, labels, autoclean=True)
# new_data = new_data.astype(np.uint16)

# Write seg
# writeNIFTI(output_fldr + view.upper() + '.nii.gz', new_data, affine)
new_seg = nib.Nifti1Image(new_data, affine)
nib.save(new_seg, output_fldr + view.upper() + '.nii.gz')
seg_file = output_fldr + view.upper() + '.nii.gz'

# Read it back
data, transform, pixspacing, _ = readFromNIFTI2(seg_file, 0, is_seg=True)

# Extract slices
n=2
seg = data[:,:,n]

im=plt.imshow(seg)
plt.colorbar(im)
plt.show()

# Get contour
labels = {'lv': 2., 'rv': 3., 'lvbp': 1.}
LVendo = np.isclose(seg, labels['lvbp'])
LVepi = np.isclose(seg, labels['lv'])
if not np.all(~LVepi):
    LVepi += LVendo
RVendo = np.isclose(seg, labels['rv'])

plt.imshow(LVendo)
plt.show()
plt.imshow(LVepi)
plt.show()

# Get contours
LVendoCS = getContoursFromMask(LVendo, irregMaxSize = 20)
LVepiCS = getContoursFromMask(LVepi, irregMaxSize = 20)
RVendoCS = getContoursFromMask(RVendo, irregMaxSize = 20)

is_2chr = False
if (len(LVendoCS) == 0) and (len(LVepiCS) == 0) and (len(RVendoCS) > 0):    # 2CHr, only RV present
    is_2chr = True

# Check that LVepi and LVendo do not share any points (in SA)
if 'sa' in view:
    [dup, _, _] = ut.sharedRows(LVepiCS, LVendoCS)
    if len(dup) > 0:  # If they share rows, the slice is not valid
        print('Shared rows!!!')
    else:
        print('No shared rows')