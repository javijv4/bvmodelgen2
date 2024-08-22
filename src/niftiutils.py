#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:41:24 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import nibabel as nib

def readFromNIFTI(segName, frameNum, correct_ras=True):
    ''' Helper function used by masks2ContoursSA() and masks2ContoursLA(). Returns (seg, transform, pixSpacing). '''
    # Load NIFTI image and its header.
    if os.path.isfile(segName):
        ext = ''
    elif os.path.isfile(segName + '.nii.gz'):
        ext = '.nii.gz'
    elif os.path.isfile(segName + '.nii'):
        ext = '.nii'
    else:
        raise FileNotFoundError('File {} was not found'.format(segName))

    img = nib.load(segName + ext)
    hdr = img.header

    # Get the segmentation from the NIFTI file.
    seg = img.get_fdata()
    if seg.ndim > 3:  # if segmentation includes all time points
        seg = seg[:, :, :, frameNum]

    # Get the 4x4 homogeneous affine matrix.
    transform = img.affine  # In the MATLAB script, we transposed the transform matrix at this step. We do not need to do this here due to how nibabel works.
    if correct_ras:
        transform[0:2, :] = -transform[0:2, :] # This edit has to do with RAS system in Nifti
    # Initialize pixSpacing. In MATLAB, pix_spacing is info.PixelDimensions(1). After converting from 1-based
    # indexing to 0-based indexing, one might think that that means pixSpacing should be pixdim[0], but this is not the
    # case due to how nibabel stores NIFTI headers.
    pixdim = hdr.structarr["pixdim"][1:3 + 1]
    pixSpacing = pixdim[1]

    return (seg, transform, pixdim, hdr)



# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import math
import os
import pickle
import warnings
from collections import defaultdict
from itertools import product, starmap
from pathlib import PurePath
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib

def correct_nifti_header_if_necessary(img_nii):
    """
    Check nifti object header's format, update the header if needed.
    In the updated image pixdim matches the affine.
    Args:
        img_nii: nifti image object
    """
    dim = img_nii.header["dim"][0]
    if dim >= 5:
        return img_nii  # do nothing for high-dimensional array
    # check that affine matches zooms
    pixdim = np.asarray(img_nii.header.get_zooms())[:dim]
    norm_affine = np.sqrt(np.sum(np.square(img_nii.affine[:dim, :dim]), 0))
    if np.allclose(pixdim, norm_affine):
        return img_nii
    if hasattr(img_nii, "get_sform"):
        return rectify_header_sform_qform(img_nii)
    return img_nii


def rectify_header_sform_qform(img_nii):
    """
    Look at the sform and qform of the nifti object and correct it if any
    incompatibilities with pixel dimensions
    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/io/misc_io.py
    Args:
        img_nii: nifti image object
    """
    d = img_nii.header["dim"][0]
    pixdim = np.asarray(img_nii.header.get_zooms())[:d]
    sform, qform = img_nii.get_sform(), img_nii.get_qform()
    norm_sform = np.sqrt(np.sum(np.square(sform[:d, :d]), 0))
    norm_qform = np.sqrt(np.sum(np.square(qform[:d, :d]), 0))
    sform_mismatch = not np.allclose(norm_sform, pixdim)
    qform_mismatch = not np.allclose(norm_qform, pixdim)

    if img_nii.header["sform_code"] != 0:
        if not sform_mismatch:
            return img_nii
        if not qform_mismatch:
            img_nii.set_sform(img_nii.get_qform())
            return img_nii
    if img_nii.header["qform_code"] != 0:
        if not qform_mismatch:
            return img_nii
        if not sform_mismatch:
            img_nii.set_qform(img_nii.get_sform())
            return img_nii

    norm = np.sqrt(np.sum(np.square(img_nii.affine[:d, :d]), 0))
    warnings.warn(f"Modifying image pixdim from {pixdim} to {norm}")

    img_nii.header.set_zooms(norm)
    return img_nii