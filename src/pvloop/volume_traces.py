#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 22:30:00 2024

@author: jjv
"""
from matplotlib import pyplot as plt
import numpy as np
import meshio as io
import cheartio as chio
import nibabel as nib

path = '/Users/jjv/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/'
patient = 'AB-17'

patients = ['VB-1', 'AV-19', 'BI-18', 'CA-15', 'ZS-11', 'TS-9', 'KL-4', 'KL-5', 'AB-17', 'JL-3', 'JN-8', 'DM-23', 'AS-10']
patients = ['AV-19']

method = 'trapz'

for patient in patients:
    imgs_fldr = path + patient + '/Images/'

    rv = 1
    lv = 3

    try:
        sa_seg = nib.load(imgs_fldr + 'SA_seg.nii')
    except:
        sa_seg = nib.load(imgs_fldr + 'SA_seg.nii.gz')
    data = sa_seg.get_fdata().astype(int)

    timepoints = data.shape[-1]
    nslices = data.shape[-2]
    voxel_dim = sa_seg.header.get_zooms()
    if method == 'voxvol':
        voxel_vol = voxel_dim[0]*voxel_dim[1]*voxel_dim[2]
    elif method == 'trapz':
        voxel_vol = voxel_dim[0]*voxel_dim[1]

    mask = data > 0
    zmask = np.sum(np.sum(mask, axis=0), axis=0)
    zmask = np.all(zmask>0, axis=1)

    data[:,:,~zmask] = 0

    lv_mask = data == lv
    rv_mask = data == rv

    lv_mask = lv_mask.reshape([-1,nslices,timepoints])
    rv_mask = rv_mask.reshape([-1,nslices,timepoints])

    z_coord = np.arange(nslices)*voxel_dim[2]

    lv_vol = np.sum(lv_mask, axis=0)*voxel_vol
    rv_vol = np.sum(rv_mask, axis=0)*voxel_vol

    if method == 'trapz':
        lv_vol = np.trapz(lv_vol, z_coord, axis=0)
        rv_vol = np.trapz(rv_vol, z_coord, axis=0)

    plt.figure(1, clear=True)
    plt.plot(lv_vol)
    plt.plot(rv_vol)
    plt.savefig('check.png', bbox_inches='tight')

    lv_es_volume = np.min(lv_vol)
    rv_es_volume = np.min(rv_vol)
    lv_ed_volume = np.max(lv_vol)
    rv_ed_volume = np.max(rv_vol)

    lv_sv = lv_ed_volume - lv_es_volume
    rv_sv = rv_ed_volume - rv_es_volume

    lv_ef = lv_sv/lv_ed_volume
    rv_ef = rv_sv/rv_ed_volume
    print(patient, lv_sv, rv_sv, lv_ef)

    np.savez(imgs_fldr + 'img_stats.npz', lv_ed_volume=lv_ed_volume, lv_es_volume=lv_es_volume,
            rv_ed_volume=rv_ed_volume, rv_es_volume=rv_es_volume, lv_sv=lv_sv, rv_sv=rv_sv,
            lv_ef=lv_ef, rv_ef=rv_ef, lv_vol=lv_vol, rv_vol=rv_vol)