#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 22:30:00 2024

@author: jjv
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import meshio as io
import cheartio as chio
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/'

patients = ['AB-17', 'AS-10', 'AV-19', 'BI-18', 'CA-15', 'CW-21', 'DM-23', 'JL-3', 'JN-8', 'KL-4', 'KL-5', 'KR-13', 'MB-16', 'SL-16', 'TS-9', 'VB-1', 'ZS-11']
# patients = ['AV-19']

method = 'trapz'

rv = 1
lv = 3

for patient in patients:
    imgs_fldr = path + patient + '/Images/'
    pngs_fldr = path + patient + '/pngs/'
    if not os.path.exists(pngs_fldr):
        os.makedirs(pngs_fldr)

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

    # data[:,:,~zmask] = 0

    lv_mask = data == lv
    rv_mask = data == rv

    lv_mask = lv_mask.reshape([-1,nslices,timepoints])
    rv_mask = rv_mask.reshape([-1,nslices,timepoints])

    z_coord = np.arange(nslices)*voxel_dim[2]

    lv_vol = np.sum(lv_mask, axis=0)*voxel_vol
    rv_vol = np.sum(rv_mask, axis=0)*voxel_vol

    if method == 'trapz':
        lv_volume = np.trapz(lv_vol, z_coord, axis=0)
        rv_volume = np.trapz(rv_vol, z_coord, axis=0)

        # # Check where is the apex
        # sum_vol = np.sum(lv_vol, axis=1)
        # ind1, ind2 = np.where(sum_vol>0)[0][np.array([0,-1])]
        # if sum_vol[ind1] > sum_vol[ind2]: # put apex first
        #     lv_vol = np.flipud(lv_vol)
        #     rv_vol = np.flipud(rv_vol)

        # # Get the z coordinates of the base
        # zlims = np.zeros([timepoints, 2], dtype=int)
        # for i in range(timepoints):
        #     zlims[i] = np.where(lv_vol[:,i]>0)[0][np.array([0,-1])]
        # zlims[:,0] -= 1

        # lv_volume = np.zeros(timepoints)
        # rv_volume = np.zeros(timepoints)

        # for i in range(timepoints):
        #     lv_volume[i] = np.trapz(z_coord[zlims[i,0]:zlims[i,1]], lv_vol[zlims[i,0]:zlims[i,1],i])


        # # Get the z coordinates of the base
        # zlims = np.zeros([timepoints, 2], dtype=int)
        # for i in range(timepoints):
        #     zlims[i] = np.where(rv_vol[:,i]>0)[0][np.array([0,-1])]
        # zlims[:,0] -= 1

        # for i in range(timepoints):
        #     rv_volume[i] = np.trapz(z_coord[zlims[i,0]:zlims[i,1]], rv_vol[zlims[i,0]:zlims[i,1],i])
        #     plt.plot(z_coord[zlims[i,0]:zlims[i,1]], rv_vol[zlims[i,0]:zlims[i,1],i])


    elif method == 'trapz_test':

        # Check where is the apex
        sum_vol = np.sum(lv_vol, axis=1)
        ind1, ind2 = np.where(sum_vol>0)[0][np.array([0,-1])]
        if sum_vol[ind1] > sum_vol[ind2]: # put apex first
            lv_vol = np.flipud(lv_vol)


        # Get the z coordinates of the base
        zlims = np.zeros([timepoints, 2])
        for i in range(timepoints):
            zlims[i] = np.where(lv_vol[:,i]>0)[0][np.array([0,-1])]

        maxz = np.max(zlims[:,1])*voxel_dim[2]
        minz = np.min(zlims[:,1])*voxel_dim[2]

        base = np.tile(zlims[:,1],5)
        base_smooth = gaussian_filter(base, 3)
        base_z = base_smooth[timepoints*2:timepoints*3]*voxel_dim[2]
        base_z = (base_z - np.min(base_z))/(np.max(base_z)-np.min(base_z))*(maxz-minz) + minz

        lv_volume = np.zeros(timepoints)
        rv_volume = np.zeros(timepoints)

        for i in range(timepoints):
            ind1, ind2 = np.where(lv_vol[:,i]>0)[0][np.array([0,-1])]
            max_ind = np.argmax(lv_vol[:,i])
            area_func = interp1d(z_coord[ind1:ind2],lv_vol[ind1:ind2,i], fill_value='extrapolate', kind='linear')
            area_func_inv = interp1d(lv_vol[ind1:max_ind,i], z_coord[ind1:max_ind], fill_value='extrapolate', kind='linear')

            apex_z = np.max([z_coord[ind1-1], area_func_inv(0)])
            zs = np.append(apex_z, z_coord[ind1:ind2])

            vol_z = np.append(0, lv_vol[ind1:ind2,i])

            area_func = interp1d(zs, vol_z, fill_value='extrapolate', kind='linear')
            zs = np.append(zs, base_z[i])

            lv_volume[i] = np.trapz(zs, area_func(zs))

            ind1, ind2 = np.where(rv_vol[:,i]>0)[0][np.array([0,-1])]
            if rv_vol[ind1,i] < rv_vol[ind2,i]:
                ind1 -= 1
            else:
                ind2 += 1
            rv_volume[i] = np.trapz(rv_vol[ind1:ind2+1,i], z_coord[ind1:ind2+1])

            # plt.plot(rv_volume, label='RV')

    elif method == 'voxvol':
        lv_volume = lv_vol
        rv_volume = rv_vol


    lv_es_volume = np.min(lv_volume)
    rv_es_volume = np.min(rv_volume)
    lv_ed_volume = np.max(lv_volume)
    rv_ed_volume = np.max(rv_volume)

    lv_sv = lv_ed_volume - lv_es_volume
    rv_sv = rv_ed_volume - rv_es_volume

    lv_ef = lv_sv/lv_ed_volume
    rv_ef = rv_sv/rv_ed_volume

    print(patient, lv_ef)

    np.savez(imgs_fldr + 'img_stats.npz', lv_ed_volume=lv_ed_volume, lv_es_volume=lv_es_volume,
            rv_ed_volume=rv_ed_volume, rv_es_volume=rv_es_volume, lv_sv=lv_sv, rv_sv=rv_sv,
            lv_ef=lv_ef, rv_ef=rv_ef)


    plt.figure(1, clear=True)
    plt.plot(lv_volume/1000, label='LV')
    plt.plot(rv_volume/1000, label='RV')
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('Volume (mL)')
    plt.annotate('LV EF = {:.2f}'.format(lv_ef), (0.5,0.9), xycoords='axes fraction')
    plt.annotate('RV EF = {:.2f}'.format(rv_ef), (0.5,0.8), xycoords='axes fraction')
    plt.savefig(pngs_fldr + 'volume_traces.png')