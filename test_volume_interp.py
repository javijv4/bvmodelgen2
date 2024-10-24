#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/09/26 14:11:24

@author: Javiera Jilberto Vallejos 
'''
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

def clean_volume_trace(vol_trace):
    npoints = len(vol_trace)
    vol_trace = np.tile(vol_trace, 3)
    voi = np.arange(0, len(vol_trace), 1) - npoints

    # Find peaks
    diastole_peaks, _ = find_peaks(vol_trace, distance=len(vol_trace)/4)
    diastole_peaks = np.append(0, diastole_peaks)
    systole_peaks, _ = find_peaks(-vol_trace, distance=len(vol_trace)/4)

    # Find when the heart is contracting
    arr = np.arange(0, len(vol_trace), 1)
    contraction_mask = np.zeros(len(vol_trace), dtype=bool)
    for i in range(len(diastole_peaks)):
        contraction_mask[(arr > diastole_peaks[i]) & (arr <= systole_peaks[i])] = True

    # Find mid cycle mask
    mid_cycle_mask = np.zeros(len(vol_trace), dtype=bool)
    mid_cycle_mask[npoints:2*npoints] = True
    print(np.where(mid_cycle_mask)) 

    len_bad_points = 1
    while len_bad_points > 0:
        # Find where contraction ends
        diff_vol = np.diff(vol_trace)
        diff_vol = np.insert(diff_vol, 0, 0)

        cycle = 1   # We only work with the center
        contraction_points = np.where(contraction_mask*mid_cycle_mask)[0]
        relaxation_points = np.where(~contraction_mask*mid_cycle_mask)[0]
        print(diff_vol[contraction_points])

        # Find points where the volume increases during contraction or decreases during relaxation
        bad_points = diff_vol[contraction_points] > 0
        bad_points = np.where(np.append(bad_points, diff_vol[relaxation_points] < 0))[0] - 1 + cycle*npoints
            
        voi_clean = np.delete(voi, bad_points)
        vol_trace_clean = np.delete(vol_trace, bad_points)
        mid_cycle_mask = np.delete(mid_cycle_mask, bad_points)
        contraction_mask = np.delete(contraction_mask, bad_points)

        len_bad_points = len(bad_points)

        voi = voi_clean
        vol_trace = vol_trace_clean

        print(bad_points)

    return voi_clean, vol_trace

# Inputs
patient = 'ZS-11'
path = '/Users/jjv/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/Images/'
lv_vol = np.loadtxt(path + 'lv_volume_raw.txt')


voi = np.arange(0, len(lv_vol), 1)
lv_vol_interp = PchipInterpolator(voi, lv_vol)

voi_clean, lv_vol_clean = clean_volume_trace(lv_vol)
lv_vol_clean_interp = PchipInterpolator(voi_clean, lv_vol_clean)

x = np.linspace(0, len(lv_vol), 1000)
plt.figure(1, clear=True)
plt.plot(lv_vol, 'o')
# plt.plot(arr[contraction_mask], lv_vol[contraction_mask==1], 'o')
# plt.plot(diastole_peaks, lv_vol[diastole_peaks], 'x')
# plt.plot(systole_peaks, lv_vol[systole_peaks], 'x')
# plt.plot(voi_clean, lv_vol_clean, 'o')
plt.plot(x, lv_vol_interp(x), '-')
plt.plot(x, lv_vol_clean_interp(x), '-')
plt.show()
