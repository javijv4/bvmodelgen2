#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:26:30 2023

@author: Javiera Jilberto Vallejos
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ImageData import ImageData, compute_volume_from_contours

# Inputs
patient = 'ZS-11'
path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/Images/'
dump_folder = path + 'vol_calc/'
png_folder = path + 'pngs/'
clean = False  ## If true it will erase the dump_folder (If you do this, every time you run the code you'll need to manually select LA points to delete)

sa_labels = {'lv': 2., 'rv': 1., 'lvbp': 3.}
la_labels = {'lv': 2., 'rv': 3., 'lvbp': 1.}
segs = {'sa': path + 'SA_seg',
            'la_4ch': path + 'LA_4CH_seg'}

if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)
if not os.path.exists(png_folder):
    os.makedirs(png_folder)

lv_volume = []
lv_areas = []
lv_zscoord = []

rv_volume = []
rv_areas = []
rv_zscoord = []

for i in range(30):
    output_fldr = dump_folder + 'vol_frame{:d}/'.format(i)

    imgdata = ImageData(segs, sa_labels, la_labels, i, output_fldr, autoclean=True)
    imgdata.align_slices(translation_files_path=path + 'frame0/')
    imgdata.generate_contours(plot=False)

    vol, ar, zs = compute_volume_from_contours(imgdata.contours, which='lvendo', return_areas=True, use_la_for_extrapolation=False)
    lv_volume.append(vol)
    lv_areas.append(ar)
    lv_zscoord.append(zs)

    vol, ar, zs = compute_volume_from_contours(imgdata.contours, which='rvendo', return_areas=True, use_la_for_extrapolation=False)
    rv_volume.append(vol)
    rv_areas.append(ar)
    rv_zscoord.append(zs)

    print('Frame %i' % i + ' volume = %2.6f' % lv_volume[-1])


if clean:
    os.system('rm -r ' + dump_folder)

# Save volumes
np.savetxt(path + 'lv_volume.txt', lv_volume[:,None])
np.savetxt(path + 'rv_volume.txt', rv_volume[:,None])

# Save plots
plt.figure(1, clear=True)
for i in range(30):
    plt.plot(lv_zscoord[i], lv_areas[i], 'o-', color=plt.cm.inferno(i/30))
    plt.annotate(str(i), (lv_zscoord[i][-1], lv_areas[i][-1]), textcoords="offset points", xytext=(-5,-10), ha='center')
cmap = plt.cm.inferno
norm = plt.Normalize(0, 30)
plt.xlabel('Long Axis Coordinate')
plt.ylabel('Area')
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Frame', ax=plt.gca())
plt.savefig(png_folder + 'lv_area_vs_zcoord.png', dppi=180, bbox_inches='tight')

plt.figure(2, clear=True)
for i in range(30):
    plt.plot(rv_zscoord[i], rv_areas[i], 'o-', color=plt.cm.inferno(i/30))
    plt.annotate(str(i), (rv_zscoord[i][-1], rv_areas[i][-1]), textcoords="offset points", xytext=(-5,-10), ha='center')
cmap = plt.cm.inferno
norm = plt.Normalize(0, 30)
plt.xlabel('Long Axis Coordinate')
plt.ylabel('Area')
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Frame', ax=plt.gca())
plt.savefig(png_folder + 'rv_area_vs_zcoord.png', dppi=180, bbox_inches='tight')

plt.figure(3, clear=True)
plt.plot(range(30), lv_volume, 'o-', label='lv')
plt.plot(range(30), rv_volume, 'o-', label='rv')
plt.xlabel('Frame')
plt.ylabel('Volume [mm3]')
plt.legend()
plt.grid(True)
plt.savefig(png_folder + 'volume.png', dppi=180, bbox_inches='tight')
plt.show()

print('LV stroke volume = %2.6f' % (np.max(lv_volume) - np.min(lv_volume)))
print('RV stroke volume = %2.6f' % (np.max(rv_volume) - np.min(rv_volume)))
print('Initial LV volume = %2.6f' % lv_volume[0] + ', final LV volume = %2.6f' % lv_volume[-1])
print('Initial RV volume = %2.6f' % rv_volume[0] + ', final RV volume = %2.6f' % rv_volume[-1])