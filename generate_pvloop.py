#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 8 18:08:25 2020

@author: javijv4
"""

import os
from matplotlib import pyplot as plt
import numpy as np
import cheartio as chio
from PatientData import PatientData

path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/'
patient = 'ZS-11'
imgs_fldr = path + patient + '/Images/'
data_fldr = path + patient + '/es_data_ms25/'
pngs_fldr = path + patient + '/pngs/'

if not os.path.exists(data_fldr):
    os.makedirs(data_fldr)
if not os.path.exists(pngs_fldr):
    os.makedirs(pngs_fldr)

lv_valve_frames = {'mvc': 0, 'avo': 3, 'avc': 11, 'mvo': 14, 'tcycle': 30}
rv_valve_frames = {'tvc': 0, 'pvo': 3, 'pvc': 10, 'tvo': 13, 'tcycle': 30}

volume_files = (imgs_fldr + 'lv_volume.txt', imgs_fldr + 'rv_volume.txt')

lv_ed_pressure = 8.0/7.50062   # kPa
lv_es_pressure = 130.0/7.50062 # kPa
rv_ed_pressure = 4.0/7.50062   # kPa
rv_es_pressure = 20.0/7.50062  # kPa

# Load data
pdata = PatientData(imgs_fldr + 'SA.nii', {'la_4ch': imgs_fldr + 'LA_4CH.nii'}, volume_files,
                    lv_valve_frames=lv_valve_frames, rv_valve_frames=rv_valve_frames, lv_pressures=(lv_ed_pressure, lv_es_pressure),
                    rv_pressures=(rv_ed_pressure, rv_es_pressure))


lv_vol, lv_pres, lv_vtimes = pdata.get_lv_pv_loop(0, {'avo': 0, 'avc': 0, 'mvo': 0})
rv_vol, rv_pres, rv_vtimes = pdata.get_rv_pv_loop(0, {'pvo': 0, 'pvc': 0, 'tvo': 0})

pdata.plot_volume_pressure_traces(None, lv_vol, lv_pres, rv_vol, rv_pres, lv_vtimes, rv_vtimes)
plt.savefig(pngs_fldr + 'pv_trace_raw.png', dpi=180, bbox_inches='tight')

pdata.plot_pv_loop(None, lv_vol, lv_pres, rv_vol, rv_pres, lv_vtimes, rv_vtimes)
plt.savefig(pngs_fldr + 'pv_loop_raw.png', dpi=180, bbox_inches='tight')

# Optimize PV loop
sol = pdata.optimize_pv_area(side='lv')
lv_shift, avo, avc, mvo = sol.x
lv_vol, lv_pres, lv_vtimes = pdata.get_lv_pv_loop(lv_shift, {'avo': avo, 'avc': avc, 'mvo': mvo})

sol = pdata.optimize_pv_area(side='rv')
rv_shift, pvo, pvc, tvo = sol.x
rv_vol, rv_pres, rv_vtimes = pdata.get_rv_pv_loop(rv_shift, {'pvo': pvo, 'pvc': pvc, 'tvo': tvo})

pdata.plot_volume_pressure_traces(None, lv_vol, lv_pres, rv_vol, rv_pres, lv_vtimes, rv_vtimes)
plt.savefig(pngs_fldr + 'pv_trace_opt.png', dpi=180, bbox_inches='tight')

pdata.plot_pv_loop(None, lv_vol, lv_pres, rv_vol, rv_pres, lv_vtimes, rv_vtimes)
plt.savefig(pngs_fldr + 'pv_loop_opt.png', dpi=180, bbox_inches='tight')

# Correct using klotz
lv_pres = pdata.correct_pv_using_klotz(lv_vol, lv_pres)
rv_pres = pdata.correct_pv_using_klotz(rv_vol, rv_pres)

# Correct by mesh
lv_ed_volume = 208840.70996492467   # in mm^3 from the mesh
rv_ed_volume = 196409.60738376155   # in mm^3 from the mesh

lv_es_volume = 82889.047255798709   # in mm^3 from the mesh
rv_es_volume = 92392.942933852479   # in mm^3 from the mesh

print('LV EF = %2.6f' % ((lv_ed_volume - lv_es_volume)/lv_ed_volume))
print('RV EF = %2.6f' % ((rv_ed_volume - rv_es_volume)/rv_ed_volume))

lv_ef = (np.max(lv_vol) - np.min(lv_vol))/np.max(lv_vol)
rv_ef = (np.max(rv_vol) - np.min(rv_vol))/np.max(rv_vol)

norm_vol = (lv_vol - np.min(lv_vol))/(lv_vol[0] - np.min(lv_vol))

lv_edv = lv_vol[0]
lv_esv = np.min(lv_vol)

new_lv_vol = norm_vol*(lv_ed_volume-lv_es_volume) + lv_es_volume
lv_vol = new_lv_vol

lv_ef = (np.max(new_lv_vol) - np.min(new_lv_vol))/np.max(new_lv_vol)
lv_sv = np.max(new_lv_vol) - np.min(new_lv_vol)

norm_vol = (rv_vol - np.min(rv_vol))/(rv_vol[0] - np.min(rv_vol))
new_esv = rv_ed_volume - lv_sv

new_rv_vol = norm_vol*(rv_ed_volume-new_esv) + new_esv
rv_vol = new_rv_vol

rv_ef = (np.max(new_rv_vol) - np.min(new_rv_vol))/np.max(new_rv_vol)
rv_sv = np.max(new_rv_vol) - np.min(new_rv_vol)

print('LV EF = %2.6f' % (lv_ef))
print('RV EF = %2.6f' % (rv_ef))
print(np.min(new_lv_vol), np.max(new_lv_vol), new_lv_vol[0])
print(np.min(new_rv_vol), np.max(new_rv_vol), new_rv_vol[0])
print('LV SV = %2.6f' % (lv_sv))
print('RV SV = %2.6f' % (rv_sv))

# Save PV loop
save = np.vstack([pdata.time/1000, new_lv_vol]).T
chio.write_dfile(data_fldr + 'lv_volume.INIT', save)
save = np.vstack([pdata.time/1000, lv_pres]).T
chio.write_dfile(data_fldr + 'lv_pressure.INIT', save)

save = np.vstack([pdata.time/1000, new_rv_vol]).T
chio.write_dfile(data_fldr + 'rv_volume.INIT', save)
save = np.vstack([pdata.time/1000, rv_pres]).T
chio.write_dfile(data_fldr + 'rv_pressure.INIT', save)


# Save valve times
chio.write_ch_dictionary(data_fldr + 'lv_valve_times.INIT', lv_vtimes)
lv_vframes = {k: np.round(v/pdata.la_dt).astype(int) for k, v in lv_vtimes.items()}
chio.write_ch_dictionary(data_fldr + 'lv_valve_frames.INIT', lv_vframes)
chio.write_ch_dictionary(data_fldr + 'rv_valve_times.INIT', rv_vtimes)
rv_vframes = {k: np.round(v/pdata.la_dt).astype(int) for k, v in rv_vtimes.items()}
chio.write_ch_dictionary(data_fldr + 'rv_valve_frames.INIT', rv_vframes)

# Plots
pdata.plot_volume_pressure_traces(None, lv_vol, lv_pres, rv_vol, rv_pres, lv_vtimes, rv_vtimes)
plt.savefig(pngs_fldr + 'pv_trace.png', dpi=180, bbox_inches='tight')

pdata.plot_pv_loop(None, lv_vol, lv_pres, rv_vol, rv_pres, lv_vtimes, rv_vtimes, add_klotz=False)
plt.savefig(pngs_fldr + 'pv_loop.png', dpi=180, bbox_inches='tight')
