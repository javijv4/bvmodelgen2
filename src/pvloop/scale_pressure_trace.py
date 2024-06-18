#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 8 18:08:25 2020

@author: javijv4
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
#doi: 10.1093/eurheartj/ehs016
path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/'
patient = 'AV-19'
imgs_fldr = path + patient + '/Images/'

sa_dt = 0.02311
la_dt = 0.03417

patient_valve_frames = {'mvc': 0, 'avo': 3, 'avc': 10, 'mvo': 14, 'tcycle': 19}
ed_pressure = 8.0/7.50062   # kPa
es_pressure = 120.0/7.50062 # kPa

normalized_time, normalized_pressure = np.load('refdata/normalized_human_pressure.npy').T
normalized_valve_times = np.load('refdata/normalized_valve_times.npz')
events = ['mvc', 'avo', 'avc', 'mvo', 'tcycle']

volume = np.load(imgs_fldr + 'img_stats.npz')['lv_vol']

patient_valve_times = {key: value*la_dt for key, value in patient_valve_frames.items()}
print(patient_valve_times)
HR = 60/patient_valve_times['tcycle']

normalized_func = interp1d(normalized_time, normalized_pressure)


patient_pressure = []
patient_time = []

for i in range(1, len(events)):
    x = np.linspace(normalized_valve_times[events[i-1]],normalized_valve_times[events[i]],1001, endpoint=True)
    segment_pressure = normalized_func(x)
    segment_time = np.linspace(patient_valve_times[events[i-1]],patient_valve_times[events[i]],1001, endpoint=True)

    patient_pressure.append(segment_pressure[1:])
    patient_time.append(segment_time[1:])

patient_pressure = np.hstack(patient_pressure)
patient_time = np.hstack(patient_time)
patient_pressure = np.append(normalized_pressure[0], patient_pressure)
patient_time = np.append(0, patient_time)

# Rescale pressure
aux = patient_pressure.copy()
a = (es_pressure - ed_pressure)/(1.0 - aux[0])
patient_pressure[aux > aux[0]] = aux[aux > aux[0]]*a + es_pressure - a
patient_pressure[aux <= aux[0]] = aux[aux <= aux[0]]/aux[0]*ed_pressure


plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(2, 1, clear=True, figsize=(5,8), num=1)
axs[0].plot(normalized_time, normalized_pressure, 'k')
for key, value in normalized_valve_times.items():
    axs[0].axvline(x=value, color='r', linestyle='--')
axs[0].axhline(y=normalized_pressure[0], color='b', linestyle='--')
axs[0].axhline(y=1.0, color='b', linestyle='--')
axs[0].set_xlabel('Normalized time')
axs[0].set_ylabel('Normalized pressure')


axs[1].plot(patient_time, patient_pressure*7.50062, 'k')
for key, value in patient_valve_times.items():
    axs[1].axvline(x=value, color='r', linestyle='--')
axs[1].axhline(y=es_pressure*7.50062, color='b', linestyle='--')
axs[1].axhline(y=ed_pressure*7.50062, color='b', linestyle='--')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Pressure (mmHg)')

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)

plt.savefig('scaled_pressure.png', bbox_inches='tight', dpi=180)

#%%
vol_time = np.arange(len(volume))*sa_dt
print(len(volume))
print(vol_time[-1], patient_time[-1])
vol_func = interp1d(vol_time, volume, fill_value='extrapolate')
pres_func = interp1d(patient_time, patient_pressure, fill_value='extrapolate')

pres_time = np.arange(patient_valve_frames['tcycle'])*la_dt
patient_volume = vol_func(patient_time)
valve_times = np.array(list(patient_valve_times.values()))

plt.figure(2, clear=True)
plt.plot(patient_time, vol_func(patient_time))

plt.figure(3, clear=True)
plt.plot(vol_func(patient_time), patient_pressure)
plt.plot(vol_func(vol_time), pres_func(vol_time), 'o', label='SA timepoints')
plt.plot(vol_func(pres_time), pres_func(pres_time), 'o', label='LA timepoints')
plt.plot(vol_func(valve_times), pres_func(valve_times), 'x', label='valve timepoints?')
plt.legend(frameon=False)
plt.show()