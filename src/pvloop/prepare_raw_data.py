#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 8 18:08:25 2020

@author: javijv4
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

normalized_pressure = loadmat('refdata/normalized_human_pressure.mat')['vv'].flatten()
normalized_time = np.linspace(0,1,1001, endpoint=True)
normalized_valve_times = {'mvc': 0.155, 'tvc': 0.155,
                          'avo': 0.277, 'pvo': 0.277,
                          'avc': 0.612, 'pvc': 0.612,
                          'mvo': 0.755, 'tvo': 0.755,
                          'dias': 0.980,
                          'tcycle': 1+0.155}


plt.figure(1, clear=True)
plt.plot(normalized_time, normalized_pressure, 'k')
for key, value in normalized_valve_times.items():
    plt.axvline(x=value, color='r', linestyle='--')

plt.xlim([0,1])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('Normalised time')
plt.ylabel('Normalised pressure')
# plt.ylim([0,0.1])
plt.savefig('normalised_human_pressure_raw.png')

ind = np.where(normalized_time > normalized_valve_times['mvc'])[0][0]
aux = normalized_pressure.copy()
normalized_pressure[0:-ind] = aux[ind:]
normalized_pressure[-ind:] = aux[:ind]

normalized_pressure_raw = normalized_pressure.copy()
normalized_time_raw = normalized_time.copy()

ind_min = np.argmin(normalized_pressure[1:]) + 1
diff = np.diff(normalized_pressure[ind_min:])
diff = np.append(diff, 0)
diff[diff<0] = 0
normalized_pressure[ind_min:] = np.cumsum(diff) + normalized_pressure[ind_min]


normalized_pressure = np.concatenate([normalized_pressure[1:]]*10)
normalized_pressure = np.append(0, normalized_pressure)
normalized_time = np.linspace(0,1,10001, endpoint=True)


normalized_pressure = gaussian_filter1d(normalized_pressure, 20)

normalized_pressure = normalized_pressure/np.max(normalized_pressure)
normalized_pressure = normalized_pressure[8000:9001]
normalized_time = np.linspace(0,1,1001, endpoint=True)
normalized_valve_times = {key: value-normalized_valve_times['mvc'] for key, value in normalized_valve_times.items()}

save = np.vstack([normalized_time, normalized_pressure]).T
np.save('refdata/normalized_human_pressure.npy', save)
np.savez('refdata/normalized_valve_times.npz', **normalized_valve_times)


plt.figure(1, clear=True)
plt.plot(normalized_time_raw, normalized_pressure_raw, '0.5')
plt.plot(normalized_time, normalized_pressure, 'k')
for key, value in normalized_valve_times.items():
    plt.axvline(x=value, color='r', linestyle='--')

plt.xlim([0,1])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('Normalised time')
plt.ylabel('Normalised pressure')
# plt.ylim([0,0.1])
plt.savefig('normalised_human_pressure_raw.png')