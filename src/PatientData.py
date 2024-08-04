#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 8 18:08:25 2020

@author: javijv4
"""

import numpy as np
from scipy.interpolate import interp1d
import nibabel as nib
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy.spatial import KDTree
from masks2contours import m2c, slicealign

class PatientData:
    def __init__(self, sa_img_path, la_img_paths, volume_files,
                 lv_valve_frames=None, rv_valve_frames=None, lv_pressures=None, rv_pressures=None):

        # Load SA data
        img = nib.load(sa_img_path)
        self.sa_dt = img.header['pixdim'][4]
        sa_cycle_time = img.header['pixdim'][4]*img.header['dim'][4]
        self.sa_img = img
        self.cycle_time = sa_cycle_time

        # Get volume traces
        sa_lv_vol = np.loadtxt(volume_files[0])
        sa_rv_vol = np.loadtxt(volume_files[1])
        self.lv_volume = sa_lv_vol
        self.rv_volume = sa_rv_vol
        self.volume_time = np.arange(len(self.lv_volume))*self.sa_dt

        # Load LA data
        self.la_imgs, self.la_dt = self.load_la_imgs(la_img_paths)

        # Compute valve times and HR
        self.lv_valve_frames = lv_valve_frames
        if self.la_dt is None:
            raise ValueError('la_img_paths must be provided alongside lv_valve_frames to compute valve times')
        self.lv_valve_times = {key: value*self.la_dt for key, value in lv_valve_frames.items()}
        self.rv_valve_frames = rv_valve_frames
        self.rv_valve_times = {key: value*self.la_dt for key, value in rv_valve_frames.items()}

        # Loading reference pressure data # TODO put everything in a single file and pass it as argument
        self.normalized_time, self.normalized_pressure = np.load('src/pvloop/refdata/normalized_human_pressure.npy').T
        self.normalized_func = interp1d(self.normalized_time, self.normalized_pressure)
        self.normalized_valve_times = np.load('src/pvloop/refdata/normalized_valve_times.npz')

        self.lv_events = ['mvc', 'avo', 'avc', 'mvo', 'tcycle']
        self.rv_events = ['tvc', 'pvo', 'pvc', 'tvo', 'tcycle']

        # Load pressure data
        if lv_pressures is None:
            self.lv_ed_pressure = -1
            self.lv_es_pressure = -1
        else:
            self.lv_ed_pressure = lv_pressures[0]
            self.lv_es_pressure = lv_pressures[1]
        if rv_pressures is None:
            self.rv_ed_pressure = -1
            self.rv_es_pressure = -1
        else:
            self.rv_ed_pressure = rv_pressures[0]
            self.rv_es_pressure = rv_pressures[1]

        # Rescale pressure data
        if self.lv_valve_times != {}:
            self.lv_pressure, self.pressure_time = self.rescale_pressure_time(self.lv_valve_times, self.lv_events)

            if self.lv_ed_pressure == -1 or self.lv_es_pressure == -1:
                self.lv_pressure = None
            else:
                self.lv_pressure = self.rescale_pressure_magnitude(self.lv_pressure, self.lv_ed_pressure, self.lv_es_pressure)

            self.rv_pressure, _ = self.rescale_pressure_time(self.rv_valve_times, self.rv_events)
            if self.rv_ed_pressure == -1 or self.rv_es_pressure == -1:
                self.rv_pressure = None
            else:
                self.rv_pressure = self.rescale_pressure_magnitude(self.rv_pressure, self.rv_ed_pressure, self.rv_es_pressure)

        # Get time and interpolant functions
        if self.lv_pressure is not None:
            self.time, self.lv_vol_func, self.lv_pres_func, self.rv_vol_func, self.rv_pres_func = self.get_volume_pressure_funcs()
            self.HR = 60/self.time[-1]*1000


    def load_la_imgs(self, la_img_paths):
        #TODO check that all dts are the same
        la_imgs = {}
        for key, la_img_path in la_img_paths.items():
            la_imgs[key] = nib.load(la_img_path)
            la_dt = la_imgs[key].header['pixdim'][4]
        la_cycle_time = la_imgs[key].header['pixdim'][4]*la_imgs[key].header['dim'][4]

        self.cycle_time = np.max([self.cycle_time, la_cycle_time])
        self.la_dt = la_dt*self.cycle_time/la_cycle_time
        return la_imgs, self.la_dt

    @staticmethod
    def calculate_stroke_volume(volume):
        es_volume = np.min(volume)
        ed_volume = np.max(volume)
        stroke_volume = ed_volume - es_volume
        return stroke_volume

    def correct_volume_traces(self, lv_vol, rv_vol):
        # lv_sv = self.calculate_stroke_volume(lv_vol)
        # rv_sv = self.calculate_stroke_volume(rv_vol)

        # sv = np.max([lv_sv, rv_sv])

        mean_lv_vol = np.mean(lv_vol)
        mean_rv_vol = np.mean(rv_vol)

        rv_vol = lv_vol + (mean_rv_vol - mean_lv_vol)

        return lv_vol, rv_vol


    def get_volume_traces(self, sa_seg_path, sa_labels, correct=True, method='trapz'):
        sa_seg = nib.load(sa_seg_path)
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

        lv_mask = data == sa_labels['lv']
        rv_mask = data == sa_labels['rv']

        lv_mask = lv_mask.reshape([-1,nslices,timepoints])
        rv_mask = rv_mask.reshape([-1,nslices,timepoints])

        z_coord = np.arange(nslices)*voxel_dim[2]

        lv_vol = np.sum(lv_mask, axis=0)*voxel_vol
        rv_vol = np.sum(rv_mask, axis=0)*voxel_vol

        if method == 'trapz':
            lv_vol = np.trapz(lv_vol, z_coord, axis=0)
            rv_vol = np.trapz(rv_vol, z_coord, axis=0)

        # Correct trace based on stroke volume
        if correct:
            lv_vol, rv_vol = self.correct_volume_traces(lv_vol, rv_vol)

        return lv_vol, rv_vol


    def rescale_pressure_time(self, valve_times, events):
        chamber_pressure = []
        pressure_time = []

        for i in range(1, len(events)):
            x = np.linspace(self.normalized_valve_times[events[i-1]],self.normalized_valve_times[events[i]],1001, endpoint=True)
            segment_pressure = self.normalized_func(x)
            segment_time = np.linspace(valve_times[events[i-1]],valve_times[events[i]],1001, endpoint=True)

            chamber_pressure.append(segment_pressure[1:])
            pressure_time.append(segment_time[1:])

        # Check when is diastasis
        self.dias_time = 0.2625*(segment_time[-1] - segment_time[0]) + segment_time[0]

        chamber_pressure = np.hstack(chamber_pressure)
        pressure_time = np.hstack(pressure_time)
        chamber_pressure = np.append(self.normalized_pressure[0], chamber_pressure)
        pressure_time = np.append(0, pressure_time)
        return chamber_pressure, pressure_time


    def rescale_pressure_magnitude(self, pressure, ed_pressure, es_pressure):
        # Calculate mean pressure
        mbp = 2/3*ed_pressure + 1/3*es_pressure
        sp = es_pressure
        dp = ed_pressure

        aux = pressure.copy()
        a = (sp - dp)/(1.0 - aux[0])
        pressure[aux > aux[0]] = aux[aux > aux[0]]*a + sp - a
        pressure[aux <= aux[0]] = aux[aux <= aux[0]]/aux[0]*dp
        return pressure


    def get_volume_pressure_funcs(self):
        # Repeat volume and pressure traces three times
        self.lv_volume = np.concatenate([self.lv_volume, self.lv_volume, self.lv_volume])
        self.rv_volume = np.concatenate([self.rv_volume, self.rv_volume, self.rv_volume])
        self.volume_time = np.arange(len(self.lv_volume))*self.sa_dt - self.volume_time[-1]

        self.pressure_time = np.concatenate([self.pressure_time - self.pressure_time[-1], self.pressure_time, self.pressure_time[-1] + self.pressure_time])
        self.lv_pressure = np.concatenate([self.lv_pressure, self.lv_pressure, self.lv_pressure])
        self.rv_pressure = np.concatenate([self.rv_pressure, self.rv_pressure, self.rv_pressure])

        lv_vol_func = interp1d(self.volume_time, self.lv_volume, fill_value='extrapolate')
        lv_pres_func = interp1d(self.pressure_time, self.lv_pressure, fill_value='extrapolate')

        rv_vol_func = interp1d(self.volume_time, self.rv_volume, fill_value='extrapolate')
        rv_pres_func = interp1d(self.pressure_time, self.rv_pressure, fill_value='extrapolate')

        time = np.linspace(0, self.cycle_time, 1001)
        return time, lv_vol_func, lv_pres_func, rv_vol_func, rv_pres_func


    def get_lv_pv_loop(self, vol_time_shift, valve_times_shift):
        vol_time_shift = vol_time_shift*self.lv_valve_times['tcycle']/10
        avo_time = self.lv_valve_times['avo'] - valve_times_shift['avo']*self.la_dt
        avc_time = self.lv_valve_times['avc'] - valve_times_shift['avc']*self.la_dt
        mvo_time = self.lv_valve_times['mvo'] - valve_times_shift['mvo']*self.la_dt

        valve_times = {'mvc': 0., 'avo': avo_time, 'avc': avc_time, 'mvo': mvo_time, 'tcycle': self.lv_valve_times['tcycle']}

        lv_pressure, time_pressure = self.rescale_pressure_time(valve_times, self.lv_events)
        lv_pressure = self.rescale_pressure_magnitude(lv_pressure, self.lv_ed_pressure, self.lv_es_pressure)
        lv_pres_func = interp1d(time_pressure, lv_pressure, fill_value='extrapolate')
        lv_pressure = lv_pres_func(self.time)
        lv_volume = self.lv_vol_func(self.time + vol_time_shift)

        return lv_volume, lv_pressure, valve_times

    def get_rv_pv_loop(self, vol_time_shift, valve_times_shift):
        vol_time_shift = vol_time_shift*self.rv_valve_times['tcycle']/10
        pvo_time = self.rv_valve_times['pvo'] - valve_times_shift['pvo']*self.la_dt
        pvc_time = self.rv_valve_times['pvc'] - valve_times_shift['pvc']*self.la_dt
        tvo_time = self.rv_valve_times['tvo'] - valve_times_shift['tvo']*self.la_dt

        valve_times = {'tvc': 0., 'pvo': pvo_time, 'pvc': pvc_time, 'tvo': tvo_time, 'tcycle': self.rv_valve_times['tcycle']}

        rv_pressure, time_pressure = self.rescale_pressure_time(valve_times, self.rv_events)
        rv_pressure = self.rescale_pressure_magnitude(rv_pressure, self.rv_ed_pressure, self.rv_es_pressure)
        rv_pres_func = interp1d(time_pressure, rv_pressure, fill_value='extrapolate')
        rv_pressure = rv_pres_func(self.time)
        rv_volume = self.rv_vol_func(self.time + vol_time_shift)

        return rv_volume, rv_pressure, valve_times


    @staticmethod
    def compute_pv_area(x_coords, y_coords):
        polygon = Polygon(zip(x_coords, y_coords))
        area = polygon.area
        return area


    def optimize_pv_area(self, side='lv'):
        bounds = Bounds([-1., 0., 0., 0.], [1., 1., 1., 1.])

        def func(x):
            vol_time_shift = x[0]
            if side == 'lv':
                valve_times_shift = {'avo': x[1], 'avc': x[2], 'mvo': x[3]}
                lv_vol, lv_pres, _ = self.get_lv_pv_loop(vol_time_shift, valve_times_shift)
                return -self.compute_pv_area(lv_vol, lv_pres)
            elif side == 'rv':
                valve_times_shift = {'pvo': x[1], 'pvc': x[2], 'tvo': x[3]}
                rv_vol, rv_pres, _ = self.get_rv_pv_loop(vol_time_shift, valve_times_shift)
                return -self.compute_pv_area(rv_vol, rv_pres)

        x0 = np.array([0., 0., 0., 0.])
        sol = minimize(func, x0, method='trust-constr', bounds=bounds)

        return sol


    def klotz_curve(self, lv_volume, lv_pressure, nv=200, return_a_b=False):
        # Tranform inputs
        Vm = lv_volume[0]/1000
        Pm = lv_pressure[0]*7.50062

        # Calculating curve
        V0 = klotz_V0(Vm, Pm)
        V30 = klotz_V30(V0, Vm, Pm)
        alpha, beta = klotz_ab(V30, Vm, Pm)

        V = np.linspace(np.min(lv_volume/1000), Vm, nv)
        P = alpha*V**beta

        # Return in mm3, kpa
        if return_a_b:
            return V*1000, P*0.133322, alpha, beta
        else:
            return V*1000, P*0.133322


    def correct_pv_using_klotz(self, lv_vol, lv_pres):
        _, _, alpha, beta = self.klotz_curve(lv_vol, lv_pres, return_a_b=True)

        pres_klotz = (alpha*(lv_vol/1000)**beta)*0.133322
        below_the_curve = np.where(lv_pres < pres_klotz)[0]
        if len(below_the_curve) == 0: return lv_pres
        st_ind = below_the_curve[0]

        if self.time[st_ind] > self.dias_time:
            # print('Correcting diastasis', st_ind, self.dias_time)
            # # Find closest point to diastasis
            dias_ind = np.where(self.time < self.dias_time)[0][-1]
            # ramp = np.linspace(0, 1, len(lv_pres[dias_ind:st_ind]))**2
            # ramp = ramp[::-1]

            # lv_pres[dias_ind:st_ind] = lv_pres[dias_ind:st_ind]*ramp + pres_klotz[dias_ind:st_ind]*(1-ramp)
            pres_error = np.abs(pres_klotz[dias_ind:st_ind]-lv_pres[dias_ind:st_ind])
            diff_pres = np.diff(pres_error)
            indexes =  np.where(diff_pres > 0)[0]
            if len(indexes) > 0:
                st_ind = indexes[0] + dias_ind

            # print(np.abs(pres_klotz[dias_ind:st_ind]-lv_pres[dias_ind:st_ind]))
            # st_ind = np.argmin(np.abs(pres_klotz[dias_ind:st_ind]-lv_pres[dias_ind:st_ind])) + dias_ind
            # print(st_ind)
            # pass

        lv_pres[st_ind:] = pres_klotz[st_ind:]
        return lv_pres



    def interpolate_sa_to_la(self, sa_seg_path):
        sa_time = np.arange(0, self.cycle_time, self.sa_dt)[:-1]
        la_time = np.arange(0, self.cycle_time, self.la_dt)[:-1]

        tree = KDTree(sa_time[:, np.newaxis])
        _, sa_frames = tree.query(la_time[:, np.newaxis])

        data = self.sa_img.get_fdata()
        data = data[:,:,:,sa_frames]

        img = nib.Nifti1Image(data, self.sa_img.affine, self.sa_img.header)
        if 'nii.gz' in self.sa_img.get_filename():
            nib.save(img, self.sa_img.get_filename().replace('.nii.gz', '_interp2la.nii.gz'))
        elif 'nii' in self.sa_img.get_filename():
            nib.save(img, self.sa_img.get_filename().replace('.nii', '_interp2la.nii'))

        seg = nib.load(sa_seg_path)
        seg_data = seg.get_fdata()
        seg_data = seg_data[:,:,:,sa_frames]
        img = nib.Nifti1Image(seg_data, seg.affine, seg.header)
        if 'nii.gz' in seg.get_filename():
            nib.save(img, seg.get_filename().replace('.nii.gz', '_interp2la.nii.gz'))
        elif 'nii' in seg.get_filename():
            nib.save(img, seg.get_filename().replace('.nii', '_interp2la.nii.gz'))



    """
    Plotting functions
    """

    def plot_volume_pressure_traces(self, lv_vol=None, lv_pres=None, rv_vol=None, rv_pres=None, lv_valve_times=None, rv_valve_times=None):
        if lv_vol is None:
            lv_vol_func = self.lv_vol_func
        else:
            lv_vol_func = interp1d(self.time, lv_vol, fill_value='extrapolate')
        if lv_pres is None:
            lv_pres_func = self.lv_pres_func
        else:
            lv_pres_func = interp1d(self.time, lv_pres, fill_value='extrapolate')
        if rv_vol is None:
            rv_vol_func = self.rv_vol_func
        else:
            rv_vol_func = interp1d(self.time, rv_vol, fill_value='extrapolate')
        if rv_pres is None:
            rv_pres_func = self.rv_pres_func
        else:
            rv_pres_func = interp1d(self.time, rv_pres, fill_value='extrapolate')
        if lv_valve_times is None:
            lv_valve_times = self.lv_valve_times
        if rv_valve_times is None:
            rv_valve_times = self.rv_valve_times

        fig, axs = plt.subplots(2, 1, clear=True, figsize=(4,8), num=1, sharex=True)
        axs[0].plot(self.time, lv_pres_func(self.time)*7.50062, 'k', label='LV')
        axs[0].plot(self.time, rv_pres_func(self.time)*7.50062, 'k--', label='RV')
        axs[1].plot(self.time, lv_vol_func(self.time)/1000, 'k')
        axs[1].plot(self.time, rv_vol_func(self.time)/1000, 'k--')
        axs[0].set_ylabel('Pressure (mmHg)')
        axs[1].set_xlabel('Time (ms)')
        axs[1].set_ylabel('Volume (mL)')

        # Add valve times as points in the trace
        for event in self.lv_events:
            if event != 'tcycle':
                lv_valve_time = lv_valve_times[event]
                lv_valve_pressure = lv_pres_func(lv_valve_time)
                lv_valve_volume = lv_vol_func(lv_valve_time)
                axs[0].plot(lv_valve_time, lv_valve_pressure*7.50062, 'ro')
                axs[1].plot(lv_valve_time, lv_valve_volume/1000, 'ro')

        for event in self.rv_events:
            if event != 'tcycle':
                rv_valve_time = rv_valve_times[event]
                rv_valve_pressure = rv_pres_func(rv_valve_time)
                rv_valve_volume = rv_vol_func(rv_valve_time)
                axs[0].plot(rv_valve_time, rv_valve_pressure*7.50062, 'ro')
                axs[1].plot(rv_valve_time, rv_valve_volume/1000, 'ro')

        axs[0].plot([],[],'ro', label='Valve events')
        axs[0].legend(frameon=False)
        axs[0].set_ylim([-5, 135])


    def plot_pv_loop(self, ax=None, lv_vol=None, lv_pres=None, rv_vol=None, rv_pres=None, lv_valve_times=None, rv_valve_times=None, add_klotz=False):
        if lv_vol is None:
            lv_vol_func = self.lv_vol_func
        else:
            lv_vol_func = interp1d(self.time, lv_vol, fill_value='extrapolate')
        if lv_pres is None:
            lv_pres_func = self.lv_pres_func
        else:
            lv_pres_func = interp1d(self.time, lv_pres, fill_value='extrapolate')
        if rv_vol is None:
            rv_vol_func = self.rv_vol_func
        else:
            rv_vol_func = interp1d(self.time, rv_vol, fill_value='extrapolate')
        if rv_pres is None:
            rv_pres_func = self.rv_pres_func
        else:
            rv_pres_func = interp1d(self.time, rv_pres, fill_value='extrapolate')
        if lv_valve_times is None:
            lv_valve_times = self.lv_valve_times
        if rv_valve_times is None:
            rv_valve_times = self.rv_valve_times

        if ax is None:
            fig, ax = plt.subplots(1, 1, clear=True, figsize=(5,4), num=2)
        ax.plot(lv_vol_func(self.time)/1000, lv_pres_func(self.time)*7.50062, 'k', label='LV')
        ax.plot(rv_vol_func(self.time)/1000, rv_pres_func(self.time)*7.50062, 'k--', label='RV')
        ax.set_xlabel('Volume (mL)')
        ax.set_ylabel('Pressure (mmHg)')
        ax.set_ylim([-5, 135])

        # Add valve times as points in the trace
        for event in self.lv_events:
            if event != 'tcycle':
                lv_valve_time = lv_valve_times[event]
                lv_valve_pressure = lv_pres_func(lv_valve_time)
                lv_valve_volume = lv_vol_func(lv_valve_time)
                ax.plot(lv_valve_volume/1000, lv_valve_pressure*7.50062, 'ro')

        for event in self.rv_events:
            if event != 'tcycle':
                rv_valve_time = rv_valve_times[event]
                rv_valve_pressure = rv_pres_func(rv_valve_time)
                rv_valve_volume = rv_vol_func(rv_valve_time)
                ax.plot(rv_valve_volume/1000, rv_valve_pressure*7.50062, 'ro')

        if add_klotz:
            V, P = self.klotz_curve(lv_vol, lv_pres)
            ax.plot(V/1000, P*7.50062, '0.5', label='Klotz curve')

        ax.legend()

"""
Klotz curve functions
"""


def klotz_V0(Vm, Pm):
    return Vm*(0.6 - 0.006*Pm)

def klotz_V30(V0, Vm, Pm, An = 27.78, Bn = 2.76):
    return V0 + (Vm-V0)/(Pm/An)**(1/Bn)

def klotz_ab(V30, Vm, Pm):
    beta = np.log(Pm/30)/np.log(Vm/V30)
    alpha = 30/V30**beta
    return alpha, beta