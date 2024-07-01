#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 22:30:00 2024

@author: jjv
"""
import numpy as np
import meshio as io
import cheartio as chio
from scipy.spatial import KDTree

def tet_volume(xyz, ien):
    points = xyz[ien]
    assert points.shape[1] == 4 and points.shape[2] == 3

    l0 = points[:,1,:] - points[:,0,:]
    l2 = points[:,0,:] - points[:,2,:]
    l3 = points[:,3,:] - points[:,0,:]

    norm_l0 = np.linalg.norm(l0, axis=1)
    norm_l2 = np.linalg.norm(l2, axis=1)
    norm_l3 = np.linalg.norm(l3, axis=1)

    cross_l2_l0 = np.cross(l2,l0, axisa=1, axisb=1)
    cross_l3_l0 = np.cross(l3,l0, axisa=1, axisb=1)
    cross_l3_l2 = np.cross(l3,l2, axisa=1, axisb=1)

    aux  = norm_l3[:,None]**2*cross_l2_l0
    aux += norm_l2[:,None]**2*cross_l3_l0
    aux += norm_l0[:,None]**2*cross_l3_l2

    volume = np.sum(cross_l2_l0*l3, axis=1)/6

    return volume

def compute_fibrosis_percentage(mesh):
    elem_size = tet_volume(xyz, ien)
    lv_elems = np.where(mesh.cell_data['rvlv'][0]==1)[0]
    lv_wall_volume = np.sum(elem_size[lv_elems])

    fibrosis = mesh.point_data['fibrosis']
    fib_elems = np.mean(fibrosis[ien], axis=1)
    mesh.cell_data['fib_elems'] = [fib_elems]

    lv_fib_volume = np.sum(fib_elems[lv_elems]*elem_size[lv_elems])
    lv_fib_percentage = lv_fib_volume/lv_wall_volume

    return lv_fib_percentage

def compute_wall_thickness(mesh):
    lv_elems = np.where(mesh.cell_data['rvlv'][0]==1)[0]
    lv_nodes = np.unique(ien[lv_elems])

    ctl = np.vstack([mesh.point_data['circ'], mesh.point_data['trans'], mesh.point_data['long_plane']]).T
    points = np.array([[np.pi/2, 0., 0.5],
                    [np.pi/2, 1., 0.5],
                    [-np.pi/2, 0., 0.5],
                    [-np.pi/2, 1., 0.5]])
    tree = KDTree(ctl[lv_nodes])
    _, nodes = tree.query(points)
    nodes = lv_nodes[nodes]
    wall_thickness1 = np.linalg.norm(xyz[nodes[::2]] - xyz[nodes[1::2]], axis=1)

    ctl = np.vstack([mesh.point_data['circ_aux'], mesh.point_data['trans'], mesh.point_data['long_plane']]).T

    points = np.array([[0., 0., 0.5],
                    [0., 1., 0.5],
                    [1., 0., 0.5],
                    [1., 1., 0.5]])
    tree = KDTree(ctl[lv_nodes])
    _, nodes = tree.query(points)
    nodes = lv_nodes[nodes]
    wall_thickness2 = np.linalg.norm(xyz[nodes[::2]] - xyz[nodes[1::2]], axis=1)
    wall_thickness = (np.sum(wall_thickness1)+wall_thickness2[1])/3

    return wall_thickness


path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/'
patients = ['VB-1', 'AV-19', 'BI-18', 'CA-15', 'ZS-11', 'TS-9', 'KL-4', 'KL-5', 'AB-17', 'JL-3', 'JN-8', 'DM-23', 'AS-10']
for patient in patients:
    mesh_fldr = path + patient + '/mesh/'
    data_fldr = path + patient + '/data/'

    mesh = io.read('{}/bv_mesh.vtu'.format(mesh_fldr))
    xyz = mesh.points
    ien = mesh.cells[0].data

    # Fibrosis percentage
    fib_percentage = compute_fibrosis_percentage(mesh)

    # Wall thickness
    wall_thickness = compute_wall_thickness(mesh)

    # End diastolic volume
    lv_volume = np.loadtxt('{}/volume_lv.norm'.format(mesh_fldr), usecols=1)
    rv_volume = np.loadtxt('{}/volume_rv.norm'.format(mesh_fldr), usecols=1)

    np.savez(mesh_fldr + 'mesh_stats.npz', fib_percentage=fib_percentage, wall_thickness=wall_thickness, lv_volume=lv_volume, rv_volume=rv_volume)