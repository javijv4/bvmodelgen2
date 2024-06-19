#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:55:55 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np

def map_fibrosis(fib_func, mesh, mesh_info):
    mesh_la_vector = mesh_info['mv_centroid'] - mesh.points[mesh_info['apex_lv_epi']]
    mesh_la_length = np.linalg.norm(mesh_la_vector)
    mesh_la_vector = mesh_la_vector/mesh_la_length

    long = mesh_la_vector@(mesh.points - mesh.points[mesh_info['apex_lv_epi']]).T
    long = long/mesh_la_length
    circ = mesh.point_data['circ']
    trans = mesh.point_data['trans']

    lv_nodes = np.unique(mesh.cells[0].data[mesh.cell_data['rvlv'][0] == 1])

    ctl = np.vstack([circ[lv_nodes], trans[lv_nodes], long[lv_nodes]]).T

    fibrosis = fib_func(ctl)
    fibrosis[np.isnan(fibrosis)] = 0

    bv_fibrosis = np.zeros(len(mesh.points))
    bv_fibrosis[lv_nodes] = fibrosis

    return bv_fibrosis, long

