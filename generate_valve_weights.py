#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/08/04 12:32:34

@author: Javiera Jilberto Vallejos
'''

import os
import numpy as np
import meshio as io
import cheartio as chio

patient = 'KL-4'
mesh_folder = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/es_mesh_ms25/'
data_path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/es_data_ms25/'

propagation = 16 # Propagation away from the valves

if not os.path.exists(data_path): os.mkdir(data_path)


mesh_path = mesh_folder + 'bv_model'

boundaries = {'lv_endo'     : 1,
              'rv_endo'     : 2,
              'lv_epi'      : 3,
              'rv_epi'      : 4,
              'rv_septum'   : 5,
              'av'          : 6,
              'pv'          : 7,
              'tv'          : 8,
              'mv'          : 9,
              'rv_lv_junction'          : 10}

# Reading mesh
mesh = chio.read_mesh(mesh_path, meshio=True)
bdata = chio.read_bfile(mesh_path)

xyz = mesh.points
ien = mesh.cells[0].data

# Find how far an element is from the valves
nx = np.zeros(len(xyz))
for i in range(len(ien)):
    nx[ien[i]] += 4

# Valve weight
valve_weight_fields = []
for valve in ['av', 'mv', 'tv', 'pv']:
    valve_elems = bdata[(bdata[:,-1]==boundaries[valve]), 1:-1]

    valve_weight = np.zeros(len(xyz))
    valve_weight[valve_elems.ravel()] = 1

    # Propagation
    for i in range(propagation):
        aux = valve_weight.copy()
        for j in range(len(ien)):
            ax = np.sum(valve_weight[ien[j]])
            aux[ien[j]] += ax
        aux = aux / nx
        aux[valve_elems.ravel()] = 1
        valve_weight = aux

    # Cutoff
    valve_weight[valve_weight > 1] = 1

    x = 2*valve_weight - 1.0
    sigmoid_valve_weight = (np.tanh(2*x)/np.tanh(2)+1)/2
    sigmoid_valve_weight[sigmoid_valve_weight<1e-2] = 0 

    valve_weight_fields.append(sigmoid_valve_weight)
    mesh.point_data[valve] = sigmoid_valve_weight

# Need to find common elements for the mv and av
mv = valve_weight_fields[1]
av = valve_weight_fields[0]

bridge = np.sqrt(5*mv*av)
bridge[bridge > 1] = 1

valve_weight = np.sum(np.array(valve_weight_fields), axis=0)
valve_weight[valve_weight > 1] = 1
valve_weight = valve_weight-bridge

#%%
# # Apply sigmoid function to valve_weight
# x = 2*valve_weight - 1.0
# sigmoid_valve_weight = (np.tanh(2*x)/np.tanh(2)+1)/2
# sigmoid_valve_weight[sigmoid_valve_weight<1e-2] = 0 

# x = 2*bridge - 1.0
# sigmoid_bridge = (np.tanh(x)/np.tanh(1)+1)/2
# sigmoid_bridge[sigmoid_bridge<1e-2] = 0

mesh.point_data['mv_av_bridge'] = bridge
mesh.point_data['valve_weight'] = valve_weight
io.write('check.vtu', mesh)

# Saving
chio.write_dfile(data_path + 'valve_weight.FE', valve_weight)
chio.write_dfile(data_path + 'lv_bridge_weight.FE', bridge)