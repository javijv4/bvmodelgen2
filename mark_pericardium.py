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
from scipy.spatial import ConvexHull
import networkx as nx
from tqdm import tqdm

patient = 'ZS-11'
mesh_folder = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/es_mesh_ms25/'
data_path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/es_data_ms25/'

mv_normal = chio.read_dfile(mesh_folder + 'mv_normal.FE')

mesh = chio.read_mesh(mesh_folder + 'bv_model', meshio='True')
bdata = chio.read_bfile(mesh_folder + 'bv_model')

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

xyz = mesh.points
ien = mesh.cells[0].data

# hull = ConvexHull(xyz)
# io.write_points_cells('check.vtu', xyz, {'triangle':hull.simplices})
# io.write_points_cells('check.vtu', xyz, {'tetra':ien})



# Find apex and calculate long axis distance
mv_surf = bdata[bdata[:,-1]==boundaries['mv'], 1:-1]
mv_nodes = np.unique(mv_surf.ravel())
mv_normal = mv_normal/np.linalg.norm(mv_normal)
mv_centroid = np.mean(mesh.points[mv_nodes], axis=0)

la_dist = np.dot(mesh.points - mv_centroid, mv_normal)
apex = np.argmin(la_dist)
la_dist = la_dist - la_dist[apex]
la_dist = la_dist/np.max(la_dist)

# The lowest av node has la_dist = 1
av_surf = bdata[bdata[:,-1]==boundaries['av'], 1:-1]
av_nodes = np.unique(av_surf.ravel())

la_dist_av = la_dist[av_nodes]
min_av_dist = np.min(la_dist_av)
la_dist = la_dist/min_av_dist

# Grab valve and epicardium surfaces
av_surf = bdata[bdata[:,-1]==boundaries['av'], 1:-1]
pv_surf = bdata[bdata[:,-1]==boundaries['pv'], 1:-1]
tv_surf = bdata[bdata[:,-1]==boundaries['tv'], 1:-1]
lv_epi_surf = bdata[bdata[:,-1]==boundaries['lv_epi'], 1:-1]
rv_epi_surf = bdata[bdata[:,-1]==boundaries['rv_epi'], 1:-1]

epi_surf = np.vstack((lv_epi_surf, rv_epi_surf))
epi_nodes = np.unique(epi_surf.ravel())

surf = np.vstack((mv_surf, av_surf, pv_surf, tv_surf, epi_surf))

arr = np.array([[0,1],[1,2],[2,0]])
lines = np.vstack(surf[:,arr])

# Create a graph
G = nx.Graph()

# Generate the graph from the FE mesh
G.add_nodes_from(np.arange(len(xyz)))
# Calculate edge distances
edge_distances = np.linalg.norm(mesh.points[lines[:, 0]] - mesh.points[lines[:, 1]], axis=1)

# Add edge distances to the graph
for i, edge in enumerate(lines):
    G.add_edge(edge[0], edge[1], distance=edge_distances[i])

# Get nodes for the valves
epi_nodes = np.unique(epi_surf.ravel())
av_nodes = np.intersect1d(np.unique(av_surf.ravel()), epi_nodes)[::3]
pv_nodes = np.intersect1d(np.unique(pv_surf.ravel()), epi_nodes)[::3]
tv_nodes = np.intersect1d(np.unique(tv_surf.ravel()), epi_nodes)[::3]
mv_nodes = np.intersect1d(np.unique(mv_surf.ravel()), epi_nodes)[::3]

marker = np.zeros(len(xyz))

# Find the shortest path between the valves
def find_shortest_path_between_valves(G, nodes1, nodes2, marker):
    for i, node in tqdm(enumerate(nodes1)):
        for j, node2 in enumerate(nodes2):
            path = nx.shortest_path(G, node, node2, weight='distance')
            marker[path] = 1

find_shortest_path_between_valves(G, pv_nodes, tv_nodes, marker)
find_shortest_path_between_valves(G, pv_nodes, mv_nodes, marker)
find_shortest_path_between_valves(G, tv_nodes, mv_nodes, marker)

# # add nodes missing
av_nodes = np.unique(av_surf.ravel())
pv_nodes = np.unique(pv_surf.ravel())
tv_nodes = np.unique(tv_surf.ravel())
mv_nodes = np.unique(mv_surf.ravel())
marker[av_nodes] = 1
marker[pv_nodes] = 1
marker[tv_nodes] = 1
marker[mv_nodes] = 1

non_marked = np.where(marker==0)[0]
marked = np.where(marker==1)[0]

changed = 1
while changed > 0:
    sum_marker = np.sum(marker)
    for node in non_marked:
        neighbors = list(G.neighbors(node))
        base_nodes = np.sum(marker[neighbors])
        if base_nodes == 0: continue
        base_nodes = base_nodes/(len(neighbors)-2)
        if base_nodes >= 1.0: marker[node] = 1
    changed = np.sum(marker) - sum_marker

epi_nodes = 1-marker

#%%
# # Calculate the surface distance of the epi nodes to the base nodes
epi_elems = np.where(np.sum(np.isin(surf, np.where(epi_nodes)[0]), axis=1) > 2)[0]
epi_elems = surf[epi_elems]
epi_weight = np.zeros(len(mesh.points))
epi_weight[epi_elems.ravel()] = 1

propagation = 16

# Find how far an element is from the valves
nx = np.zeros(len(mesh.points))
for i in range(len(ien)):
    nx[ien[i]] += 4

# Propagation
for i in range(propagation):
    aux = epi_weight.copy()
    for j in range(len(ien)):
        ax = np.sum(epi_weight[ien[j]])
        aux[ien[j]] += ax
    aux = aux / nx
    aux[epi_elems.ravel()] = 1
    epi_weight = aux

#%%
epi_weight[epi_weight>1] = 1
epi_weight[av_nodes] = 0
epi_weight[pv_nodes] = 0
epi_weight[tv_nodes] = 0
epi_weight[mv_nodes] = 0
epi_weight[epi_weight<1e-2] = 0

epi_nodes = np.unique(surf.ravel())
non_epi_nodes = np.setdiff1d(np.arange(len(mesh.points)), epi_nodes)
epi_weight[non_epi_nodes] = 0

chio.write_dfile(data_path + 'pericardium_mask.FE', epi_weight)

# mesh.point_data['marker'] = marker
# mesh.point_data['epi_weight'] = epi_weight
# io.write('check.vtu', mesh)
