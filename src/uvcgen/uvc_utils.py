#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:31:32 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import meshio as io


def get_surface_orthogonal_vector(mesh, bdata, patch):
    from scipy.optimize import minimize
    def pnorm(v, p):
        return np.sum(np.abs(v)**p)**(1/p)
    def func(la_vec, normals, p=1.373):
        la_vec = la_vec/np.linalg.norm(la_vec)
        dot_p = normal@la_vec
        return pnorm(dot_p, p)

    xyz = mesh.points
    lv_endo_ien = bdata[bdata[:,-1] == patch, 1:-1]

    v1 = xyz[lv_endo_ien[:,1]] - xyz[lv_endo_ien[:,0]]
    v2 = xyz[lv_endo_ien[:,2]] - xyz[lv_endo_ien[:,0]]

    normal = np.cross(v1, v2, axisa=1, axisb=1)
    normal = normal/np.linalg.norm(normal, axis=1)[:,None]

    sol = minimize(func, np.array([1,0,0]), args=(normal))

    return sol.x/np.linalg.norm(sol.x)

def get_surface_mesh(mesh):
    ien = mesh.cells[0].data

    array = np.array([[0,1,2],[1,2,3],[0,1,3],[2,0,3]])
    nelems = np.repeat(np.arange(ien.shape[0]),4)
    faces = np.vstack(ien[:,array])
    sort_faces = np.sort(faces,axis=1)

    f, i, c = np.unique(sort_faces, axis=0, return_counts=True, return_index=True)
    ind = i[np.where(c==1)[0]]
    bfaces = faces[ind]
    belem = nelems[ind]

    return belem, bfaces


def get_surface_normals(points, ien, vol_elems=None):
    points_elems = points[ien]
    if ien.shape[1] == 2:   # Lines
        v1 = points_elems[:,1] - points_elems[:,0]
        v2 = np.array([0,0,1])

        normal = np.cross(v1, v2, axisa=1)
        normal = normal/np.linalg.norm(normal,axis=1)[:,None]

    if ien.shape[1] == 3:

        v1 = points_elems[:,1] - points_elems[:,0]
        v2 = points_elems[:,2] - points_elems[:,0]

        normal = np.cross(v1, v2, axisa=1, axisb=1)
        normal = normal/np.linalg.norm(normal,axis=1)[:,None]

    if vol_elems is None:
        return normal

    elem_midpoint = np.mean(points[vol_elems], axis=1)
    face_midpoint = np.mean(points[ien], axis=1)

    vector = face_midpoint-elem_midpoint
    dot = np.sum(normal*vector, axis=1)
    normal[dot<0] *= -1

    return normal



def create_submesh(mesh, map_mesh_submesh_elems):
    submesh_elems = mesh.cells[0].data[map_mesh_submesh_elems]
    submesh_xyz = np.zeros([len(np.unique(submesh_elems)),3])
    map_mesh_submesh = np.ones(mesh.points.shape[0], dtype=int)*-1
    map_submesh_mesh = np.zeros(submesh_xyz.shape[0], dtype=int)
    child_elems_new = np.zeros(submesh_elems.shape, dtype=int)

    cont = 0
    for e in range(submesh_elems.shape[0]):
        for i in range(submesh_elems.shape[1]):
            if map_mesh_submesh[submesh_elems[e,i]] == -1:
                child_elems_new[e,i] = cont
                submesh_xyz[cont] = mesh.points[submesh_elems[e,i]]
                map_mesh_submesh[submesh_elems[e,i]] = cont
                map_submesh_mesh[cont] = submesh_elems[e,i]
                cont += 1
            else:
                child_elems_new[e,i] = map_mesh_submesh[submesh_elems[e,i]]

    submesh = io.Mesh(submesh_xyz, {mesh.cells[0].type: child_elems_new})
    return submesh, map_mesh_submesh, map_submesh_mesh


def create_submesh_bdata(submesh, mesh_bdata, map_mesh_submesh, map_submesh_mesh_elems, method):
    if method == 'parent':  # This means it will only look for the faces defined in the parent mesh
        belem = map_submesh_mesh_elems[mesh_bdata[:,0]]
        submesh_marker = belem >= 0
        belem = belem[submesh_marker]
        bfaces = map_mesh_submesh[mesh_bdata[submesh_marker,1:-1]]
        marker = mesh_bdata[submesh_marker,-1]

    elif method == 'boundary':  # This will look for the boundaries of the submesh and compare it with the parent to get markers
        belem, bfaces = get_surface_mesh(submesh)

        # Create face marker using bv mesh
        nb = np.unique(mesh_bdata[:,-1])

        marker = np.zeros(bfaces.shape[0], dtype=int)
        for b in nb:
            b_ien = map_mesh_submesh[mesh_bdata[mesh_bdata[:,-1]== b, 1:-1]]
            b_ien = b_ien[np.min(b_ien >= 0, axis=1)]
            marker[np.sum(np.isin(bfaces, b_ien), axis=1) == 3] = b

    bdata = np.hstack([belem[:,None], bfaces, marker[:,None]])

    return bdata


def find_isoline(field, value, ien):
    field_elem = field[ien]
    nodes_per_elem = ien.shape[1]
    nplus = np.sum(field_elem > value, axis=1)
    div_elems = np.where( (nplus > 0) & (nplus < nodes_per_elem))[0]
    div_nodes = np.unique(ien[div_elems])

    pos_nodes = div_nodes[field[div_nodes] > value]
    neg_nodes = div_nodes[field[div_nodes] <= value]

    return pos_nodes, neg_nodes, div_nodes, div_elems

def get_normal_plane_three_points(p0, p1, p2):
    v1 = p1 - p0
    v1 = v1/np.linalg.norm(v1)
    v2 = p2 - p0
    v2 = v2/np.linalg.norm(v2)

    return np.cross(v2,v1)

def get_normal_plane_svd(points):   # Find the plane that minimizes the distance given N points
    centroid = np.mean(points, axis=0)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1]
    normal = normal/np.linalg.norm(normal)
    return normal, centroid


# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def rotate_vector_rodriguez(vector, axis, theta):
    if len(vector.shape) == 1:  # Only one vector
        return vector*np.cos(theta) + np.cross(axis, vector)*np.sin(theta) + axis*(np.dot(axis, vector))*(1-np.cos(theta))
    else:   # Multiple vectors stacked
        return vector*np.cos(theta)[:,None] + \
                np.cross(axis, vector, axisa=1, axisb=1)*np.sin(theta)[:,None] + \
                axis*(np.sum(axis*vector))*(1-np.cos(theta))[:,None]

# https://vtk.org/Wiki/images/6/6b/VerdictManual-revA.pdf
def tet_quality_radius_ratio(xyz, ien):
    points = xyz[ien]
    assert points.shape[1] == 4 and points.shape[2] == 3

    l0 = points[:,1,:] - points[:,0,:]
    l1 = points[:,2,:] - points[:,1,:]
    l2 = points[:,0,:] - points[:,2,:]
    l3 = points[:,3,:] - points[:,0,:]
    l4 = points[:,3,:] - points[:,1,:]

    norm_l0 = np.linalg.norm(l0, axis=1)
    norm_l2 = np.linalg.norm(l2, axis=1)
    norm_l3 = np.linalg.norm(l3, axis=1)

    cross_l2_l0 = np.cross(l2,l0, axisa=1, axisb=1)
    cross_l3_l0 = np.cross(l3,l0, axisa=1, axisb=1)
    cross_l4_l1 = np.cross(l4,l1, axisa=1, axisb=1)
    cross_l3_l2 = np.cross(l3,l2, axisa=1, axisb=1)

    aux  = norm_l3[:,None]**2*cross_l2_l0
    aux += norm_l2[:,None]**2*cross_l3_l0
    aux += norm_l0[:,None]**2*cross_l3_l2

    volume = np.sum(cross_l2_l0*l3, axis=1)/6
    area = 0.5*(np.linalg.norm(cross_l2_l0, axis=1)+
                np.linalg.norm(cross_l3_l0, axis=1)+
                np.linalg.norm(cross_l4_l1, axis=1)+
                np.linalg.norm(cross_l3_l2, axis=1))

    aux = np.linalg.norm(aux, axis=1)
    quality = aux*area/(108*volume**2)

    r = 3*volume/area
    R = aux/(12*volume)
    quality = R/(3*r)

    return quality