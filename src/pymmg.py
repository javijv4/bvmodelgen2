#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:27:58 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import meshio as io
import subprocess as sp
import os

def load_sol(fname):
    cont = 0
    marker = 0

    f = open(fname)
    lines = f.readlines()
    f.close()

    values = []
    for line in lines:
        if line == '\n': continue
        cont += 1
        if 'Sol' in line:
            marker += 1
        elif marker == 1:
            marker += 1
        elif marker == 2:
            marker += 1
        elif marker > 2 and ('End' not in line):

            values.append(float(line))
    array = np.array(values)
    return array

def write_sol(fname, array, dim):
    shape = array.shape
    if len(shape) == 1:
        nnodes = shape[0]
        arrsize = 1
    else:
        nnodes, arrsize = shape

    header = 'MeshVersionFormatted 2\nDimension {:d}\nSolAtVertices\n{:d}\n{:d} {:d}'.format(dim, nnodes, 1, arrsize)
    np.savetxt(fname, array, header=header, footer='End', comments='')


def mmg_isoline_meshing(mesh, function, isovalue=0, funcs_to_interpolate=[], bdata=None):
    dim = 3     # TODO figure out this from the mesh

    # Get mesh size
    obj_size = np.max(np.max(mesh.points, axis=0) - np.min(mesh.points, axis=0))

    # Writing files in mmg format
    io.write('mesh.mesh', mesh)
    write_sol('func.sol', function, dim)
    with open('mmg.log', 'w') as ofile:
        p = sp.Popen(['mmg3d', 'mesh.mesh', '-sol', 'func.sol', '-ls', '{:f}'.format(isovalue),
                      '-nr', '-noinsert', '-noswap', '-nomove',
                      '-hausd', '{:f}'.format(obj_size/100),
                      '-out', 'output.o.meshb'], stdout=ofile, stderr=ofile)
        p.wait()

    out_mesh = io.read('output.o.meshb')
    new_func = np.zeros(len(out_mesh.points))
    new_func[:len(mesh.points)] = function
    new_func[len(mesh.points):] = isovalue
    tet_mesh = io.Mesh(out_mesh.points, {'tetra': out_mesh.cells_dict['tetra']},
                       point_data = {'f': new_func})

    # Cleaning
    os.remove('mesh.mesh')
    os.remove('func.sol')
    os.remove('output.o.meshb')
    os.remove('output.o.sol')
    os.remove('mmg.log')

    elem_map = False
    if len(funcs_to_interpolate) != 0:
        new_nodes, node_map = generate_element_map(tet_mesh, mesh, function, isovalue)
        elem_map = True
        ien = mesh.cells[0].data
        xyz = mesh.points
        interp_basis = np.zeros([len(new_nodes),4])
        for i in range(len(new_nodes)):
            elem = node_map[i]
            point = tet_mesh.points[new_nodes[i]]
            vertex = xyz[ien[elem]]
            interp_basis[i] = interpolate_tet(point, vertex)


        for func in funcs_to_interpolate:
            new_func = np.zeros(len(tet_mesh.points))
            new_func[:len(mesh.points)] = mesh.point_data[func]
            elem_func = new_func[ien[node_map]]
            new_func[new_nodes] = np.sum(elem_func*interp_basis, axis=1)
            tet_mesh.point_data[func] = new_func

    if bdata is None:
        return tet_mesh
    else:
        if not elem_map:
            _, _ = generate_element_map(tet_mesh, mesh, function, isovalue)
        mmg_bdata = generate_tri_element_map(tet_mesh, bdata)
        return tet_mesh, mmg_bdata



def interpolate_tet(point, vertex):
    A = np.vstack([vertex.T, np.ones(4)])
    b = np.append(point,1)
    xis = np.linalg.solve(A,b)

    return xis

def det_mat_tet(mat):
    a,b,c,d,e,f,g,h,i,j,k,l = mat.reshape([-1,12]).T

    det = a*e*i - a*e*l - a*f*h + a*f*k + a*h*l - a*i*k \
        - b*d*i + b*d*l + b*f*g - b*f*j - b*g*l + b*i*j \
        + c*d*h - c*d*k - c*e*g + c*e*j + c*g*k - c*h*j \
        - d*h*l + d*i*k + e*g*l - e*i*j - f*g*k + f*h*j

    return det


def det_mat_tri(mat):
    a,b,c,d,e,f = mat.reshape([-1,6]).T

    det = a*d - a*f - b*c + b*e + c*f - d*e
    return det


def generate_element_map(mmg_mesh, mesh, function, isovalue):
    """ Returns an element in the original mesh to which the new node belongs to """
    xyz = mesh.points
    ien = mesh.cells[0].data
    mmg_xyz = mmg_mesh.points
    mmg_ien = mmg_mesh.cells[0].data

    div_elems = np.where(np.any(function[ien]>=isovalue, axis=1)*np.any(function[ien]<isovalue, axis=1))[0]
    new_nodes = np.arange(len(xyz), len(mmg_xyz), dtype=int)   # New nodes are div nodes
    new_elems = np.where(np.any(np.isin(mmg_ien, new_nodes), axis=1))[0]

    div_ien = ien[div_elems]
    mmg_div_ien = mmg_ien[new_elems]
    mmg_div_midpoints = np.mean(mmg_xyz[mmg_div_ien], axis=1)

    elem_to_elem = np.arange(len(mmg_div_ien))
    for i in range(len(mmg_div_ien)):
        ocurrences = np.sum(np.isin(div_ien, mmg_div_ien[i]),axis=1)
        celems = np.where(ocurrences == np.max(ocurrences))[0]          # Candidate elems
        vertex = xyz[div_ien[celems]]
        point = mmg_div_midpoints[i]

        d4 = det_mat_tet(vertex)
        work = np.copy(vertex)
        work[:,0] = point
        d0 = det_mat_tet(work)
        work = np.copy(vertex)
        work[:,1] = point
        d1 = det_mat_tet(work)
        work = np.copy(vertex)
        work[:,2] = point
        d2 = det_mat_tet(work)
        work = np.copy(vertex)
        work[:,3] = point
        d3 = det_mat_tet(work)

        ds = np.vstack([d0,d1,d2,d3,d4]).T
        ind = np.where(np.abs(np.sum(np.sign(ds), axis=1))==5)[0]
        if len(ind)==0:         # Not part of the division
            elem_to_elem[i] = new_elems[i]
        else:
            elem_to_elem[i] = div_elems[celems[ind[0]]]

    node_to_elem = np.arange(len(new_nodes))
    for i in range(len(new_nodes)):
        elem = np.where(mmg_div_ien==new_nodes[i])[0][0]     # I only need one, so I just take the first
        node_to_elem[i] = elem_to_elem[elem]

    arr_elem_map = np.arange(len(mmg_mesh.cells[0].data))
    arr_elem_map[new_elems] = elem_to_elem
    mmg_mesh.cell_data['map'] = [arr_elem_map]


    return new_nodes, node_to_elem

def get_face_normal(xyz, ien):
    vertex = xyz[ien]
    if len(ien.shape) == 1:
        normal = np.cross(vertex[1]-vertex[0],vertex[2]-vertex[0])
        normal = normal/np.linalg.norm(normal)
    else:
        normal = np.cross(vertex[:,1]-vertex[:,0],vertex[:,2]-vertex[:,0], axisa=1, axisb=1)
        normal = normal/np.linalg.norm(normal, axis=1)[:,None]
    return normal



def generate_tri_element_map(mmg_mesh, bdata):
    faces = bdata[:,1:-1]
    tet_belems = bdata[:,0]
    marker = bdata[:,-1]
    mmg_xyz = mmg_mesh.points
    mmg_ien = mmg_mesh.cells[0].data
    elem_to_elem = mmg_mesh.cell_data['map'][0]
    mmg_tet_belems = np.where(np.isin(elem_to_elem, tet_belems))[0]

    # Generate mesh of faces
    array = np.array([[0,1,2],[1,2,3],[0,1,3],[2,0,3]])
    nelems = np.repeat(np.arange(mmg_ien.shape[0])[mmg_tet_belems],4)
    mmg_faces = np.vstack(mmg_ien[:, array][mmg_tet_belems])

    face_to_face = -np.ones(len(mmg_faces), dtype=int)
    for i in range(len(mmg_faces)):         # I can make this better. I know what element the face belongs to and what element that was in the og mesh
        mmg_tet_elem = nelems[i]
        tet_elem = elem_to_elem[mmg_tet_elem]
        face_id = np.where(tet_belems == tet_elem)[0]
        if len(face_id) == 0:
            continue
        else:
            # Check if normal is correct
            normals = get_face_normal(mmg_xyz, faces[face_id])
            normal = get_face_normal(mmg_xyz, mmg_faces[i])
            dot = np.abs(np.sum(normals*normal, axis=1))
            ind = np.where(np.isclose(dot,1))[0]
            if len(ind) == 0:
                continue
            else:
                face_to_face[i] = face_id[ind][0]


    bmarker = face_to_face>=0
    mmg_faces = mmg_faces[bmarker]
    face_to_face = face_to_face[bmarker]

    mmg_bdata = np.vstack([nelems[bmarker],mmg_faces.T,marker[face_to_face]]).T

    return mmg_bdata

def mmg_surf_isoline_meshing(mesh, function, isovalue=0):
    dim = 3     # TODO figure out this from the mesh

    # Get mesh size
    obj_size = np.max(np.max(mesh.points, axis=0) - np.min(mesh.points, axis=0))

    # Writing files in mmg format
    io.write('mesh.mesh', mesh)
    write_sol('func.sol', function, dim)
    with open('mmg.log', 'w') as ofile:
        p = sp.Popen(['mmg3d', 'mesh.mesh', '-sol', 'func.sol', '-lssurf',  '{:f}'.format(isovalue),
                      '-nr', '-noinsert', '-noswap', '-nomove',
                      '-hausd', '{:f}'.format(obj_size/100),
                      '-out', 'output.o.meshb'], stdout=ofile, stderr=ofile)
        p.wait()

    out_mesh = io.read('output.o.meshb')
    new_func = np.zeros(len(out_mesh.points))
    new_func[:len(mesh.points)] = function
    new_func[len(mesh.points):] = isovalue
    tet_mesh = io.Mesh(out_mesh.points, {'tetra': out_mesh.cells_dict['tetra']},
                       point_data = {'f': new_func})
    elem_map = generate_element_map(tet_mesh, mesh, function, isovalue)
    tet_mesh.cell_data['map'] = [elem_map]

    mesh.cell_data['map'] = [np.arange(len(mesh.cells[0].data))]

    # Cleaning
    os.remove('mesh.mesh')
    os.remove('func.sol')
    os.remove('output.o.meshb')
    os.remove('output.o.sol')
    os.remove('mmg.log')

    return tet_mesh

def get_mesh_size(mesh):
    ien = mesh.cells[0].data
    xyz = mesh.points
    elems = np.random.randint(0, ien.shape[0], size=20)    # Randomly taking 20 elements
    xyz = xyz[ien[elems]]

    l0 = np.linalg.norm(xyz[:,1,:]-xyz[:,0,:], axis=1)
    return np.mean(l0)