#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:20:42 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
from mpi4py import MPI
from dolfinx import io
import os


vol_to_surf_elem = {'hexahedron': 'quad',
                    'tetra': 'triangle',
                    'tetra10': 'triangle6'}

def read_mesh(mesh_path, grid_name = 'Grid'):
    """
    Parameters
    ----------
    mesh_path : str
        path to XDMF mesh
    Returns
    -------
    mesh : dolfinx mesh
    """
    with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
        mesh = xdmf.read_mesh(name=grid_name)
    return mesh

def read_meshtags(mt_path, mesh):
    """
    Parameters
    ----------
    mt_path : str
        path to XDMF surface mesh
    mesh : dolfinx mesh
    Returns
    -------
    meshtags : dolfinx mesh tags
    """

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with io.XDMFFile(MPI.COMM_WORLD, mt_path, "r") as xdmf:
        meshtags = xdmf.read_meshtags(mesh, name="Grid")

    return meshtags


def read_meshio_mesh(mesh, bdata=None, clean=True):
    import meshio as meshio
    meshio.write('mesh.xdmf', mesh)

    if bdata is None:
        mesh = read_mesh('mesh.xdmf')

        # cleaning
        os.remove('mesh.xdmf')
        os.remove('mesh.h5')

        return mesh
    else:
        vol_element = mesh.cells[0].type
        surf_element = vol_to_surf_elem[vol_element]
        surf_mesh = meshio.Mesh(mesh.points, {surf_element: bdata[:,1:-1]},
                            cell_data = {'patches': [bdata[:,-1]]})
        meshio.write('mt.xdmf', surf_mesh)
        mesh = read_mesh('mesh.xdmf')
        mt = read_meshtags('mt.xdmf', mesh)

        # cleaning      # TODO do this smarter
        if clean:
            os.remove('mesh.xdmf')
            os.remove('mt.xdmf')
            os.remove('mesh.h5')
            os.remove('mt.h5')

        return mesh, mt


def read_ch_mesh(mesh_path, boundary = True):
    import cheartio as chio

    # Mesh
    mesh = chio.read_mesh(mesh_path, meshio = True)

    if not boundary:
        out = read_meshio_mesh(mesh, bdata=None)
    else:
        bdata = chio.read_bfile(mesh_path, ordervtu=True)
        out = read_meshio_mesh(mesh, bdata=bdata)

    return out


def visualize_mesh(fname, mesh):
    with io.XDMFFile(mesh.comm, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)


def visualize_meshtags(fname, mesh, mt):
    with io.XDMFFile(mesh.comm, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(mt)


def visualize_function(fname, u):
    mesh = u.function_space.mesh
    with io.XDMFFile(mesh.comm, fname, "w") as file:
        file.write_mesh(mesh)
        file.write_function(u)


def find_vtu_dx_mapping(mesh, cells=False):
    if cells:
        map_dx_vtu = mesh.topology.original_cell_index
        map_vtu_dx = np.argsort(map_dx_vtu)

    else:
        # from dx to vtu
        map_dx_vtu = mesh.geometry.input_global_indices
        map_vtu_dx = np.argsort(map_dx_vtu)


    return map_vtu_dx, map_dx_vtu