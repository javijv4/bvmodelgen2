#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 16:24:19 2023

@author: Javiera Jilberto Vallejos

Based on a MATLAB implementation and translated using chatGPT.
"""
import numpy as np
import scipy.sparse as sparse

# labels = {'inlet': 1,
#           'outlet': 2,
#           'walls': 3,
#           'exterior': -1,
#           }

inlet = 1
outlet = 2
walls = 3
exterior = -1
interior = 0


def uvc_get_index_mapping(labels):
    ind = np.zeros_like(labels, dtype=int)

    # Label all nodes (not exterior) ...
    ide = np.where(labels != exterior)
    n = len(ide[0])
    ind[ide] = np.arange(n, dtype=int)

    return n, ind


def solve_laplace(A, RHS, mask):
    sol = sparse.linalg.spsolve(A,RHS)

    field = np.zeros(mask.shape)
    field[mask==1] =  sol

    return field


def uvc_get_ablock(im, ind, labels, n, pixdim):
    # Image type ...
    image_is_3D = len(im.shape) == 3
    if not image_is_3D:
        im_ = im[:,:,None]
        ind_ = ind[:,:,None]
        labels_ = labels[:,:,None]
    else:
        im_ = im
        ind_ = ind
        labels_ = labels

    # Constructing the block matrix using position (I,J) and value (V)
    I = np.ones(6*n, dtype=int)
    J = np.ones(6*n, dtype=int)
    V = np.zeros(6*n)

    # Building the block entries ...
    # loop over dimensions
    it = 0
    for i in range(im_.shape[0]):
        for j in range(im_.shape[1]):
            for k in range(im_.shape[2]):
                row = ind_[i,j,k]  # Get row numbering based on indexing ind_
                if labels_[i,j,k] == exterior:
                    continue  # If exterior, continue
                elif labels_[i,j,k] == inlet or labels_[i,j,k] == outlet:  # If inlet or wall
                    I[it] = row
                    J[it] = row
                    V[it] = 1
                    it += 1
                    continue
                # If outlet or interior, get appropriate stencil values
                sx, sx_value, sy, sy_value, sz, sz_value = get_Ablock_stencil(labels_, i, j, k)
                # Set x-direction pos (I,J) and values (V)
                I[it:it+len(sx)] = row
                J[it:it+len(sx)] = ind_[i+sx, j, k]
                V[it:it+len(sx)] = (sx_value/(pixdim[0]**2))
                it += len(sx)
                # Set y-direction pos (I,J) and values (V)
                I[it:it+len(sy)] = row
                J[it:it+len(sy)] = ind_[i, j+sy, k]
                V[it:it+len(sy)] = (sy_value/(pixdim[1]**2))
                it += len(sy)
                if image_is_3D:
                    # Set z-direction pos (I,J) and values (V)
                    I[it:it+len(sz)] = row
                    J[it:it+len(sz)] = ind_[i, j, k+sz]
                    V[it:it+len(sz)] = (sz_value/(pixdim[2]**2))
                    it += len(sz)

    # Generate sparse matrix A. NOTE: Repeated (I,J) gets added (V) automatic!
    A = sparse.coo_matrix((V, (I, J)), shape=(n, n))
    A = A.tocsr()

    # Check whether max(row) < 1e-8, if so --> A(row,row) = 1, dirichlet(row) = 1
    vector = np.max(A, axis=1).todense()
    idx = np.where(vector < 1e-8)[0]
    A[idx,idx] = 1

    return A


def get_Ablock_stencil(labels, i, j, k):
    if labels[i, j, k] == interior: # If interior...
        sx = np.array([-1, 0, 1])
        sx_value = np.array([1, -2, 1])
        sy = sx
        sy_value = sx_value
        sz = sx
        sz_value = sx_value
    elif labels[i, j, k] == outlet:
        raise ValueError('stencil should not apply to walls')
    elif labels[i, j, k] == inlet:
        raise ValueError('stencil should not apply to inlet')
    elif labels[i, j, k] == exterior:
        raise ValueError('stencil should not apply to exterior')
    elif labels[i, j, k] == walls: # Special stencils for boundary
        sx, sx_value = get_boundary_stencilx(labels, i, j, k)
        sy, sy_value = get_boundary_stencily(labels, i, j, k)
        sz, sz_value = get_boundary_stencilz(labels, i, j, k)

    return sx, sx_value, sy, sy_value, sz, sz_value


def get_boundary_stencilx(labels, i, j, k):
    if i == 0:
        if labels[i+1,j,k] != exterior:
            sd = np.array([0, 1])
            sd_value = np.array([-1, 1])
            return sd, sd_value
        else:
            sd = np.array([0])
            sd_value = np.array([0])
            return sd, sd_value
    elif i == labels.shape[0] - 1:
        if labels[i-1,j,k] != exterior:
            sd = np.array([-1, 0])
            sd_value = np.array([1, -1])
            return sd, sd_value
        else:
            sd = np.array([0])
            sd_value = np.array([0])
            return sd, sd_value
    elif labels[i-1,j,k] == exterior and labels[i+1,j,k] == exterior:
        sd = np.array([0])
        sd_value = np.array([0])
        return sd, sd_value
    elif labels[i-1,j,k] != exterior and labels[i+1,j,k] == exterior:
        sd = np.array([-1, 0])
        sd_value = np.array([1, -1])
        return sd, sd_value
    elif labels[i-1,j,k] == exterior and labels[i+1,j,k] != exterior:
        sd = np.array([0, 1])
        sd_value = np.array([-1, 1])
        return sd, sd_value
    else:
        sd = np.array([-1, 0, 1])
        sd_value = np.array([1, -2, 1])
        return sd, sd_value

def get_boundary_stencily(labels, i, j, k):
    if j == 0 and labels[i, j+1, k] != exterior: # lower part of dim_1, no j-1
        sd = np.array([0, 1])
        sd_value = np.array([-1, 1])
        return sd, sd_value
    elif j == 0 and labels[i, j+1, k] == exterior:
        sd = np.array([0])
        sd_value = np.array([0])
    elif j == labels.shape[1]-1 and labels[i, j-1, k] != exterior: # upper part of dim_1, no j+1
        sd = np.array([-1, 0])
        sd_value = np.array([1, -1])
    elif j == labels.shape[1]-1 and labels[i, j-1, k] == exterior:
        sd = np.array([0])
        sd_value = np.array([0])
    elif labels[i, j-1, k] == exterior and labels[i, j+1, k] == exterior: # mid-part of dim_1, both j-1 and j+1 exist
        sd = np.array([0])
        sd_value = np.array([0])
    elif labels[i, j-1, k] != exterior and labels[i, j+1, k] == exterior: # lower neighbour != exterior
        sd = np.array([-1, 0])
        sd_value = np.array([1, -1])
    elif labels[i, j-1, k] == exterior and labels[i, j+1, k] != exterior: # upper neighbour != exterior
        sd = np.array([0, 1])
        sd_value = np.array([-1, 1])
    else: # both neighbours != exterior
        sd = np.array([-1, 0, 1])
        sd_value = np.array([1, -2, 1])
    return sd, sd_value


def get_boundary_stencilz(labels, i, j, k, image_is_3D=False):
    if image_is_3D:
        if k == 0 and labels[i, j, k+1] != exterior:
            # If lower part of dim_1, no i-1
            sd = np.array([0, 1])
            sd_value = np.array([-1, 1])
            return sd, sd_value
        elif k == 0 and labels[i, j, k+1] == exterior:
            sd = np.array([0])
            sd_value = np.array([0])
        elif k == labels.shape[2]-1 and labels[i, j, k-1] != exterior:
            # If upper part of dim_1, no i+1
            sd = np.array([-1, 0])
            sd_value = np.array([1, -1])
        elif k == labels.shape[2]-1 and labels[i, j, k-1] == exterior:
            sd = np.array([0])
            sd_value = np.array([0])
        elif labels[i, j, k-1] == exterior and labels[i, j, k+1] == exterior:
            # If mid-part of dim_1, both i+1 and i-1 exist
            sd = np.array([0])
            sd_value = np.array([0])
        elif labels[i, j, k-1] != exterior and labels[i, j, k+1] == exterior:
            # If lower neighbour != exterior
            sd = np.array([-1, 0])
            sd_value = np.array([1, -1])
        elif labels[i, j, k-1] == exterior and labels[i, j, k+1] != exterior:
            # If upper neighbour != exterior
            sd = np.array([0, 1])
            sd_value = np.array([-1, 1])
        else:
            # If both neighbours != exterior
            sd = np.array([-1, 0, 1])
            sd_value = np.array([1, -2, 1])
    else:
        sd = np.array([0])
        sd_value = np.array([0])

    return sd, sd_value

def get_image_gradient(im, labels, pixdim):
    # Image type ...
    image_is_3D = len(im.shape) == 3
    if not image_is_3D:
        im_ = im[:,:,None]
        labels_ = labels[:,:,None]
        grad_image = np.zeros([im_.shape[0],im_.shape[1],2])
    else:
        im_ = im
        labels_ = labels
        grad_image = np.zeros([im_.shape[0],im_.shape[1],im_.shape[2],3])



    for i in range(im_.shape[0]):
        for j in range(im_.shape[1]):
            for k in range(im_.shape[2]):
                if labels_[i,j,k] == exterior:
                    continue  # If exterior, continue
                # If outlet or interior, get appropriate stencil values
                grad_x = get_gradient_x(im_, labels_, i, j, k)/pixdim[0]
                grad_y = get_gradient_y(im_, labels_, i, j, k)/pixdim[1]
                if image_is_3D:
                    grad_z = get_gradient_z(im_, labels_, i, j, k)/pixdim[2]
                    grad_image[i,j,k] = np.array([grad_x, grad_y, grad_z])
                else:
                    grad_image[i,j] = np.array([grad_x, grad_y])

    return grad_image


def get_gradient_x(image, labels, i, j, k):
    if i == 0:
        if labels[i+1,j,k] != exterior:
            grad_x = (image[i+1,j,k] - image[i,j,k])
            return grad_x
        else:
            grad_x = 0
            return grad_x
    elif i == labels.shape[0] - 1:
        if labels[i-1,j,k] != exterior:
            grad_x = (image[i,j,k] - image[i-1,j,k])
            return grad_x
        else:
            grad_x = 0
            return grad_x
    elif labels[i-1,j,k] == exterior and labels[i+1,j,k] == exterior:
        grad_x = 0
        return grad_x
    elif labels[i-1,j,k] != exterior and labels[i+1,j,k] == exterior:
        grad_x = (image[i,j,k] - image[i-1,j,k])
        return grad_x
    elif labels[i-1,j,k] == exterior and labels[i+1,j,k] != exterior:
        grad_x = (image[i+1,j,k] - image[i,j,k])
        return grad_x
    else:
        grad_x = (image[i+1,j,k] - image[i-1,j,k])/2
        return grad_x

def get_gradient_y(image, labels, i, j, k):
    if j == 0 and labels[i, j+1, k] != exterior: # lower part of dim_1, no j-1
        grad_y = (image[i,j+1,k] - image[i,j,k])
        return grad_y
    elif j == 0 and labels[i, j+1, k] == exterior:
        return 0
    elif j == labels.shape[1]-1 and labels[i, j-1, k] != exterior: # upper part of dim_1, no j+1
        grad_y = (image[i,j,k] - image[i,j-1,k])
        return grad_y
    elif j == labels.shape[1]-1 and labels[i, j-1, k] == exterior:
        return 0
    elif labels[i, j-1, k] == exterior and labels[i, j+1, k] == exterior: # mid-part of dim_1, both j-1 and j+1 exist
        return 0
    elif labels[i, j-1, k] != exterior and labels[i, j+1, k] == exterior: # lower neighbour != exterior
        grad_y = (image[i,j,k] - image[i,j-1,k])
        return grad_y
    elif labels[i, j-1, k] == exterior and labels[i, j+1, k] != exterior: # upper neighbour != exterior
        grad_y = (image[i,j+1,k] - image[i,j,k])
        return grad_y
    else: # both neighbours != exterior
        grad_y = (image[i,j+1,k] - image[i,j-1,k])/2
        return grad_y


def get_gradient_z(image, labels, i, j, k, image_is_3D=False):
    if image_is_3D:
        if k == 0 and labels[i, j, k+1] != exterior:
            grad_z = (image[i,j,k+1] - image[i,j,k])
            return grad_z
        elif k == 0 and labels[i, j, k+1] == exterior:
            return 0
        elif k == labels.shape[2]-1 and labels[i, j, k-1] != exterior:
            # If upper part of dim_1, no i+1
            grad_z = (image[i,j,k] - image[i,j,k-1])
            return grad_z
        elif k == labels.shape[2]-1 and labels[i, j, k-1] == exterior:
            return 0
        elif labels[i, j, k-1] == exterior and labels[i, j, k+1] == exterior:
            return 0
        elif labels[i, j, k-1] != exterior and labels[i, j, k+1] == exterior:
            grad_z = (image[i,j,k] - image[i,j,k-1])
            return grad_z
        elif labels[i, j, k-1] == exterior and labels[i, j, k+1] != exterior:
            grad_z = (image[i,j,k+1] - image[i,j,k])
            return grad_z
        else:
            grad_z = (image[i,j,k+1] - image[i,j,k-1])/2
            return grad_z
    else:
        grad_z = 0

    return grad_z

def uvc_get_TP_ablock(im, tv, ind, labels, n, pixdim):
    # Image type ...
    image_is_3D = len(im.shape) == 3
    if not image_is_3D:
        im_ = im[:,:,None]
        ind_ = ind[:,:,None]
        labels_ = labels[:,:,None]
    else:
        im_ = im
        ind_ = ind
        labels_ = labels

    # Constructing the block matrix using position (I,J) and value (V)
    I = np.ones(6*n, dtype=int)
    J = np.ones(6*n, dtype=int)
    V = np.zeros(6*n)

    # Building the block entries ...
    # loop over dimensions
    it = 0
    for i in range(im_.shape[0]):
        for j in range(im_.shape[1]):
            for k in range(im_.shape[2]):
                row = ind_[i,j,k]  # Get row numbering based on indexing ind_
                if labels_[i,j,k] == exterior:
                    continue  # If exterior, continue
                elif labels_[i,j,k] == inlet or labels_[i,j,k] == outlet:  # If inlet or wall
                    I[it] = row
                    J[it] = row
                    V[it] = 1
                    it += 1
                    continue
                # If outlet or interior, get appropriate stencil values
                sx, sx_value, sy, sy_value = get_TP_Ablock_stencil(labels_, i, j, k)
                # Set x-direction pos (I,J) and values (V)
                I[it:it+len(sx)] = row
                J[it:it+len(sx)] = ind_[i+sx, j, k]
                V[it:it+len(sx)] = (sx_value/pixdim[0]*tv[i,j,0])
                it += len(sx)
                # Set y-direction pos (I,J) and values (V)
                I[it:it+len(sy)] = row
                J[it:it+len(sy)] = ind_[i, j+sy, k]
                V[it:it+len(sy)] = (sy_value/pixdim[1]*tv[i,j,1])
                it += len(sy)
                if image_is_3D:
                    raise 'Not implemented'

    # Generate sparse matrix A. NOTE: Repeated (I,J) gets added (V) automatic!
    A = sparse.coo_matrix((V, (I, J)), shape=(n, n))
    A = A.tocsr()

    # Check whether max(row) < 1e-8, if so --> A(row,row) = 1, dirichlet(row) = 1
    vector = np.max(A, axis=1).todense()
    idx = np.where(vector < 1e-8)[0]
    A[idx,idx] = 1

    return A

def get_TP_Ablock_stencil(labels, i, j, k):
    if labels[i, j, k] == interior: # If interior...
        sx = np.array([-1, 1])
        sx_value = np.array([-1, 1])/2
        sy = sx
        sy_value = sx_value
    elif labels[i, j, k] == outlet:
        raise ValueError('stencil should not apply to walls')
    elif labels[i, j, k] == inlet:
        raise ValueError('stencil should not apply to inlet')
    elif labels[i, j, k] == exterior:
        raise ValueError('stencil should not apply to exterior')
    elif labels[i, j, k] == walls: # Special stencils for boundary
        sx, sx_value = get_boundary_TP_stencilx(labels, i, j, k)
        sy, sy_value = get_boundary_TP_stencily(labels, i, j, k)

    return sx, sx_value, sy, sy_value


def get_boundary_TP_stencilx(labels, i, j, k):
    if i == 0:
        if labels[i+1,j,k] != exterior:
            sd = np.array([0, 1])
            sd_value = np.array([-1, 1])
            return sd, sd_value
        else:
            sd = np.array([0])
            sd_value = np.array([0])
            return sd, sd_value
    elif i == labels.shape[0] - 1:
        if labels[i-1,j,k] != exterior:
            sd = np.array([-1, 0])
            sd_value = np.array([1, -1])
            return sd, sd_value
        else:
            sd = np.array([0])
            sd_value = np.array([0])
            return sd, sd_value
    elif labels[i-1,j,k] == exterior and labels[i+1,j,k] == exterior:
        sd = np.array([0])
        sd_value = np.array([0])
        return sd, sd_value
    elif labels[i-1,j,k] != exterior and labels[i+1,j,k] == exterior:
        sd = np.array([-1, 0])
        sd_value = np.array([1, -1])
        return sd, sd_value
    elif labels[i-1,j,k] == exterior and labels[i+1,j,k] != exterior:
        sd = np.array([0, 1])
        sd_value = np.array([-1, 1])
        return sd, sd_value
    else:
        sd = np.array([-1, 1])
        sd_value = np.array([-1, 1])/2
        return sd, sd_value

def get_boundary_TP_stencily(labels, i, j, k):
    if j == 0 and labels[i, j+1, k] != exterior: # lower part of dim_1, no j-1
        sd = np.array([0, 1])
        sd_value = np.array([-1, 1])
        return sd, sd_value
    elif j == 0 and labels[i, j+1, k] == exterior:
        sd = np.array([0])
        sd_value = np.array([0])
    elif j == labels.shape[1]-1 and labels[i, j-1, k] != exterior: # upper part of dim_1, no j+1
        sd = np.array([-1, 0])
        sd_value = np.array([1, -1])
    elif j == labels.shape[1]-1 and labels[i, j-1, k] == exterior:
        sd = np.array([0])
        sd_value = np.array([0])
    elif labels[i, j-1, k] == exterior and labels[i, j+1, k] == exterior: # mid-part of dim_1, both j-1 and j+1 exist
        sd = np.array([0])
        sd_value = np.array([0])
    elif labels[i, j-1, k] != exterior and labels[i, j+1, k] == exterior: # lower neighbour != exterior
        sd = np.array([-1, 0])
        sd_value = np.array([1, -1])
    elif labels[i, j-1, k] == exterior and labels[i, j+1, k] != exterior: # upper neighbour != exterior
        sd = np.array([0, 1])
        sd_value = np.array([-1, 1])
    else: # both neighbours != exterior
        sd = np.array([-1, 1])
        sd_value = np.array([-1, 1])/2
    return sd, sd_value
