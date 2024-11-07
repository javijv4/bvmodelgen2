#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/10/21 11:14:52

@author: Javiera Jilberto Vallejos 
'''

import os
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from plot_functions import show_point_cloud
from scipy.spatial.transform import Rotation as R 

def Eidolon_affine(img):
    '''Given a Nibabel object `img', return the position, rotation, scaling, time start, and time offset values.'''
    hdr=dict(img.header)

    pixdim=hdr['pixdim']
    xyzt_units=hdr['xyzt_units']
    x=float(hdr['qoffset_x'])
    y=float(hdr['qoffset_y'])
    z=float(hdr['qoffset_z'])
    b=float(hdr['quatern_b'])
    c=float(hdr['quatern_c'])
    d=float(hdr['quatern_d'])
    toffset=float(hdr['toffset'])
    interval=float(pixdim[4])

    if interval==0.0 and len(img.shape)>=4 and img.shape[3]>1:
        interval=1.0

    qfac=float(pixdim[0]) or 1.0
    spacing=np.array([pixdim[1],pixdim[2],qfac*pixdim[3]])

    if int(hdr['qform_code'])>0:
        position=np.array([x,y,z])
        rot=np.array([-c,b,np.sqrt(max(0,1.0-(b*b+c*c+d*d))),-d])
    else:
        affine=img.get_affine()
        position=np.array([affine[0,3],affine[1,3],affine[2,3]])
        rmat=np.asarray([affine[0,:3]/-spacing.x(),affine[1,:3]/-spacing.y(),affine[2,:3]/spacing.z()])
        rot=np.array(*rmat.flatten().tolist())

    print(spacing)
    # convert from nifti-space to real space
    position=position*np.array([-1,-1,1])
    rot = R.from_quat(rot)
    rotz = R.from_rotvec(np.array([0,0,1])*np.pi/2)
    print((rot*rotz).as_quat())
    rot = (rot*rotz).as_matrix()

    # Multiply by the z spacing to get the correct scaling
    vector_x = rot@np.array([1,0,0])-rot@np.array([0,0,0])
    if not np.isclose(np.linalg.norm(vector_x),spacing[0]):
        vector_x /= np.linalg.norm(vector_x)
        vector_y = rot@np.array([0,1,0])-rot@np.array([0,0,0])
        vector_y /= np.linalg.norm(vector_y)
        vector_z = rot@np.array([0,0,1])-rot@np.array([0,0,0])
        vector_z /= np.linalg.norm(vector_z)    
        U = spacing[0]*np.outer(vector_x, vector_x) + spacing[1]*np.outer(vector_y, vector_y) + spacing[2]*np.outer(vector_z, vector_z)
        rot = U @ rot

    affine = np.eye(4)
    affine[:3, :3] = rot
    affine[:3, 3] = position

    print(np.linalg.norm(affine[:3,:3]@np.array([0,0,1])-affine[:3,:3]@np.array([0,0,0])))
    print(np.linalg.norm(affine[:3,:3]@np.array([0,1,0])-affine[:3,:3]@np.array([0,0,0])))
    return affine



fldr = '/home/jilberto/Downloads/LGE_Javi_check/LGE/'

seg_files = {'sa': fldr + 'sa_seg',
            'la_2ch': fldr + 'LA_2CH_seg',
            'la_3ch': fldr + 'LA_3CH_seg',
            'la_4ch': fldr + 'LA_4CH_seg',
            }

fig = go.Figure()

file = seg_files['sa']
img = nib.load(file + '.nii.gz')
data = img.get_fdata().astype(float)
affine = Eidolon_affine(img)
points_ijk = np.vstack(np.where(data)).T
points_xyz = nib.affines.apply_affine(affine, points_ijk)
show_point_cloud(points_xyz, fig=fig, opacity=0.5, size=5, label=os.path.basename(file).split('.')[0])

file = seg_files['la_2ch']
img = nib.load(file + '.nii.gz')
data = img.get_fdata().astype(float)
affine = Eidolon_affine(img)
points_ijk = np.vstack(np.where(data)).T
points_xyz = nib.affines.apply_affine(affine, points_ijk)
show_point_cloud(points_xyz, fig=fig, opacity=0.5, size=5, label=os.path.basename(file).split('.')[0])

fig.show()


