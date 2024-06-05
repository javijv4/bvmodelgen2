#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2024/06/02 14:00:30

@author: Javiera Jilberto Vallejos
'''

from matplotlib import pyplot as plt
import numpy as np
import nibabel as nib
from interpfuncs import get_displacement_between_images, warp_image
from matplotlib.widgets import Slider
from tqdm import tqdm
from skimage import morphology

#Inputs
patient = 'AV-19'
img_path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/{}/Images/'.format(patient)

la_img = nib.load(img_path + 'LA_4CH.nii.gz')
la_data = la_img.get_fdata()
la_seg = nib.load(img_path + 'LA_4CH_seg_la.nii.gz')
la_seg_data = la_seg.get_fdata()
la_seg_data = (la_seg_data == 1).astype(int)

la_dt = la_img.header['pixdim'][4]
la_cycle_time = la_dt * la_img.header['dim'][4]

sa_img = nib.load(img_path + 'SA.nii.gz')
sa_dt = sa_img.header['pixdim'][4]
sa_cycle_time = sa_dt * sa_img.header['dim'][4]

cycle_time = np.max([sa_cycle_time, la_cycle_time])
la_dt = la_dt*cycle_time/la_cycle_time

sa_nframes = sa_img.header['dim'][4]
la_nframes = la_img.header['dim'][4]

sa_timepoints = np.linspace(0, cycle_time, sa_nframes+1, endpoint=True)
la_timepoints = np.linspace(0, cycle_time, la_nframes+1, endpoint=True)


la_interp_imgs = np.zeros([la_data.shape[0], la_data.shape[1], la_data.shape[2], sa_nframes])
la_interp_seg = np.zeros([la_data.shape[0], la_data.shape[1], la_data.shape[2], sa_nframes], dtype=int)
for sa_frame in tqdm(range(sa_nframes)):
    # Find the interval where the frame is in the la_timepoints
    interval = 0, 0
    for i in range(len(la_timepoints)-1):
        if la_timepoints[i] <= sa_timepoints[sa_frame] <= la_timepoints[i+1]:
            interval = i, i+1
            break

    time_up = la_timepoints[interval[1]]
    time_dn = la_timepoints[interval[0]]
    alpha = (sa_timepoints[sa_frame] - time_dn) / (time_up - time_dn)

    if interval[1] > la_nframes-1:
        interval = la_nframes-1, la_nframes-1
        alpha = 0


    img_up = la_data[:,:,:,interval[1]].squeeze()
    img_dn = la_data[:,:,:,interval[0]].squeeze()

    disp_up = get_displacement_between_images(img_up, img_dn)
    disp_dn = get_displacement_between_images(img_dn, img_up)

    warped_dn = warp_image(img_dn, disp_up*alpha)
    warped_up = warp_image(img_up, disp_dn*(1-alpha))

    warped_img = warped_up*alpha + warped_dn*(1-alpha)
    la_interp_imgs[:,:,0,sa_frame] = warped_img

    seg_dn = la_seg_data[:,:,:,interval[0]].squeeze()
    seg_up = la_seg_data[:,:,:,interval[1]].squeeze()

    warped_dn = warp_image(seg_dn, disp_up*alpha)
    warped_up = warp_image(seg_up, disp_dn*(1-alpha))
    seg = (warped_up*alpha + warped_dn*(1-alpha) > 0).astype(int)
    seg = morphology.binary_opening(seg, footprint=morphology.disk(2))
    la_interp_seg[:,:,0,sa_frame] = seg

# Save the interpolated images
img = nib.Nifti1Image(la_interp_imgs, la_img.affine, la_img.header)
nib.save(img, img_path + 'LA_4CH_interp2sa.nii.gz')

img = nib.Nifti1Image(la_interp_seg, la_seg.affine, la_seg.header)
nib.save(img, img_path + 'LA_4CH_seg_interp2sa.nii.gz')

# plt.figure(1, clear=True)
# plt.plot(np.arange(len(sa_timepoints)), sa_timepoints, 'rx')
# plt.plot(np.arange(len(la_timepoints)), la_timepoints, 'bx')
# plt.savefig('check.png')

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

# ax1.imshow(img_up, cmap='gray')
# ax1.set_title('img_up')

# ax2.imshow(img_dn, cmap='gray')
# ax2.set_title('img_dn')

# ax3.imshow(warped_img, cmap='gray')
# ax3.set_title('warped_img')

# plt.tight_layout()
# plt.show()


fig, ax = plt.subplots(num=1, clear=True)
plt.subplots_adjust(bottom=0.25)

img = la_interp_seg
nframes = img.shape[3]
# Initialize the image plot
img_plot = ax.imshow(img[:, :, 0, 0], cmap='gray')

# Create a slider for selecting the frame
frame_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
frame_slider = Slider(frame_slider_ax, 'Frame', 0, nframes-1, valinit=0, valstep=1)

# Update the image plot when the slider value changes
def update_frame(val):
    frame = int(frame_slider.val)
    img_plot.set_data(img[:, :, 0, frame])
    fig.canvas.draw_idle()

frame_slider.on_changed(update_frame)

plt.show()