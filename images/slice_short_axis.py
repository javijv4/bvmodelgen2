#!/usr/bin/python



import nibabel as nib




img=nib.load('./SER00035_new_seg.nii.gz')

data=img.get_fdata()


zstart=2
zend=7

data_new=data[:,:,zstart:zend,:]


newimg=nib.Nifti1Image(data_new,affine=img.affine,header=img.header)


nib.save(newimg,'SER00035_seg_edited.nii')



