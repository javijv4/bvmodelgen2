#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:44:11 2023

@author: jjv
"""

import numpy as np
import SimpleITK as sitk
from skimage import img_as_ubyte, img_as_float

MAX_THREADS = 1
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(MAX_THREADS)

def set_sitk_threads(MAX_THREADS):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(MAX_THREADS)

def get_displacement_between_images(moving_, target_,  init_disp=None, return_warped=False):
    moving = sitk.GetImageFromArray(moving_.astype(np.float32))
    fixed = sitk.GetImageFromArray(target_.astype(np.float32))

    matcher = sitk.HistogramMatchingImageFilter()
    if fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(20)
    demons.SetStandardDeviations(5)

    metric = []
    def command_iteration(filter):
        metric.append(filter.GetMetric())

        # Check convergence
        if len(metric) > 10:
            if metric[-1] > metric[-2]:
                filter.StopRegistration()

    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
    if init_disp is None:
        displacementField = demons.Execute(fixed, moving)
    else:
        displacementField = demons.Execute(fixed, moving, init_disp)

    outTx = sitk.DisplacementFieldTransform(displacementField)
    displacementField = outTx.GetDisplacementField()

    displacement = sitk.GetArrayFromImage(displacementField)

    if return_warped:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)
        resampler.SetTransform(outTx)

        warped = resampler.Execute(moving)
        warped = sitk.GetArrayFromImage(warped).astype(float)

        return displacement, warped

    return displacement


def warp_image(image, displacement):
    disp_im = sitk.GetImageFromArray(displacement, isVector=True)
    reference = sitk.GetImageFromArray(image)

    outTx = sitk.DisplacementFieldTransform(disp_im)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    warped = resampler.Execute(reference)
    warped = sitk.Cast(sitk.RescaleIntensity(warped), sitk.sitkUInt8)
    warped = sitk.GetArrayFromImage(warped)

    return warped

# def interpolate_in_time()