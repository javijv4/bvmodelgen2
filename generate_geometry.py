#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:26:30 2023

@author: Javiera Jilberto Vallejos
"""

from src.ImageData import ImageData

patients = ['VB-1']


# Inputs
frame = 0
sa_labels = {'lv': 2., 'rv': 1., 'lvbp': 3.}
la_labels = {'lv': 2., 'rv': 3., 'lvbp': 1.}

for patient in patients:
    path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/Images/'

    segs = {'sa': path + 'SA_seg',
                'la_4ch': path + 'LA_4CH_seg',
                'la_3ch': path + 'LA_3CH_seg',
                'la_2ch': path + 'LA_2CH_seg'}

    valves = {'la_4ch': path + 'LA_4CH_valves',
                'la_3ch': path + 'LA_3CH_valves',
                'la_2ch': path + 'LA_2CH_valves'}

    output_fldr = path + 'test_frame{:d}/'.format(frame)
    imgdata = ImageData(segs, sa_labels, la_labels, frame, output_fldr, autoclean=True)
    imgdata.align_slices(output_fldr, show=False)
    imgdata.generate_contours(valve_file_paths=valves, show=False)
    imgdata.template_fitting(rv_thickness=3, show=True)
