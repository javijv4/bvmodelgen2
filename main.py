#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:26:30 2023

@author: Javiera Jilberto Vallejos
"""

import os
from src.ImageData import ImageData, compute_volume_from_contours

# Inputs
patient = 'ZS-11'
path = '/home/jilberto/Dropbox (University of Michigan)/Projects/Desmoplakin/Models/DSPPatients/' + patient + '/Images/'

sa_labels = {'lv': 2., 'rv': 1., 'lvbp': 3.}
la_labels = {'lv': 2., 'rv': 3., 'lvbp': 1.}
segs = {'sa': path + 'SA_seg',
            'la_4ch': path + 'LA_4CH_seg',
            'la_3ch': path + 'LA_3CH_seg',
            'la_2ch': path + 'LA_2CH_seg',
            'la_2chr': path + 'LA_2CHr_seg'}

valves = {'la_4ch': path + 'LA_4CH_valves',
            'la_3ch': path + 'LA_3CH_valves',
            'la_2ch': path + 'LA_2CH_valves',
            'la_2chr': path + 'LA_2CHr_valves'}

frame = 0
output_fldr = path + 'test_frame{:d}/'.format(frame)
imgdata = ImageData(segs, sa_labels, la_labels, frame, output_fldr)
imgdata.align_slices(output_fldr)
imgdata.generate_contours(valve_file_paths=valves, plot=True)
imgdata.template_fitting()
