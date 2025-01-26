#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:27:30 2023

@author: Javiera Jilberto Vallejos
"""
import os
import numpy as np
import nibabel as nib
import meshio as io

from masks2contours.m2c import CMRImage, correct_labels
from masks2contours import m2c, utils
from masks2contours import slicealign
from bvfitting import BiventricularModel, GPDataSet, ContourType

from niftiutils import readFromNIFTI
from masksutils import check_seg_valid
from plot_functions import plot_seg_files

import plot_functions as pf
from plotly.offline import  plot
import plotly.graph_objs as go

class ImageData:

    template_fitting_weights = {'apex_endo': 2, 'apex_epi': 2, 'mv': 0.5, 'tv': 1, 'av': 1, 'pv': 1,
                                'mv_phantom': 1.5, 'tv_phantom': 1., 'av_phantom': 1., 'pv_phantom': 1,
                                'rv_insert': 1.5,
                                'la_rv_endo': 3, 'la_rv_epi': 2, 'la_lv_endo': 2, 'la_lv_epi': 1,
                                'sa_lv_epi': 1, 'sa_lv_endo': 2, 'sa_rv_endo': 1, 'sa_rv_epi': 1}

    def __init__(self, seg_paths, sa_labels, la_labels, frame, output_fldr, autoclean=False):
        self.seg_paths = seg_paths
        self.sa_labels = sa_labels
        self.la_labels = la_labels
        self.frame = frame
        self.output_fldr = output_fldr
        if not os.path.exists(self.output_fldr): os.mkdir(self.output_fldr)

        # Data dictionaries
        self.seg_files = {}
        self.cmrs = {}
        self.translations = {}
        self.slices = []
        self.contours = []

        # Load segmentations
        self.has_la = False
        self.load_segmentations(seg_paths, frame, plot=False, autoclean=autoclean, force=True)


    def prepare_segmentations(self, seg_paths, frame, autoclean=False):
        seg_files = {}
        isvalid = {}
        for view, seg_path in seg_paths.items():
            try:
                data, affine, _, header = readFromNIFTI(seg_path, frame, correct_ras=False)
            except:
                print('Segmentation ' + seg_path + ' not found')
                continue

            # Add third dimension if missing
            if data.ndim < 3:
                data = data[:, :, np.newaxis]

            # Grab correct labels
            if view == 'sa':
                labels = self.sa_labels
            else:
                labels = self.la_labels

            new_data, isvalid[view] = check_seg_valid(view, data, labels, autoclean=autoclean)

            new_seg = nib.Nifti1Image(new_data, affine, header=header)
            nib.save(new_seg, self.output_fldr + view.upper() + '.nii.gz')
            seg_files[view] = self.output_fldr + view.upper() + '.nii.gz'

        # Check if all segmentations are valid
        sa_segs = np.sum(['sa' in n for n in isvalid.keys()])
        if (sa_segs>0):
            save = True
        else:
            save = False
        la_segs = np.sum(['la' in n for n in isvalid.keys()])
        if (la_segs>0):
            self.has_la = True

        return seg_files, save


    def load_segmentations(self, seg_paths, frame, force=False, plot=True, autoclean=False):
        if not force:
            # Check if segmentations are already saved
            seg_files = {}
            for view, _ in seg_paths.items():
                if os.path.exists(self.output_fldr + view.upper() + '.nii.gz'):
                    seg_files[view] = self.output_fldr + view.upper() + '.nii.gz'

            # If SA and at least one LA is found then load directly
            if 'sa' in seg_files.keys() and ('la_2ch' in seg_files.keys()
                                             or 'la_3ch' in seg_files.keys()
                                             or 'la_4ch' in seg_files.keys()):
                force = False
            else:
                force = True

        if force:
            print('Work segmentations not found. Creating new ones...')
            seg_files, valid = self.prepare_segmentations(seg_paths, frame, autoclean)
            if not valid:
                raise ValueError('Some segmentations are invalid. Check the data.')
            else:
                print('Segmentations created successfully.')

        # Visualize segmentations
        if plot:
            plot_seg_files(list(seg_files.values()))


        # Load segmentations
        print('Loading segmentations...')
        self.seg_files = seg_files

        # Load cmrs
        cmrs = {}
        slices = []
        for view in seg_files.keys():
            if seg_files[view] is None: continue
            try:
                cmr = CMRImage(view, seg_files[view], frame=frame)
            except:
                print('{} not found'.format(seg_files[view]))
                continue
            print('Loaded ' + view.upper() + ' segmentation.')

            print('Extracting slices for segmentation...')
            slices += cmr.extract_slices(defseptum=True)
            cmrs[view] = cmr

            print()

        self.cmrs = cmrs
        self.slices = slices


    def align_slices(self, translation_files_path=None, method=2, show=False):
        fig = pf.plot_slices(self.slices)
        if show:
            fig.show()
        pf.save_figure(self.output_fldr + 'init_align.html', fig)
        
        if translation_files_path is None:
            # Compute alignment
            print('Calculating alignment using Sinclair algorithm...')
            slicealign.find_SA_initial_guess(self.slices)
            if method == 2:
                slicealign.optimize_stack_translation2(self.slices, nit=100)
            elif method == 3:
                slicealign.optimize_stack_translation3(self.slices, nit=100)
            translations = slicealign.save_translations(self.output_fldr, self.cmrs, self.slices)

        else:
            print('Loading translations from file...')
            translations = {}
            found = 0
            for view in  ['sa', 'la_2ch', 'la_2chr', 'la_3ch', 'la_4ch']:
                try:
                    translations[view] = np.load(translation_files_path + view.upper() + '_translation.npy')
                    found += 1
                except:
                    print('Translation file for ' + view + ' not found.')
                    continue

            if found == 0:
                # Compute alignment
                print('No translation file found, calculating alignment using Sinclair algorithm...')
                slicealign.find_SA_initial_guess(self.slices)
                if method == 2:
                    slicealign.optimize_stack_translation2(self.slices, nit=100)
                elif method == 3:
                    slicealign.optimize_stack_translation3(self.slices, nit=100)
                translations = slicealign.save_translations(self.output_fldr, self.cmrs, self.slices)


            slices = []
            for slc in self.slices:
                view = slc.cmr.view
                trans = translations[view][slc.slice_number]
                slc.accumulated_translation = trans
                if not np.all(trans == 0):
                    slices.append(slc)
            self.slices = slices
        self.translations = translations

        fig = pf.plot_slices(self.slices)
        if show:
            fig.show()
        pf.save_figure(self.output_fldr + 'tran_salign.html', fig)


    def generate_contours(self, valve_file_paths={}, downsample=3, use_frames=None, show=True, min_length=15):
        print('Generating contours from slices...')
        contours = []
        for slc in self.slices:
            if use_frames is None:
                contours += slc.tocontours(downsample)
            else:
                if slc.slice_number in use_frames:
                    contours += slc.tocontours(downsample)

        m2c.modify_sa_weights(contours)

        # Add landmarks such as valves and apex
        # If valve files are provided
        if len(valve_file_paths) > 0:
            print('Adding valves to contours...')
            valves = {}
            for view in ['la_2ch', 'la_2chr', 'la_3ch', 'la_4ch']:
                try:
                    valves[view], _, _, _ = readFromNIFTI(valve_file_paths[view], self.frame)
                except:
                    continue
                print('Loading valves from ' + view.upper() + ' view')
            m2c.add_valves(contours, valves, self.cmrs, self.translations)
            m2c.add_apex(contours, self.cmrs)
            m2c.add_rv_apex(contours, self.cmrs)
            m2c.remove_base_nodes(contours, min_length=min_length)

        else:
            if self.has_la:
                print('Adding valves from LA views...')
                apex, mv = m2c.find_apex_mv_estimate(contours)
                m2c.remove_base_nodes(contours, apex, mv, min_length=min_length)
                m2c.find_valves_from_la(contours)
                m2c.add_apex(contours, self.cmrs)
                m2c.add_rv_apex(contours, self.cmrs)


        # Save results
        m2c.writeResults(self.output_fldr + 'contours.txt', contours)

        self.contours = contours
        # Visualize
        fig = pf.plot_contours(contours, background=True)
        pf.save_figure(self.output_fldr + 'contours.html', fig)
        if show:
            fig.show()

        self.vertex_contours = pf.contours2vertex(contours)
        io.write(self.output_fldr + 'contours.vtu', self.vertex_contours)


    def template_fitting(self, load_control_points=None, weight_GP=1, low_smoothing_weight=10, transmural_weight=20, rv_thickness=3, mesh_subdivisions=2, show=True):
        # Filename containing guide points (from contours/masks)
        filename = os.path.join(self.output_fldr, 'contours.txt')
        dataset = GPDataSet(filename)

        # Loads biventricular control_mesh
        model_path = "src/bvfitting/template" # folder of the control mesh
        bvmodel = BiventricularModel(model_path, filemod='_mod')

        if load_control_points:
            bvmodel.control_mesh = np.load(load_control_points)
            bvmodel.et_pos = np.linalg.multi_dot([bvmodel.matrix,
                                                        bvmodel.control_mesh])

        else:
            # Procrustes alignment
            bvmodel.update_pose_and_scale(dataset)

            # perform a stiff fit
            displacement, err = bvmodel.lls_fit_model(weight_GP,dataset,1e10)
            bvmodel.control_mesh = np.add(bvmodel.control_mesh,
                                                    displacement)
            bvmodel.et_pos = np.linalg.multi_dot([bvmodel.matrix,
                                                            bvmodel.control_mesh])


        # Generates 30 BP_point phantom points and 30 tricuspid phantom points.
        # We do not have any pulmonary points or aortic points in our dataset but if you do,
        # I recommend you to do the same.
        mitral_points = dataset.create_valve_phantom_points(30, ContourType.MITRAL_VALVE)
        tri_points = dataset.create_valve_phantom_points(30, ContourType.TRICUSPID_VALVE)
        aorta_points = dataset.create_valve_phantom_points(10, ContourType.AORTA_VALVE)
        # pulmonary_points = data_set.create_valve_phantom_points(20, ContourType.PULMONARY_VALVE)


        # Generates RV epicardial point if they have not been contoured
        rv_epi_points,rv_epi_contour, rv_epi_slice = dataset.create_rv_epicardium(
            rv_thickness=rv_thickness)

        # Plot rigid fit
        contourPlots = dataset.PlotDataSet([ContourType.LAX_RA,
                    ContourType.SAX_RV_FREEWALL, ContourType.LAX_RV_FREEWALL,
                    ContourType.SAX_RV_SEPTUM, ContourType.LAX_RV_SEPTUM,
                    ContourType.SAX_LV_ENDOCARDIAL,
                    ContourType.SAX_LV_EPICARDIAL, ContourType.RV_INSERT,
                    ContourType.APEX_ENDO_POINT, ContourType.APEX_EPI_POINT,
                    ContourType.MITRAL_VALVE, ContourType.TRICUSPID_VALVE,
                    ContourType.SAX_RV_EPICARDIAL, ContourType.LAX_RV_EPICARDIAL,
                    ContourType.LAX_LV_ENDOCARDIAL, ContourType.LAX_LV_EPICARDIAL,
                    ContourType.LAX_RV_EPICARDIAL, ContourType.SAX_RV_OUTLET,
                    ContourType.PULMONARY_VALVE, ContourType.AORTA_VALVE,
                    ContourType.AORTA_PHANTOM, ContourType.MITRAL_PHANTOM,
                    ContourType.TRICUSPID_PHANTOM,
                    ])
        model = bvmodel.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)",
                                    "Initial model", "all")
        data = model + contourPlots

        # Exmple on how to set different weights for different points group
        dataset.assign_weights(self.template_fitting_weights)


        # 'Stiff' fit - implicit diffeomorphic constraints
        bvmodel.MultiThreadSmoothingED(weight_GP, dataset)

        model = bvmodel.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)","Initial model","all")
        data = model + contourPlots
        plot(go.Figure(data),filename=os.path.join(self.output_fldr, 'step1_fitted.html'), auto_open=False)


        # 'Soft' fit - explicit diffeomorphic constraints
        bvmodel.SolveProblemCVXOPT(dataset,weight_GP,low_smoothing_weight,transmural_weight)

        model = bvmodel.PlotSurface("rgb(0,127,0)", "rgb(0,0,127)", "rgb(127,0,0)","Initial model","all")
        data = model + contourPlots
        fig = go.Figure(data)
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
        plot(fig,filename=os.path.join(self.output_fldr, 'step2_fitted.html'), auto_open=show)

        # Save .stl and control points
        bvmesh, valve_mesh, septum_mesh = bvmodel.get_bv_surface_mesh(subdivisions=mesh_subdivisions)
        io.write(self.output_fldr + 'bv_surface.stl', bvmesh)
        io.write(self.output_fldr + 'valve_surfaces.stl', valve_mesh)
        io.write(self.output_fldr + 'septum_surface.stl', septum_mesh)

        # Save control points
        np.save(self.output_fldr + 'control_points.npy', bvmodel.control_mesh)

        self.bvmodel = bvmodel
        self.dataset = dataset

    def clean_folder(self):
        import shutil
        shutil.rmtree(self.output_fldr)
        os.mkdir(self.output_fldr)
        print('Folder cleaned.')


def compute_volume_from_sa_cmr(cmr):
    pass


def compute_volume_from_contours(contours, which='lvendo', use_la_for_extrapolation=True, return_areas=False):
    # Grab contours
    sa_contours = []
    sa_septum = []
    la_contours = []
    mv_points = []

    for ctr in contours:
        if ctr.view == 'sa' and ctr.ctype == which:
            sa_contours.append(ctr)
        elif ctr.view == 'sa' and ctr.ctype == 'rvsep':
            sa_septum.append(ctr)
        elif 'la' in ctr.view and ctr.ctype == which:
            la_contours.append(ctr)
        elif ctr.ctype == 'apexendo':
            apex_point = ctr.points
        if (ctr.ctype == 'mv'):
            mv_points.append(ctr.points)
        elif (ctr.ctype == 'tv'):
            mv_points.append(ctr.points)
        # else:
        #     if (ctr.ctype == 'tv') and  (which == 'lvendo'):
        #         mv_points.append(ctr.points)
        #     elif (ctr.ctype == 'mv') and  (which == 'rvendo'):
        #         mv_points.append(ctr.points)

    # If RV endo I need to merge the septum with the contours
    if which == 'rvendo':
        new_contours = []
        for sa_ctr in sa_contours:
            found_septum = False
            for sep_ctr in sa_septum:
                if sa_ctr.slice == sep_ctr.slice:
                    insert_points = sep_ctr.points[np.array([0, -1])]

                    # Find point in sa_ctr that is closest to the insert_points
                    dist1 = np.linalg.norm(sa_ctr.points - insert_points[0], axis=1)
                    dist2 = np.linalg.norm(sa_ctr.points - insert_points[1], axis=1)

                    insert1 = np.argmin(dist1)
                    insert2 = np.argmin(dist2)

                    # Need to check wheter the septum points start in insert1 or insert2
                    if np.linalg.norm(sep_ctr.points[0] - sa_ctr.points[insert1]) < np.linalg.norm(sep_ctr.points[0] - sa_ctr.points[insert2]):
                        sep_points = sep_ctr.points[::-1]
                    else:
                        sep_points = sep_ctr.points


                    sa_points = np.vstack([sa_ctr.points[:insert1], sep_points, sa_ctr.points[insert2+1:]])

                    new_ctr = m2c.CMRContour(sa_points, 'rvendo', sa_ctr.slice, 'sa', sa_ctr.normal)
                    new_contours.append(new_ctr)
                    found_septum = True
                    break
            if found_septum == False:
                new_contours.append(sa_ctr)
        sa_contours = new_contours


    # if which == 'rvendo':
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     for contour in sa_contours:
    #         points = contour.points
    #         ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r')

    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')

    #     plt.show()

    # Calculate sa_area
    sa_areas = []
    sa_midpoints = []
    for ctr in sa_contours:
        sa_points = ctr.points
        sa_vector = ctr.normal
        sa_area = utils.calculate_area_of_polygon_3d(sa_points, sa_vector)

        # Save in list
        sa_areas.append(sa_area)
        sa_midpoints.append(np.mean(sa_points, axis=0))


    # Calculate valve centroid
    mv_points = np.vstack(mv_points)
    mv_centroid = np.mean(mv_points, axis=0)

    # Grab la points
    la_points = np.vstack([ctr.points for ctr in la_contours])

    # Check what SA slice is closer to the apex
    apex_distance = np.linalg.norm(sa_midpoints - apex_point, axis=1)
    closest_sa_slice = np.argmin(apex_distance)
    if closest_sa_slice != 0:   # flip
        sa_midpoints = sa_midpoints[::-1]
        sa_areas = sa_areas[::-1]

    # Check that the sa_vector points towards the MV
    if np.dot(mv_centroid - apex_point, sa_vector) < 0:
        sa_vector = -sa_vector

    # Check that all slices are between apex and mv
    dist_between_slices = np.linalg.norm(sa_midpoints[0] - sa_midpoints[1])

    sa_midpoints = np.array(sa_midpoints)
    sa_areas = np.array(sa_areas)
    val = (sa_midpoints - mv_centroid)@sa_vector.T
    sa_midpoints = sa_midpoints[val < 0]
    sa_areas = sa_areas[val < 0]

    val = (sa_midpoints - apex_point)@sa_vector.T
    sa_midpoints = sa_midpoints[val > 0]
    sa_areas = sa_areas[val > 0]

    sa_midpoints = sa_midpoints.tolist()
    sa_areas = sa_areas.tolist()

    # Slice the LA views up to the MV centroid
    if use_la_for_extrapolation:
        mv_vector = mv_centroid - sa_midpoints[-1]
        mv_vector = mv_vector / np.linalg.norm(mv_vector)

        while True:
            new_origin = sa_midpoints[-1] + dist_between_slices * sa_vector

            # Check if the new midpoint is over the MV centroid
            if np.dot(new_origin - mv_centroid, sa_vector) < 0:
                # Find intersection between the new slice and the LA points
                dist_plane = slicealign.point_plane_intersection(la_points, new_origin, sa_vector)
                inter_points = np.where(np.abs(dist_plane) < dist_between_slices/5)[0]

                points = la_points[inter_points]
                centroid = dist_between_slices/(np.dot(sa_vector, mv_vector)) * mv_vector + sa_midpoints[-1]
                radius = np.linalg.norm(points - centroid, axis=1)
                sa_midpoints.append(centroid)
                sa_areas.append(np.pi * np.mean(radius)**2)
            else:
                break

        # Slice the LA views down to the Apex
        apex_vector = apex_point - sa_midpoints[0]
        apex_vector = apex_vector / np.linalg.norm(apex_vector)
        while True:
            new_origin = sa_midpoints[0] - dist_between_slices * sa_vector

            # Check if the new midpoint is below the apex
            if np.dot(new_origin - apex_point, sa_vector) > 0:
                # Find intersection between the new slice and the LA points
                dist_plane = slicealign.point_plane_intersection(la_points, new_origin, sa_vector)
                inter_points = np.where(np.abs(dist_plane) < dist_between_slices/5)[0]

                points = la_points[inter_points]
                centroid = -dist_between_slices/(np.dot(sa_vector, apex_vector)) * apex_vector + sa_midpoints[0]
                radius = np.linalg.norm(points - centroid, axis=1)
                sa_midpoints = [centroid] + sa_midpoints
                sa_areas = [np.pi * np.mean(radius)**2] + sa_areas
            else:
                break

    sa_midpoints = np.array(sa_midpoints)
    sa_areas = np.array(sa_areas)

    # Add apex
    sa_midpoints = np.append(apex_point[None], sa_midpoints, axis=0)
    sa_areas = np.append(0, sa_areas)

    # Add mv
    sa_midpoints = np.append(sa_midpoints, mv_centroid[None], axis=0)
    sa_areas = np.append(sa_areas, sa_areas[-1])


    # Compute the long axis coordinate
    sa_z_coord = np.dot(sa_midpoints - sa_midpoints[0], sa_vector)

    # Compute volume using the trapezoidal rule
    sa_volume = np.trapz(sa_areas, sa_z_coord)

    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot SA contours
    # for contour in contours:
    #     points = contour.points
    #     if contour.ctype == 'lvendo':
    #         if 'la' in contour.view:
    #             ax.plot(points[:, 0], points[:, 1], points[:, 2], 'bo')
    #         elif 'sa' in contour.view:
    #             ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r')


    # # Plot centroids
    # ax.scatter(sa_midpoints[:, 0], sa_midpoints[:, 1], sa_midpoints[:, 2], c='g', marker='o')

    # # Plot apex
    # ax.scatter(apex_point[0], apex_point[1], apex_point[2], c='m', marker='o')

    # # Plot mv
    # ax.scatter(mv_points[:, 0], mv_points[:, 1], mv_points[:, 2], c='y', marker='o')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()



    if return_areas:
        return sa_volume, sa_areas, sa_z_coord
    else:
        return sa_volume
