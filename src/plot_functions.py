#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:07:18 2022

@author: Javiera Jilberto Vallejos
"""
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

class IndexTracker:  # https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html
    def __init__(self, ax, X, cmap='viridis', vmin=None, vmax=None, levels=None):
        self.index = 0
        self.X = X
        self.ax = ax
        self.im = ax.imshow(self.X[:, :, self.index], cmap = cmap, vmin = vmin, vmax=vmax)
        self.update()

    def on_scroll(self, event):
        increment = 1 if event.button == 'up' else -1
        max_index = self.X.shape[-1] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.index])
        self.ax.set_title(
            f'Use scroll wheel to navigate\nindex {self.index}')
        self.im.axes.figure.canvas.draw()


def visualize_coord(image):
    fig, ax = plt.subplots()
    tracker = IndexTracker(ax, image)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()


pio.renderers.default='browser'

def show_mesh(vertices, triangles, fig=None, color='blue', opacity=1.0):
    if fig is None:
        fig = go.Figure()
    fig.add_mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                                     i=triangles[:,0], j=triangles[:,1], k=triangles[:,2],
                                     color=color, opacity=opacity)
    fig.write_html('mesh.html')

    return fig

def show_plane(point, normal, size = 5, fig=None, showscale=False):
    if fig is None:
        fig = go.Figure()

    xlim=(-size, size)
    ylim=(-size, size)
    # Create a grid of x,y values
    xx, yy = np.meshgrid(np.linspace(xlim[0]+point[0], xlim[1]+point[0], 11),
                         np.linspace(ylim[0]+point[1], ylim[1]+point[1], 11))
    # Calculate z values for the plane using the plane equation
    zz = (-normal[0] * xx - normal[1] * yy + np.dot(normal, point)) / normal[2]
    # Create a 3D surface plot
    colorscale = [[0, 'rgb(255,0,0)'],
              [1, 'rgb(255,0,0)']]
    fig.add_surface(x=xx, y=yy, z=zz, colorscale=colorscale, opacity=0.5, showscale=showscale)
    return fig


def show_point_cloud(points, fig=None, color=None, size=10, cmap='Viridis',
                     opacity=1, marker_symbol='circle', label=None, showscale=False,
                     cmin = None, cmax = None):
    if fig is None:
        fig = go.Figure()
    if len(points.shape) == 1:
        points = points[None]
    fig.add_scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers',
                      marker=dict(
                                    color=color,
                                    size=size,
                                    colorscale=cmap,
                                    opacity=opacity,
                                    symbol=marker_symbol,
                                    showscale=showscale,
                                    cmin = cmin,
                                    cmax = cmax
                                ),
                      name = label
        )
    return fig


def get_scatter3d(points, color=None, size=10, cmap='Viridis', opacity=1):
    scat = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers',
                      marker=dict(
                                    color=color,
                                    size=size,
                                    colorscale=cmap,
                                    opacity=opacity
                                ))
    return scat


def show_line(p0, normal, length=1, fig=None, color=None):
    if fig is None:
        fig = go.Figure()
    l = np.linspace(0,length,100)
    points = p0 + normal*l[:,None]
    if color is not None:
        line = dict(color=color)
    else:
        line = None
    fig.add_scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='lines', line=line)
    return fig


def save_figure(fname, fig):
    fig.write_html(fname)


def plot_contours(contours, background=True):
    colors = {'lvendo': 'blue', 'lvepi': 'red', 'rvendo': 'green', 'rvsep': 'cyan',
              'rvinsert': 'black', 'mv': 'yellow', 'av': 'purple', 'tv': 'magenta',
              'apexendo': 'black', 'apexepi': 'black', 'rvapex': 'black'}
    sizes = {'lvendo': 5, 'lvepi': 5, 'rvendo': 5, 'rvsep': 5,
              'rvinsert': 10, 'mv': 10, 'av': 10, 'tv': 10,
              'apexendo': 10, 'apexepi': 10, 'rvapex': 10}
    ctype = {}
    for ctr in contours:
        try:
            ctype[ctr.ctype].append(ctr.points)
        except:
            ctype[ctr.ctype] = [ctr.points]

    for key in ctype.keys():
        ctype[key] = np.vstack(ctype[key])

    fig = go.Figure()
    for key in ctype.keys():
        show_point_cloud(ctype[key], fig=fig, opacity=0.9, color=colors[key], label=key, size=sizes[key])

    if background:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    return fig


def contours2vertex(contours):
    import meshio as io
    ctype = {}
    for ctr in contours:
        try:
            ctype[ctr.ctype].append(ctr.points)
        except:
            ctype[ctr.ctype] = [ctr.points]

    for key in ctype.keys():
        ctype[key] = np.vstack(ctype[key])

    points = []
    label = []
    for i, key in enumerate(ctype.keys()):
        points.append(ctype[key])
        label.append([i]*len(ctype[key]))

    points = np.vstack(points)
    label = np.concatenate(label)
    mesh = io.Mesh(points, {'vertex': np.arange(len(points))[:,None]}, point_data={'label': label})
    return mesh



def plot_slices(slices, show=False, which='lv', lge=False):
    fig = go.Figure()
    for n in range(len(slices)):
        if slices[n].cmr.view == 'sa':
            color = 'blue'
        else:
            color = 'red'
        xyz = slices[n].get_xyz_trans(which)
        if lge:
            fig = show_point_cloud(xyz[slices[n].lge_data==2], opacity=0.5, size=5, color='blue', fig=fig,
                                      label=(slices[n].cmr.view+str(slices[n].slice_number)))
            fig = show_point_cloud(xyz[slices[n].lge_data==1], opacity=0.5, size=5, color='red', fig=fig,
                                      label=(slices[n].cmr.view+str(slices[n].slice_number)))
        else:
            fig = show_point_cloud(xyz, opacity=0.5, size=5, color=color, fig=fig,
                                  label=(slices[n].cmr.view+str(slices[n].slice_number)))

    if show:
        fig.show()
    return fig


def plot_seg_files(seg_files, which=2):
    fig = go.Figure()
    for file in seg_files:
        # Load data
        img = nib.load(file)
        data = img.get_fdata().astype(float)

        points_ijk = np.vstack(np.where(np.isclose(data, which))).T
        points_xyz = nib.affines.apply_affine(img.affine, points_ijk)

        show_point_cloud(points_xyz, fig=fig, opacity=0.5, size=5, label=os.path.basename(file).split('.')[0])

    fig.show()