#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  27 09:14:38 2023

@author: jjv
"""

import numpy as np

def fit_elipse_2d(points, tolerance=0.01):
    """ This function fits a elipse to a set of 2D points
        Input:
            [x,y]: 2D points coordinates
            w: weights for points (optional)
        Output:
            [xc,yc]: center of the fitted circle
            r: radius of the fitted circle
    """

    (N, d) = np.shape(points)
    d = float(d)
    # Q will be our working array
    Q = np.vstack([np.copy(points.T), np.ones(N)])
    QT = Q.T

    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT, np.dot(np.linalg.inv(V),Q)))  # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u


    # center of the ellipse
    center = np.dot(points.T, u)
    # the A matrix for the ellipse
    A = np.linalg.inv(
        np.dot(points.T, np.dot(np.diag(u), points)) -
        np.array([[a * b for b in center] for a in center])) / d
    # Get the values we'd like to return
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0 / np.sqrt(s)

    return (center, radii, rotation)
