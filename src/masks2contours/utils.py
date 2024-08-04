
from glob import glob
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import scipy.io as sio
from shapely.geometry import Polygon

def fitLine3D(points):
    '''
    Given "points", an m x 3 ndarray for some m, this function returns a 1 x n ndarray containing the residuals between
    points on a line of best fit and the original points from "points". The best fit line is obtained with use of SVD.
    '''

    # First, use SVD to compute coeff.
    colMeans = np.array([np.mean(points[:, i]) for i in range(0, 2 + 1)]) # Compute mean of each col in points.
    tmp = points - colMeans # tmp is the result of subtracting colMeans from each row of points.
    (u, s, vh) = np.linalg.svd(tmp)

    # The matrices returned by the numpy SVD method seem to be transposes of those returned by the MATLAB SVD method.
    # So we access vh[0, :] here rather than vh[:, 0].
    coeff = vh[0, :]

    # Go backwards to calculate error.
    b = (tmp @ coeff).transpose()
    c = np.outer(coeff, b)
    newPoints = c.transpose() + colMeans

    # Calculate err
    err = [np.linalg.norm(newPoints[i, :] - points[i,:]) for i in range(0, points.shape[0])]
    return np.array(err)

def lineNormals2D(vertices, lineIndices = None, perpendicular = False):
    '''
    "vertices" is a m x 2 ndarray for some m, and "lineIndices" is a (m - 1) x 2 ndarray of indices
    for the "vertices" ndarray. More explicitly, the ith row of lineIndices contains the indices of the points that
    constitute the ith line segment.

    Returns an m x 2 ndarray, where the ith row contains the components of the normal vector corresponding to the direction
    of the ith line segment.

    This function was ported over from a MATLAB function written by D. Kroon from University of Twente in August 2011.
    '''

    numVertices = vertices.shape[0]

    if numVertices != 0 and numVertices < 2:
        raise ValueError("\"vertices\" must have either 0 rows or 2 or more rows; that is, there must be eiher 0 vertices or more than 2 vertices.")

    if vertices is None or (type(vertices) == np.ndarray and vertices.size == 0):
        return np.array([])

    # If nothing is passed in for "lines", initialize lines to the ndarray [[0, 1], [1, 2], ..., [numVertices - 2, numVertices - 1]].
    if lineIndices is None or (type(lineIndices) == np.ndarray and lineIndices.size == 0):
        lineIndices = np.column_stack((np.array(range(0, numVertices - 1)), np.array(range(1, (numVertices)))))

    # Calculate tangent vectors.
    DT = vertices[lineIndices[:, 0], :] - vertices[lineIndices[:, 1], :]

    # Divide each tangent vector by its length. ("Weighted central difference"?)
    LL = np.sqrt(np.power(DT[:, 0], 2) + np.power(DT[:, 1], 2))
    eps = np.spacing(1) # Returns the distance from 1 to the nearest number representible by the computer. I.e., is really small.
    LL2 = np.maximum(np.power(LL, 2), eps) # This is essentially the elementwise second power of LL.
    DT[:, 0] = np.divide(DT[:, 0], LL2) # The scaling happens in this line and the one below.
    DT[:, 1] = np.divide(DT[:, 1], LL2)

    # For each i, let the ith row of D be the entries from DT that coorespond to the ith line.
    D1 = np.zeros(vertices.shape)
    D1[lineIndices[:, 0], :] = DT
    D2 = np.zeros(vertices.shape)
    D2[lineIndices[:, 1], :] = DT
    D = D1 + D2

    # Normalize the normals.
    LL = np.sqrt(np.power(D[:, 0], 2) + np.power(D[:, 1], 2))
    LL = np.maximum(LL, eps) # The equivalent of this line was not in the MATLAB original, but is necessary to prevent division by zero ocurring as a result of division by too-small.
    normals = np.zeros(D.shape) # Note that D.shape == vertices.shape.

    if perpendicular:
        normals[:, 0] = np.divide(-D[:, 1], LL)
        normals[:, 1] = np.divide(D[:, 0], LL)
    else:
        normals[:, 0] = np.divide(D[:, 0], LL)
        normals[:, 1] = np.divide(D[:, 1], LL)

    return normals


def getRVinsertIndices(points):
    '''
    "points" is a m1 x 2 ndarray for some m1.
    Returns an 1 x m2 ndarray, for some m2, containing the indices of endoRVFWContours that correspond to the RV insert points.
    '''

    distances = pointDistances(points)
    upperThreshold = np.mean(distances) + 3 * np.std(distances, ddof = 1) # We need to use ddof = 1 to use Bessel's correction (so we need it to get the same std as is calculated in MATLAB).
    largeDistExists = np.any(distances > upperThreshold)

    # Find the index (in "distances") of the point that is furthest from its neighbor. Return an ndarray consisting of
    # this point and *its* neighbor.
    if largeDistExists != 0:
        largestDistIndex = np.argmax(distances)
        if largestDistIndex == len(points) - 1: # if the point furthest from its neighbor is the last point...
            return np.array([0, largestDistIndex]) #the neighbor to largestDistIndex is 0 in this case
        else:
            return np.array([largestDistIndex, largestDistIndex + 1])
    else:
        return np.array([])


def getLAinsert(inserts, sep_points):
    tree = KDTree(sep_points)
    dist, _ = tree.query(inserts)

    return inserts[np.argmin(dist)]

def removeFarPoints(oldPoints):
    '''
    "oldPoints" is a m1 x 2 ndarray for some m1.

    Returns a m2 x 2 ndarray, for some m2, that is the result of removing points that are "too far" from the majority of
    the points in "oldPoints".
    '''

    # If there are 2 or less points, then no points will ever be considered "too far". (The condition for "too far" is
    # given in the computation of far_indices, below).
    if oldPoints.shape[0] <= 2:
        return oldPoints

    # Record the distances between consecutive points (wrap around from the last point to the first point) in oldPoints.
    distances = pointDistances(oldPoints)

    # Find points that are far away from other points. In the below, note that simply calling np.nonzero() doesn't do
    # the trick; np.nonzero() returns a tuple, and we want the first element of that tuple.
    far_indices = np.nonzero(distances > (np.mean(distances) + 2 * np.std(distances)))[0]

    # Comment from the MATLAB script: "The point must be far from its two nearest neighbours, therefore, if
    # there are extraneous points, the length of variable 'far' must be even. If the length of 'far' is odd, the contour is open
    # (but there still may be extraneous points)." What???

    # Every point that is far from *both* of its neighbors is considered to be a "far point". We return the result
    # of removing all such far points from oldPoints.
    if np.size(far_indices) > 1: # Using > 1 instead of != 0 probably fixes some edge case.
        indicesToRemove = []
        for i in range(0, np.size(far_indices) - 1): # i covers the range {0, ..., np.size(far_indices) - 2} since np.size(far_indices) - 1 isn't included
            if (far_indices[i] == 0) and (far_indices[-1] == np.size(oldPoints) - 1):
                indicesToRemove.append(far_indices[i])
            elif far_indices[i + 1] - far_indices[i] == 1:
                indicesToRemove.append(far_indices[i + 1])

        # Remove the extraneous points.
        return deleteHelper(oldPoints, indicesToRemove)
    else:
        return oldPoints # Remove nothing in this case.

def pointDistances(points):
    '''
    "points" is an m x 2 ndarray for some m.

    Returns an m x 1 ndarray in which the ith entry is the distance from the ith point in "points" to the (i + 1)st point.
    The last entry is the distance from the last point to the first point.
    '''

    numPts = points.shape[0]
    distances = []
    for i in range(0, numPts): # i covers the range {0, ..., numPts - 1} since numPts isn't included
        if i == numPts - 1:  # If on last point, compare last point with first point
            distances.append(np.linalg.norm(points[i, :] - points[0, :]))
        else:
            distances.append(np.linalg.norm(points[i + 1, :] - points[i, :]))

    return np.array(distances)

def deleteHelper(arr, indices, axis = 0):
    '''
    Helper function for deleting parts of ndarrays that, unlike np.delete(), works when either "arr" or "indices" is empty.
    (np.delete() does not handle the case in which arr is an empty list or an empty ndarray).

    "arr" may either be a list or an ndarray.
    '''

    def emptyNdarrayCheck(x):
        return type(x) is np.ndarray and ((x == np.array(None)).any() or x.size == 0)

    def emptyListCheck(x):
        return type(x) is list and len(x) == 0

    if emptyNdarrayCheck(arr) or emptyListCheck(arr):
        return arr

    # np.delete() does not work as expected if indices is an empty ndarray. This case fixes that.
    # (If indices is an empty list, np.delete() works as expected, so there is no need to use emptyListCheck() here).
    if emptyNdarrayCheck(indices):
        return arr

    return np.delete(arr, indices, axis)

def sharedRows(arr1, arr2):
    '''
    "arr1" and "arr2" must be m x n ndarrays for some m and n. (So they can't be m x n x s ndarrays).

    Returns the list [_sharedRows, sharedIndicesArr1, sharedIndicesArr2].
    "_sharedRows" is a matrix whose rows are those that are shared by "arr1" and "arr2".
    "sharedIndicesArr1" is a list of the row-indices in arr1 whose rows appear (not necessarily with the same indices) in
    arr2.
    "sharedIndicesArr2" is a similar list of row-indices that pertains to "arr2".
    '''

    if arr1 is None or arr1.size == 0 or arr2 is None or arr2.size == 0: #If either array is empty, return a list containing three empty ndarrays.
        return[np.array([]), np.array([]), np.array([])]

    if arr1.shape[1] != arr2.shape[1]:
        raise ValueError("Arrays must have same number of columns.")

    # Get the indices of the shared rows.
    sharedIndicesArr1 = []
    sharedIndicesArr2 = []

    for i in range(0, arr1.shape[0]):
        for j in range(0, arr2.shape[0]):
            if i in sharedIndicesArr1 or j in sharedIndicesArr2:
                continue
            elif np.all(arr1[i, :] == arr2[j, :]): #If (ith row in arr1) == (jth row in arr2)
                sharedIndicesArr1.append(i)
                sharedIndicesArr2.append(j)

    # Use these indices to build the matrix of shared rows.
    _sharedRows = arr1[sharedIndicesArr1, :]

    # Convert the lists of sharedIndices to ndarrays.
    sharedIndicesArr1 = np.array(sharedIndicesArr1)
    sharedIndicesArr2 = np.array(sharedIndicesArr2)

    return [_sharedRows, sharedIndicesArr1, sharedIndicesArr2]

def removeZerorows(arr):
    '''
    Returns the result of removing rows of arr, which is an ndarray, that are all 0.
    '''

    return arr[np.any(arr, axis = 1), :]


def calcApex(epiPts1, epiPts2):
    '''
    Return the point p1 in epiPts1 such that p1 minimizes the distance between p1 and p2, where p2 can be
    any point in epiPts2.
    '''

    _epiPts1 = removeZerorows(epiPts1)
    _epiPts2 = removeZerorows(epiPts2)
    dist = cdist(_epiPts1, _epiPts2) # Compute pairwise distances.
    apexIndex = np.unravel_index(np.argmin(dist), dist.shape)[0]
    if apexIndex.shape != ():
        apexIndex = apexIndex[0]

    return epiPts1[apexIndex, :]

def manuallyCompileValvePoints(fldr, numFrames, frameNum):
    '''
    fldr is the folder where the valve points .mat files are stored.
    numFrames is the number of frames in the SA image.

    Returns the tuple (mv, tv, av, pv), where...
    - mv is a 3D ndarray such that mv[i, :, :] is the 2D ndarray containing the mitral valve points for the ith mitral valve
    file in fldr.
    - tv, av, and pv are 2D ndarrays containing the tricuspid, aortic, and pulmonary valve points. tv, av, and pv may
    all be None.
    '''

    mat_filenames = glob(fldr + "valve-motion-predicted-LA_[0-9]CH.mat")
    mat_files = [sio.loadmat(file) for file in mat_filenames]

    mv = np.zeros((len(mat_files), 2, 3))
    tv = np.zeros((2, 3))
    av = np.zeros((2, 3))
    pv = np.zeros((2, 3))

    for i, file in enumerate(mat_files):
        # Load the variable point_coords_mv from the MATLAB file.
        point_coords_mv = file["point_coords_mv"]

        # Interpolate onto applicable number of frames if needed.
        tmp = interpTime(point_coords_mv, numFrames)
        mv[i, :, :] = tmp[frameNum, :, :]

    def get_interp_mat_var(mat_var):
        var_name = "point_coords_" + mat_var
        result = None
        for file in mat_files:
            if var_name in file:
                point_coords_mat_var = file[var_name]
                result = interpTime(point_coords_mat_var, numFrames)
                result = np.squeeze(result[frameNum, :, :])
                break

        return result

    # Return stuff.
    _tv = get_interp_mat_var("tv")
    _av = get_interp_mat_var("av")
    _pv = get_interp_mat_var("pv")

    if _tv is not None:
        tv = _tv
    if _av is not None:
        av = _av
    if _pv is not None:
        pv = _pv

    return (mv, tv, av, pv)

def interpTime(oldValvePts, newNumInterpSamples):
    '''
    oldValvePts is a m x 2 ndarray, and newNumInterpSamples is a positive integer.
    Returns the new, interpolated valve positions.
    '''

    result = np.zeros((newNumInterpSamples, oldValvePts.shape[1], oldValvePts.shape[2]))

    for i in range(0, oldValvePts.shape[1]):
        for j in range(0, oldValvePts.shape[2]):
            sampledValues = np.squeeze(oldValvePts[:, i, j])
            sampledValues = deleteHelper(sampledValues, sampledValues == 0, axis = 0) # Remove all zeros from sampledValues
            if sampledValues.size != 0:
                samplePts = np.linspace(1, 100, len(sampledValues))
                queryPts = np.linspace(1, 100, newNumInterpSamples)
                result[:, i, j] = np.interp(queryPts, samplePts, sampledValues) # note, np.interp() gives slightly different results than MATLAB's interp1d()
            else:
                result[:, i, j] = np.zeros(newNumInterpSamples, 1)

    return result


def calculate_area_of_polygon_3d(points, normal):
    # Generate two orthonormal vectors to the normal
    v1 = np.array([1, 0, 0])
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    v1 = np.cross(v2, normal)
    v1 = v1 / np.linalg.norm(v1)

    # Define rotation matrix
    R = np.array([v1, v2, normal])

    # Rotate the points
    centroid = np.mean(points, axis=0)
    polygon = np.dot(points-centroid, R.T)[:, :2]

    # Compute the area of the polygon
    area = Polygon(polygon).area

    def PolyArea(x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    return area