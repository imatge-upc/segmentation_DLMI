import numpy as np


def bbox(array, point, radius):
    a = array[np.where(np.logical_and(array[:, 0] >= point[0] - radius, array[:, 0] <= point[0] + radius))]
    b = a[np.where(np.logical_and(a[:, 1] >= point[1] - radius, a[:, 1] <= point[1] + radius))]
    c = b[np.where(np.logical_and(b[:, 2] >= point[2] - radius, b[:, 2] <= point[2] + radius))]
    return c


def hausdorff(surface_a, surface_b):

    # Taking two arrays as input file, the function is searching for the Hausdorff distane of "surface_a" to "surface_b"
    dists = []

    l = len(surface_a)

    for i in xrange(l):

        # walking through all the points of surface_a
        dist_min = 1000.0
        radius = 0
        b_mod = np.empty(shape=(0, 0, 0))

        # increasing the cube size around the point until the cube contains at least 1 point
        while b_mod.shape[0] == 0:
            b_mod = bbox(surface_b, surface_a[i], radius)
            radius += 1

        # to avoid getting false result (point is close to the edge, but along an axis another one is closer),
        # increasing the size of the cube
        b_mod = bbox(surface_b, surface_a[i], radius * math.sqrt(3))

        for j in range(len(b_mod)):
            # walking through the small number of points to find the minimum distance
            dist = np.linalg.norm(surface_a[i] - b_mod[j])
            if dist_min > dist:
                dist_min = dist

        dists.append(dist_min)

    return np.max(dists)