from collections import namedtuple
import numpy as np

__all__ = [
    "sort_distances_indices",
]


def sort_distances_indices(dist, ind):
    sorted_ind = np.argsort(dist, axis=1)
    dist = np.take_along_axis(dist, sorted_ind, axis=1)
    ind = np.take_along_axis(ind, sorted_ind, axis=1)
    return namedtuple("dist_ind", ["dist", "ind"])(dist=dist, ind=ind)
