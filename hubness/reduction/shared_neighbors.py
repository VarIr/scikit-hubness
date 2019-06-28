# -*- coding: utf-8 -*-
from __future__ import annotations


class SharedNearestNeighbors:
    """ Hubness reduction with Shared Nearest Neighbors (SNN). """

    def __init__(self):
        pass

    def fit(self, X, y=None) -> SharedNearestNeighbors:
        pass

    def transform(self, neigh_dist, neigh_ind, *args, **kwargs):
        pass


class SimhubIn:
    """ Hubness reduction with unsupervised Simhub (SHI). """

    def __init__(self):
        pass

    def fit(self, X, y=None) -> SimhubIn:
        pass

    def transform(self, neigh_dist, neigh_ind, *args, **kwargs):
        pass
