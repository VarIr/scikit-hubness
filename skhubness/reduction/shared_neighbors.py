# -*- coding: utf-8 -*-
from __future__ import annotations

from .base import HubnessReduction


class SharedNearestNeighbors(HubnessReduction):
    """ Hubness reduction with Shared Nearest Neighbors (snn). """

    def __init__(self):
        super().__init__()

    def fit(self, neigh_dist, neigh_ind, X=None, *args, **kwargs) -> SharedNearestNeighbors:
        raise NotImplementedError(f'SNN is not yet implemented.')

    def transform(self, neigh_dist, neigh_ind, X=None, *args, **kwargs):
        pass


class SimhubIn(HubnessReduction):
    """ Hubness reduction with unsupervised Simhub (simhubin). """

    def __init__(self):
        super().__init__()

    def fit(self, neigh_dist, neigh_ind, X=None, *args, **kwargs) -> SimhubIn:
        raise NotImplementedError(f'Simhub is not yet implemented.')

    def transform(self, neigh_dist, neigh_ind, X=None, *args, **kwargs):
        pass
