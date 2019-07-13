# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class HubnessReduction(ABC):
    """ Base class for hubness reduction. """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, neigh_dist, neigh_ind):
        pass

    @abstractmethod
    def transform(self, neigh_dist, neigh_ind, return_distance=True):
        pass

    def fit_transform(self, neigh_dist, neigh_ind, return_distance=True):
        self.fit(neigh_dist, neigh_ind)
        return self.transform(neigh_dist, neigh_ind, return_distance=return_distance)


class NoHubnessReduction(HubnessReduction):
    """ Compatibility class for neighbor search without hubness reduction. """

    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, neigh_dist=None, neigh_ind=None):
        pass

    def transform(self, neigh_dist=None, neigh_ind=None, return_distance=True, *args, **kwargs):
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
