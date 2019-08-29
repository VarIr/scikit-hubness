# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod


class HubnessReduction(ABC):
    """ Base class for hubness reduction. """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, neigh_dist, neigh_ind, X, assume_sorted, *args, **kwargs):
        pass  # pragma: no cover

    @abstractmethod
    def transform(self, neigh_dist, neigh_ind, X, assume_sorted, return_distance=True):
        pass  # pragma: no cover

    def fit_transform(self, neigh_dist, neigh_ind, X, assume_sorted=True, return_distance=True, *args, **kwargs):
        self.fit(neigh_dist, neigh_ind, X, assume_sorted, *args, **kwargs)
        return self.transform(neigh_dist, neigh_ind, X, assume_sorted, return_distance)


class NoHubnessReduction(HubnessReduction):
    """ Compatibility class for neighbor search without hubness reduction. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):
        pass  # pragma: no cover

    def transform(self, neigh_dist, neigh_ind, X, assume_sorted=True, return_distance=True, *args, **kwargs):
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
