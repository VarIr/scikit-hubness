# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from sklearn.utils.validation import check_is_fitted

from .base import HubnessReduction


class SharedNearestNeighbors(HubnessReduction):
    """ Hubness reduction with Shared Nearest Neighbors (snn). """

    def __init__(self):
        super().__init__()

    def fit(self, neigh_dist, neigh_ind, X=None, *args, **kwargs) -> SharedNearestNeighbors:
        raise NotImplementedError(f'SNN is not yet implemented.')

    def transform(self, neigh_dist, neigh_ind, X=None, *args, **kwargs):
        check_is_fitted(self, 'neigh_dist_train_')


class SimhubIn(HubnessReduction):
    """ Hubness reduction with unsupervised Simhub (simhubin). """

    def __init__(self):
        super().__init__()

    def fit(self, neigh_dist, neigh_ind, X=None, *args, **kwargs) -> SimhubIn:
        raise NotImplementedError(f'Simhub is not yet implemented.')

    def transform(self, neigh_dist, neigh_ind, X=None, *args, **kwargs):
        check_is_fitted(self, 'neigh_dist_train_')
