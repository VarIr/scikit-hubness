# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
"""
The :mod:`skhubness.neighbors` package is a drop-in replacement for :mod:`sklearn.neighbors`,
providing all of its features, while adding transparent support for hubness reduction
and approximate nearest neighbor search.
"""
from .ball_tree import BallTree
from .base import VALID_METRICS, VALID_METRICS_SPARSE
from .classification import KNeighborsClassifier, RadiusNeighborsClassifier
from .graph import kneighbors_graph, radius_neighbors_graph
from .hnsw import HNSW
from .approximate_neighbors import UnavailableANN
try:
    from .lsh import FalconnLSH
except ImportError:
    FalconnLSH = UnavailableANN
try:
    from .lsh import PuffinnLSH
except ImportError:
    PuffinnLSH = UnavailableANN
from .kd_tree import KDTree
try:
    from .onng import ONNG
except ImportError:
    ONNG = UnavailableANN
from .random_projection_trees import RandomProjectionTree
from .dist_metrics import DistanceMetric
from .regression import KNeighborsRegressor, RadiusNeighborsRegressor
from .nearest_centroid import NearestCentroid
from .kde import KernelDensity
from .lof import LocalOutlierFactor
from .nca import NeighborhoodComponentsAnalysis
from .unsupervised import NearestNeighbors


__all__ = ['BallTree',
           'DistanceMetric',
           'FalconnLSH',
           'KDTree',
           'HNSW',
           'KNeighborsClassifier',
           'KNeighborsRegressor',
           'NearestCentroid',
           'NearestNeighbors',
           'PuffinnLSH',
           'ONNG',
           'RadiusNeighborsClassifier',
           'RadiusNeighborsRegressor',
           'RandomProjectionTree',
           'kneighbors_graph',
           'radius_neighbors_graph',
           'KernelDensity',
           'LocalOutlierFactor',
           'NeighborhoodComponentsAnalysis',
           'VALID_METRICS',
           'VALID_METRICS_SPARSE',
           ]
