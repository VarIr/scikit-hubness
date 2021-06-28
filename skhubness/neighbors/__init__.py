# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
"""
The :mod:`skhubness.neighbors` package provides wrappers for various
approximate nearest neighbor packages. These are compatible with the
scikit-learn `KNeighborsTransformer`.
"""
from ._annoy import AnnoyTransformer, LegacyRandomProjectionTree
from .ball_tree import BallTree
from .base import VALID_METRICS, VALID_METRICS_SPARSE
from .classification import KNeighborsClassifier, RadiusNeighborsClassifier
from .graph import kneighbors_graph, radius_neighbors_graph
from ._nmslib import NMSlibTransformer, LegacyHNSW
from .lsh import FalconnLSH, PuffinnLSH
from .kd_tree import KDTree
from ._ngt import NGTTransformer, LegacyNNG
from .dist_metrics import DistanceMetric
from .regression import KNeighborsRegressor, RadiusNeighborsRegressor
from .nearest_centroid import NearestCentroid
from .kde import KernelDensity
from .lof import LocalOutlierFactor
from .nca import NeighborhoodComponentsAnalysis
from .unsupervised import NearestNeighbors


__all__ = [
    "AnnoyTransformer",
    "BallTree",
    "DistanceMetric",
    "FalconnLSH",
    "KDTree",
    "LegacyHNSW",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "NearestCentroid",
    "NearestNeighbors",
    "NGTTransformer",
    "NMSlibTransformer",
    "PuffinnLSH",
    "LegacyNNG",
    "RadiusNeighborsClassifier",
    "RadiusNeighborsRegressor",
    "kneighbors_graph",
    "radius_neighbors_graph",
    "KernelDensity",
    "LocalOutlierFactor",
    "NeighborhoodComponentsAnalysis",
    "VALID_METRICS",
    "VALID_METRICS_SPARSE",
]
