# -*- coding: utf-8 -*-

"""
The :mod:`hubness.neighbors` module implements the (hubness reduced) k-nearest neighbors algorithm.
"""
# from .ball_tree import BallTree
# from .kd_tree import KDTree
# from .dist_metrics import DistanceMetric
# from .graph import kneighbors_graph, radius_neighbors_graph
# from .unsupervised import NearestNeighbors
# from .regression import KNeighborsRegressor, RadiusNeighborsRegressor
# from .nearest_centroid import NearestCentroid
# from .kde import KernelDensity
# from .lof import LocalOutlierFactor
# from .nca import NeighborhoodComponentsAnalysis
from .classification import KNeighborsClassifier  #, RadiusNeighborsClassifier
from .base import VALID_METRICS, VALID_METRICS_SPARSE
from .lsh import LSH

__all__ = [# 'BallTree',
           # 'DistanceMetric',
           # 'KDTree',
           'KNeighborsClassifier',
           # 'KNeighborsRegressor',
           'LSH',
           # 'NearestCentroid',
           # 'NearestNeighbors',
           # 'RadiusNeighborsClassifier',
           # 'RadiusNeighborsRegressor',
           # 'kneighbors_graph',
           # 'radius_neighbors_graph',
           # 'KernelDensity',
           # 'LocalOutlierFactor',
           # 'NeighborhoodComponentsAnalysis',
           'VALID_METRICS',
           'VALID_METRICS_SPARSE']
