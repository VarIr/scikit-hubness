# -*- coding: utf-8 -*-

"""
The :mod:`skhubness.reduction` package provides methods for hubness reduction.
"""

from .base import NoHubnessReduction
from .mutual_proximity import MutualProximity
from .local_scaling import LocalScaling
# from .shared_neighbors import SharedNearestNeighbors, SimhubIn

__all__ = ['NoHubnessReduction',
           'LocalScaling',
           'MutualProximity',
           # 'SharedNearestNeighbors',
           # 'SimhubIn',
           ]
