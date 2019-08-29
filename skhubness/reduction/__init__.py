# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

"""
The :mod:`skhubness.reduction` package provides methods for hubness reduction.
"""

from .base import NoHubnessReduction
from .mutual_proximity import MutualProximity
from .local_scaling import LocalScaling
from .dis_sim import DisSimLocal
# from .shared_neighbors import SharedNearestNeighbors, SimhubIn

__all__ = ['NoHubnessReduction',
           'LocalScaling',
           'MutualProximity',
           'DisSimLocal',
           # 'SharedNearestNeighbors',
           # 'SimhubIn',
           ]
