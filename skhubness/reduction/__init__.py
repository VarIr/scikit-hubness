# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

"""
The :mod:`skhubness.reduction` package provides methods for hubness reduction.
"""

from ._base import NoHubnessReduction
from ._mutual_proximity import GraphMutualProximity, MutualProximity
from ._local_scaling import GraphLocalScaling, LocalScaling
from ._dis_sim import GraphDisSimLocal, DisSimLocal

#: Supported hubness reduction algorithms
hubness_algorithms = [
    "mp",
    "ls",
    "dsl",
]
hubness_algorithms_long = [
    "mutual_proximity",
    "local_scaling",
    "dis_sim_local",
]


__all__ = [
    "NoHubnessReduction",
    "GraphLocalScaling",
    "LocalScaling",
    "GraphMutualProximity",
    "MutualProximity",
    "GraphDisSimLocal",
    "DisSimLocal",
    "hubness_algorithms",
]
