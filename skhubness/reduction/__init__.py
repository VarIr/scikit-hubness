# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

"""
The :mod:`skhubness.reduction` package provides methods for hubness reduction.
"""

from .base import NoHubnessReduction
from .mutual_proximity import MutualProximity
from .local_scaling import LocalScaling
from .dis_sim import DisSimLocal

#: Supported hubness reduction algorithms
hubness_algorithms = ['mp',
                      'ls',
                      'dsl',
                      ]
hubness_algorithms_long = ['mutual_proximity',
                           'local_scaling',
                           'dis_sim_local',
                           ]


__all__ = ['NoHubnessReduction',
           'LocalScaling',
           'MutualProximity',
           'DisSimLocal',
           'hubness_algorithms',
           ]
