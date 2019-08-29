# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

""" Python package for nearest neighbor retrieval in high-dimensional space."""

__version__ = '0.21.0a7'

from . import analysis
from . import data
from .analysis.estimation import Hubness
from . import neighbors
from . import reduction
from . import utils


__all__ = ['analysis',
           'data',
           'Hubness',
           'neighbors',
           'reduction',
           'utils',
           ]
