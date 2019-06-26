# -*- coding: utf-8 -*-

""" Python hubness package for nearest neighbor retrieval in high-dimensional space.

This file is part of the HUBNESS package available at
https://github.com/OFAI/hubness/
The HUBNESS package is licensed under the terms of the GNU GPLv3.

(c) 2018-2019, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI) and
University of Vienna, Division of Computational Systems Biology (CUBE)
Contact: <roman.feldbauer@univie.ac.at>
"""

__version__ = '0.1.201906a2'

from . import analysis
from .analysis.estimation import Hubness
from . import reduction
from . import utils


__all__ = ['analysis',
           'reduction',
           'utils',
           ]
