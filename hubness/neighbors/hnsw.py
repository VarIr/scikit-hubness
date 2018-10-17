#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUBNESS package available at
https://github.com/OFAI/hubness/
The HUBNESS package is licensed under the terms of the GNU GPLv3.

(c) 2018, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI) and
University of Vienna, Division of Computational Systems Biology (CUBE)
Contact: <roman.feldbauer@ofai.at>
"""
from sklearn.neighbors.dist_metrics import get_valid_metric_ids

__all__ = ['HNSW']

DOC_DICT = ...

VALID_METRICS = ['SEuclideanDistance', 'CosineSimil', 'InnerProduct']


class HNSW(object):

    valid_metrics = get_valid_metric_ids(VALID_METRICS)
    valid_metrics.extend(['cosine_simil', 'inner'])
