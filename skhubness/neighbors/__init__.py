# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
"""
The :mod:`skhubness.neighbors` package provides wrappers for various
approximate nearest neighbor packages. These are compatible with the
scikit-learn `KNeighborsTransformer`.
"""
from ._annoy import AnnoyTransformer, LegacyRandomProjectionTree
from .base import VALID_METRICS, VALID_METRICS_SPARSE
from .classification import KNeighborsClassifier, RadiusNeighborsClassifier
from ._nmslib import NMSlibTransformer, LegacyHNSW
from .lsh import FalconnLSH, PuffinnLSH
from ._ngt import NGTTransformer, LegacyNNG
from .regression import KNeighborsRegressor, RadiusNeighborsRegressor
from .unsupervised import NearestNeighbors


__all__ = [
    "AnnoyTransformer",
    "FalconnLSH",
    "LegacyHNSW",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "NearestNeighbors",
    "NGTTransformer",
    "NMSlibTransformer",
    "PuffinnLSH",
    "LegacyNNG",
    "RadiusNeighborsClassifier",
    "RadiusNeighborsRegressor",
    "VALID_METRICS",
    "VALID_METRICS_SPARSE",
]
