# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
"""
The :mod:`skhubness.neighbors` package provides wrappers for various
approximate nearest neighbor packages. These are compatible with the
scikit-learn `KNeighborsTransformer`.
"""
from ._annoy import AnnoyTransformer, LegacyRandomProjectionTree
from ._nmslib import NMSlibTransformer, LegacyHNSW
from ._puffinn import PuffinnTransformer, LegacyPuffinn
from ._ngt import NGTTransformer, LegacyNNG


__all__ = [
    "AnnoyTransformer",
    "LegacyHNSW",
    "LegacyNNG",
    "LegacyPuffinn",
    "LegacyRandomProjectionTree",
    "NGTTransformer",
    "NMSlibTransformer",
    "PuffinnTransformer",
]
