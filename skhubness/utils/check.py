# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer
import numpy as np

__all__ = ['check_n_candidates']


def check_n_candidates(n_candidates):
    # Check the n_neighbors parameter
    if n_candidates <= 0:
        raise ValueError(f"Expected n_neighbors > 0. Got {n_candidates:d}")
    if not np.issubdtype(type(n_candidates), np.integer):
        raise TypeError(f"n_neighbors does not take {type(n_candidates)} value, enter integer value")
    return n_candidates
