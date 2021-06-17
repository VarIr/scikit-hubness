# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.validation import check_array
import numba

__all__ = [
    "check_kneighbors_graph",
    'check_n_candidates',
    "check_matching_n_indexed",
]


@numba.jit
def _is_sorted_per_row(arr: np.ndarray) -> bool:
    n, m = arr.shape
    for i in range(n):
        for j in range(m - 1):
            if arr[i, j] >= arr[i, j + 1]:
                return False
    return True


def check_kneighbors_graph(
        kng: csr_matrix,
        check_sparse: bool = True,
        check_empty: bool = True,
        check_shape: bool = True,
        check_sorted: str = "simple",
) -> csr_matrix:
    """ Ensure validity of a k-neighbors graph, casting to CSR format if necessary.

    Parameters
    ----------
    kng : csr_matrix of shape (n_query, n_indexed)
        The k-neighbors graph with n_neighbors stored distances per query object.
    check_sparse : bool
    check_empty : bool
    check_shape : bool
    check_sorted : False, "simple", "full", default = "full"
        Ensure sorted distances (increasing).

        - "simple": Check sorting in first row as a proxy for whole array
        - "full": Check sorting in all rows
        - False: disable check

    Returns
    -------
    kneighbors_graph : csr_matrix
    """
    # Start off with standard sklearn checks
    kng = check_array(kng, accept_sparse=True)

    if check_sparse and not issparse(kng):
        raise ValueError("The k-neighbors graph is expected to be a sparse matrix.")
    kng = kng.tocsr()

    n_query, n_indexed = kng.shape
    if check_empty and (n_query < 1 or n_indexed < 1):
        raise ValueError(f"K-neighbors graph must not be empty. Got shape ({n_query}, {n_indexed}).")

    n_neighbors = kng.indptr[1]
    if check_shape:
        msg = "Misshaped k-neighbors graph. For each object, identically many neighbors must be stored."
        try:
            for arr in [kng.data, kng.indices]:
                arr.reshape(n_query, n_neighbors)
        except ValueError:
            raise ValueError(msg + " Could not reshape data or indices.")
        if kng.data.shape != kng.indices.shape:
            raise ValueError(f"Shape of data {kng.data.shape} must match shape of indices {kng.indices.shape}.")
        if np.any(np.diff(kng.indptr) != n_neighbors):
            raise ValueError(msg + " Array indptr must be a homogeneous grid.")

    if check_sorted:
        msg = "K-neighbors graph must be sorted, that is, store ascending distances per row."
        if check_sorted == "simple":
            dist = kng.data[:n_neighbors]
            if np.any(dist[:-1] > dist[1:]):
                raise ValueError(msg)
        elif check_sorted == "full" or check_sorted is True:
            if not _is_sorted_per_row(kng.data):
                raise ValueError(msg)
        else:
            raise ValueError(f"Invalid argument passed for check_sorted = {check_sorted}.")

    return kng


def check_n_candidates(n_candidates):
    # Check the n_neighbors parameter
    if n_candidates <= 0:
        raise ValueError(f"Expected n_neighbors > 0. Got {n_candidates:d}")
    if not np.issubdtype(type(n_candidates), np.integer):
        raise TypeError(f"n_neighbors does not take {type(n_candidates)} value, enter integer value")
    return n_candidates


def check_matching_n_indexed(kng, n_indexed):
    if kng.shape[1] != n_indexed:
        raise ValueError(f"Shape of query kneighbors graph {kng.shape} does not match the number of indexed data.")
