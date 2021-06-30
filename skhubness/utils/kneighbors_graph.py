# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.validation import check_array
import numba

__all__ = [
    "check_kneighbors_graph",
    "check_matching_n_indexed",
    "hubness_reduced_k_neighbors_graph",
    "sort_data_indices",
]


@numba.jit
def _is_sorted_per_row(arr: np.ndarray) -> bool:
    n, m = arr.shape
    for i in range(n):
        for j in range(m - 1):
            if arr[i, j] >= arr[i, j + 1]:
                return False
    return True


def sort_data_indices(data, indices, n_neighbors):
    """ Sort the .data and .indices array of a ``csr_matrix`` according to .data """
    data = data.reshape(-1, n_neighbors)
    indices = indices.reshape(-1, n_neighbors)
    sorted_ind = np.argsort(data, axis=1)
    sorted_data = np.take_along_axis(data, sorted_ind, axis=1)
    sorted_indices = np.take_along_axis(indices, sorted_ind, axis=1)
    return sorted_data, sorted_indices


def check_kneighbors_graph(
        kng: csr_matrix,
        check_sparse: bool = True,
        check_empty: bool = True,
        check_shape: bool = True,
        check_sorted: str = "full",
        sort_if_necessary: bool = True,
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

    sort_if_necessary : bool
        Sort `kng` according to its .data attribute (distances), if `check_sorted` test fails.

    Returns
    -------
    kneighbors_graph : csr_matrix
    """
    # Start off with standard sklearn checks
    kng: csr_matrix = check_array(kng, accept_sparse=True)  # noqa

    if check_sparse and not issparse(kng):
        raise ValueError("The k-neighbors graph is expected to be a sparse matrix.")
    kng = kng.tocsr()

    n_query, n_indexed = kng.shape
    if check_empty and (n_query < 1 or n_indexed < 1):
        raise ValueError(f"K-neighbors graph must not be empty. Got shape ({n_query}, {n_indexed}).")

    n_neighbors = kng.indptr[1]
    if check_shape:
        msg = "Misshaped sparse k-neighbors graph. For each object, identically many neighbors must be stored."
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
        msg = "Sparse k-neighbors graph must be sorted, that is, store ascending distances per row."
        if check_sorted == "simple":
            dist = kng.data[:n_neighbors]
            if np.any(dist[:-1] > dist[1:]):
                raise ValueError(msg)
        elif check_sorted == "full" or check_sorted is True:
            if not _is_sorted_per_row(kng.data.reshape(n_query, n_neighbors)):
                if sort_if_necessary:
                    data, indices = sort_data_indices(kng.data, kng.indices, kng.indptr[1])
                    kng.data = data
                    kng.indices = indices
                else:
                    raise ValueError(msg)
        else:
            raise ValueError(f"Invalid argument passed for check_sorted = {check_sorted}.")

    return kng


def check_matching_n_indexed(kng, n_indexed):
    if kng.shape[1] != n_indexed:
        raise ValueError(f"Shape of query kneighbors graph {kng.shape} does not match the number of indexed data.")


def hubness_reduced_k_neighbors_graph(
        hub_reduced_dist: np.ndarray,
        original_X: csr_matrix,
        sort_distances: bool = True,
) -> csr_matrix:
    """ Create a possibly sorted sparse kneighbors graph from hubness-reduced distances and indices.

    Parameters
    ----------
    hub_reduced_dist : array-like of shape (n_query, n_neighbors)
        Array of (usually hubness-reduced) distances.
    original_X : csr_matrix of shape (n_query, n_indexed)
        The original k-neighbors graph (usually from before hubness reduction);
        Might be sorted according to `hub_reduced_dist`.
    sort_distances : bool, default = True
        Sort the new k-neighbors graph

    Returns
    -------
    kneighbors_graph : csr_matrix
    """
    n_query, _ = original_X.shape
    hub_reduced_ind = original_X.indices.reshape(n_query, -1)

    if sort_distances:
        n_neighbors = original_X.indptr[1]
        hub_reduced_dist, hub_reduced_ind = sort_data_indices(
            data=hub_reduced_dist,
            indices=hub_reduced_ind,
            n_neighbors=n_neighbors,
        )

    # Construct CSR matrix k-neighbors graph
    A_data = hub_reduced_dist.ravel()
    A_indices = hub_reduced_ind.ravel()
    A_indptr = original_X.indptr

    kneighbors_graph = csr_matrix(
        (A_data, A_indices, A_indptr),
        shape=original_X.shape,
    )

    return kneighbors_graph
