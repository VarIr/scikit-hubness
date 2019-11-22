# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer
# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations
from typing import Tuple, Union
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

try:
    import nmslib
except (ImportError, ModuleNotFoundError):
    nmslib = None  # pragma: no cover

from .approximate_neighbors import ApproximateNearestNeighbor
from ..utils.check import check_n_candidates

__all__ = ['HNSW']


class HNSW(ApproximateNearestNeighbor):
    """Wrapper for using nmslib

    Hierarchical navigable small-world graphs are data structures,
    that allow for approximate nearest neighbor search.
    Here, an implementation from nmslib is used.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "euclidean", "manhattan", "hamming", "dot"
    method: str, default = 'hnsw',
        ANN method to use. Currently, only 'hnsw' is supported.
    post_processing: int, default = 2
        More post processing means longer index creation,
        and higher retrieval accuracy.
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose >= 2, show progress bar on indexing.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures
    """
    valid_metrics = ['euclidean', 'l2', 'minkowski', 'squared_euclidean', 'sqeuclidean',
                     'cosine', 'cosinesimil']

    def __init__(self, n_candidates: int = 5, metric: str = 'euclidean',
                 method: str = 'hnsw', post_processing: int = 2,
                 n_jobs: int = 1, verbose: int = 0):

        if nmslib is None:  # pragma: no cover
            raise ImportError(f'Please install the `nmslib` package, before using this class.\n'
                              f'$ pip install nmslib') from None

        super().__init__(n_candidates=n_candidates,
                         metric=metric,
                         n_jobs=n_jobs,
                         verbose=verbose)
        self.method = method
        self.post_processing = post_processing
        self.space = None

    def fit(self, X, y=None) -> HNSW:
        """ Setup the HNSW index from training data.

        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored

        Returns
        -------
        self: HNSW
            An instance of HNSW with a built graph
        """
        X = check_array(X)

        method = self.method
        post_processing = self.post_processing

        if self.metric in ['euclidean', 'l2', 'minkowski', 'squared_euclidean', 'sqeuclidean']:
            if self.metric in ['squared_euclidean', 'sqeuclidean']:
                self.metric = 'sqeuclidean'
            else:
                self.metric = 'euclidean'
            self.space = 'l2'
        elif self.metric in ['cosine', 'cosinesimil']:
            self.space = 'cosinesimil'
        else:
            raise ValueError(f'Invalid metric "{self.metric}". Please try "euclidean" or "cosine".')

        hnsw_index = nmslib.init(method=method,
                                 space=self.space)
        hnsw_index.addDataPointBatch(X)
        hnsw_index.createIndex({'post': post_processing,
                                'indexThreadQty': self.n_jobs,
                                },
                               print_progress=(self.verbose >= 2))
        self.index_ = hnsw_index
        self.n_samples_fit_ = len(self.index_)

        assert self.space in ['l2', 'cosinesimil'], f'Internal: self.space={self.space} not allowed'

        return self

    def kneighbors(self, X: np.ndarray = None,
                   n_candidates: int = None,
                   return_distance: bool = True) -> Union[Tuple[np.array, np.array], np.array]:
        """ Retrieve k nearest neighbors.

        Parameters
        ----------
        X: np.array or None, optional, default = None
            Query objects. If None, search among the indexed objects.
        n_candidates: int or None, optional, default = None
            Number of neighbors to retrieve.
            If None, use the value passed during construction.
        return_distance: bool, default = True
            If return_distance, will return distances and indices to neighbors.
            Else, only return the indices.
        """
        check_is_fitted(self, ["index_", ])

        if X is None:
            raise NotImplementedError(f'Please provide X to hnsw.kneighbors().')

        # Check the n_neighbors parameter
        if n_candidates is None:
            n_candidates = self.n_candidates
        n_candidates = check_n_candidates(n_candidates)

        # Fetch the neighbor candidates
        neigh_ind_dist = self.index_.knnQueryBatch(X,
                                                   k=n_candidates,
                                                   num_threads=self.n_jobs)

        # If fewer candidates than required are found for a query,
        # we save index=-1 and distance=NaN
        n_test = X.shape[0]
        neigh_ind = -np.ones((n_test, n_candidates),
                             dtype=np.int32)
        neigh_dist = np.empty_like(neigh_ind,
                                   dtype=X.dtype) * np.nan

        for i, (ind, dist) in enumerate(neigh_ind_dist):
            neigh_ind[i, :ind.size] = ind
            neigh_dist[i, :dist.size] = dist

        # Convert cosine similarities to cosine distances
        if self.space == 'cosinesimil':
            neigh_dist *= -1
            neigh_dist += 1
        elif self.space == 'l2' and self.metric == 'sqeuclidean':
            neigh_dist **= 2

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
