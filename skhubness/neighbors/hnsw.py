# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array
import nmslib
from .approximate_neighbors import ApproximateNearestNeighbor

__all__ = ['HNSW']


class HNSW(ApproximateNearestNeighbor):
    valid_metrics = ['euclidean', 'l2', 'minkowski', 'squared_euclidean', 'sqeuclidean',
                     'cosine', 'cosinesimil']

    def __init__(self, n_candidates: int = 5, metric: str = 'euclidean',
                 method: str = 'hnsw', post_processing: int = 2,
                 n_jobs: int = 1, verbose: int = 0):
        super().__init__(n_candidates=n_candidates,
                         metric=metric,
                         n_jobs=n_jobs,
                         verbose=verbose)
        self.method = method
        self.post_processing = post_processing
        self.space = None

    def fit(self, X, y=None) -> HNSW:
        """ Setup the HNSW index."""
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
        hnsw_index.createIndex({'post': post_processing},
                               print_progress=(self.verbose >= 2))
        self.index_ = hnsw_index

        assert self.space in ['l2', 'cosinesimil'], f'Internal: self.space={self.space} not allowed'

        return self

    def kneighbors(self, X: np.ndarray = None, n_candidates: int = None, return_distance: bool = True):
        check_is_fitted(self, ["index_", ])

        if X is None:
            raise NotImplementedError(f'Please provide X to hnsw.kneighbors().')

        # Check the n_neighbors parameter
        if n_candidates is None:
            n_candidates = self.n_candidates
        elif n_candidates <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_candidates:d}")
        else:
            if not np.issubdtype(type(n_candidates), np.integer):
                raise TypeError(f"n_neighbors does not take {type(n_candidates)} value, enter integer value")

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
