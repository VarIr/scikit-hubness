# -*- coding: utf-8 -*-

# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations

from functools import partial
import warnings
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.validation import check_is_fitted
import falconn
from tqdm.autonotebook import tqdm

from .approximate_neighbors import ApproximateNearestNeighbor
__all__ = ['LSH']

DOC_DICT = ...

VALID_METRICS = ['SEuclideanDistance', 'sqeuclidean',
                 'cosine', 'neg_inner', 'NegativeInnerProduct']


class LSH(ApproximateNearestNeighbor):
    valid_metrics = VALID_METRICS

    def __init__(self, n_candidates: int = 5, metric: str = 'sqeuclidean', num_probes: int = 50,
                 n_jobs: int = 1, verbose: int = 0):
        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=n_jobs, verbose=verbose)
        self.num_probes = num_probes

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> LSH:
        """ Setup the LSH index from training data. """

        if self.metric in ['sqeuclidean', 'SEuclideanDistance']:
            self.metric = 'sqeuclidean'
            distance = falconn.DistanceFunction.EuclideanSquared
        elif self.metric in ['cosine', 'NegativeInnerProduct', 'neg_inner']:
            self.metric = 'cosine'
            distance = falconn.DistanceFunction.NegativeInnerProduct
        else:
            warnings.warn(f'Invalid metric "{self.metric}". Using "sqeuclidean" instead')
            distance = falconn.DistanceFunction.EuclideanSquared

        # Set up the LSH index
        lsh_construction_params = falconn.get_default_parameters(*X.shape,
                                                                 distance=distance)
        lsh_index = falconn.LSHIndex(lsh_construction_params)
        lsh_index.setup(X)

        self.X_train_ = X
        self.y_train_ = y
        self.index_ = lsh_index

        return self

    def kneighbors(self, X: np.ndarray, n_candidates: int = None, return_distance: bool = True):
        check_is_fitted(self, ["index_", 'X_train_'])

        # Check the n_neighbors parameter
        if n_candidates is None:
            n_candidates = self.n_candidates
        elif n_candidates <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_candidates:d}")
        else:
            if not np.issubdtype(type(n_candidates), np.integer):
                raise TypeError(f"n_neighbors does not take {type(n_candidates)} value, enter integer value")

        # Constructing a query object
        query = self.index_.construct_query_object()
        query.set_num_probes(self.num_probes)

        if return_distance:
            if self.metric == 'sqeuclidean':
                distances = partial(euclidean_distances, squared=True)
            elif self.metric == 'cosine':
                distances = cosine_distances
            else:
                raise ValueError(f'Internal error: unrecognized metric "{self.metric}"')

        # Allocate memory for neighbor indices (and distances)
        n_objects = X.shape[0]
        neigh_ind = np.empty((n_objects, n_candidates), dtype=np.int32)
        if return_distance:
            neigh_dist = np.empty_like(neigh_ind, dtype=X.dtype)

        # If verbose, show progress bar on the search loop
        if not self.verbose:
            enumerate_X = enumerate(X)
        else:
            enumerate_X = tqdm(enumerate(X),
                               desc='LSH',
                               total=X.shape[0], )
        for i, x in enumerate_X:
            # LSH will find object itself as 1-NN, but we remove it here
            knn = np.array(query.find_k_nearest_neighbors(x, k=n_candidates + 1))[1:]
            neigh_ind[i, :knn.size] = knn

            if return_distance:
                neigh_dist[i, :knn.size] = distances(x.reshape(1, -1), self.X_train_[knn])

            # LSH may yield fewer neighbors than n_neighbors.
            # We set distances to NaN, and indices to -1
            if knn.size < n_candidates:
                neigh_ind[i, knn.size:] = -1
                if return_distance:
                    neigh_dist[i, knn.size:] = np.nan

        self.queryable_ = query

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
