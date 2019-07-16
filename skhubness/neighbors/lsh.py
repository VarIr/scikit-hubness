# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations

from functools import partial
import warnings
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.validation import check_is_fitted, check_array
import falconn
from tqdm.auto import tqdm

from .approximate_neighbors import ApproximateNearestNeighbor
__all__ = ['LSH']


class LSH(ApproximateNearestNeighbor):
    valid_metrics = ['euclidean', 'l2', 'minkowski', 'squared_euclidean', 'sqeuclidean',
                     'cosine', 'neg_inner', 'NegativeInnerProduct']

    def __init__(self, n_candidates: int = 5, radius: float = 1., metric: str = 'euclidean', num_probes: int = 50,
                 n_jobs: int = 1, verbose: int = 0):
        super().__init__(n_candidates=n_candidates, metric=metric, n_jobs=n_jobs, verbose=verbose)
        self.num_probes = num_probes
        self.radius = radius

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> LSH:
        """ Setup the LSH index from training data. """
        X = check_array(X, dtype=[np.float32, np.float64])

        if self.metric in ['euclidean', 'l2', 'minkowski']:
            self.metric = 'euclidean'
            distance = falconn.DistanceFunction.EuclideanSquared
        elif self.metric in ['squared_euclidean', 'sqeuclidean']:
            self.metric = 'sqeuclidean'
            distance = falconn.DistanceFunction.EuclideanSquared
        elif self.metric in ['cosine', 'NegativeInnerProduct', 'neg_inner']:
            self.metric = 'cosine'
            distance = falconn.DistanceFunction.NegativeInnerProduct
        else:
            warnings.warn(f'Invalid metric "{self.metric}". Using "euclidean" instead')
            self.metric = 'euclidean'
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

    def kneighbors(self, X: np.ndarray = None, n_candidates: int = None, return_distance: bool = True):
        check_is_fitted(self, ["index_", 'X_train_'])

        # Check the n_neighbors parameter
        if n_candidates is None:
            n_candidates = self.n_candidates
        elif n_candidates <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_candidates:d}")
        else:
            if not np.issubdtype(type(n_candidates), np.integer):
                raise TypeError(f"n_neighbors does not take {type(n_candidates)} value, enter integer value")

        if X is not None:
            X = check_array(X, dtype=self.X_train_.dtype)
            query_is_train = False
            X = check_array(X, accept_sparse='csr')
            n_retrieve = n_candidates
        else:
            query_is_train = True
            X = self.X_train_
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_retrieve = n_candidates + 1

        # Configure the LSH query object
        query = self.index_.construct_query_object()
        query.set_num_probes(self.num_probes)

        if return_distance:
            if self.metric == 'euclidean':
                distances = partial(euclidean_distances, squared=False)
            elif self.metric == 'sqeuclidean':
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
        if self.verbose:
            enumerate_X = tqdm(enumerate(X),
                               desc='LSH',
                               total=X.shape[0], )
        else:
            enumerate_X = enumerate(X)
        for i, x in enumerate_X:
            knn = np.array(query.find_k_nearest_neighbors(x, k=n_retrieve))
            if query_is_train:
                knn = knn[1:]
            neigh_ind[i, :knn.size] = knn

            if return_distance:
                neigh_dist[i, :knn.size] = distances(x.reshape(1, -1), self.X_train_[knn])

            # LSH may yield fewer neighbors than n_neighbors.
            # We set distances to NaN, and indices to -1
            if knn.size < n_candidates:
                neigh_ind[i, knn.size:] = -1
                if return_distance:
                    neigh_dist[i, knn.size:] = np.nan

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind

    def radius_neighbors(self, X: np.ndarray = None, radius: float = None, return_distance: bool = True):
        """ TODO add docstring

        Notes
        -----
        From the falconn docs: radius can be negative, and for the distance function
        'negative_inner_product' it actually makes sense.
        """
        check_is_fitted(self, ["index_", 'X_train_'])

        # Constructing a query object
        query = self.index_.construct_query_object()
        query.set_num_probes(self.num_probes)

        if return_distance:
            if self.metric == 'euclidean':
                distances = partial(euclidean_distances, squared=False)
            elif self.metric == 'sqeuclidean':
                distances = partial(euclidean_distances, squared=True)
            elif self.metric == 'cosine':
                distances = cosine_distances
            else:
                raise ValueError(f'Internal error: unrecognized metric "{self.metric}"')

        if X is not None:
            query_is_train = False
            X = check_array(X, accept_sparse='csr', dtype=self.X_train_.dtype)
        else:
            query_is_train = True
            X = self.X_train_

        if radius is None:
            radius = self.radius
        # LSH uses squared Euclidean internally
        if self.metric == 'euclidean':
            radius *= radius

        # Add a small number to imitate <= threshold
        radius += 1e-7

        # Allocate memory for neighbor indices (and distances)
        n_objects = X.shape[0]
        neigh_ind = np.empty(n_objects, dtype='object')
        if return_distance:
            neigh_dist = np.empty_like(neigh_ind)

        # If verbose, show progress bar on the search loop
        if self.verbose:
            enumerate_X = tqdm(enumerate(X),
                               desc='LSH',
                               total=X.shape[0], )
        else:
            enumerate_X = enumerate(X)
        for i, x in enumerate_X:
            knn = np.array(query.find_near_neighbors(x, threshold=radius))
            if len(knn) == 0:
                knn = np.array([], dtype=int)
            else:
                if query_is_train:
                    knn = knn[1:]
            neigh_ind[i] = knn

            if return_distance:
                if len(knn):
                    neigh_dist[i] = distances(x.reshape(1, -1), self.X_train_[knn]).ravel()
                else:
                    neigh_dist[i] = np.array([])

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
