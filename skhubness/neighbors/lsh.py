# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations

from functools import partial
import sys
from typing import Tuple, Union
import warnings
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

try:
    import puffinn
except ImportError:
    puffinn = None  # pragma: no cover
try:
    import falconn
except ImportError:
    falconn = None  # pragma: no cover

from tqdm.auto import tqdm
from .approximate_neighbors import ApproximateNearestNeighbor
from ..utils.check import check_n_candidates

__all__ = ['FalconnLSH', 'PuffinnLSH', ]


class PuffinnLSH(BaseEstimator, ApproximateNearestNeighbor):
    """ Wrap Puffinn LSH for scikit-learn compatibility.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "jaccard".
        Other metrics are partially supported, such as 'euclidean', 'sqeuclidean'.
        In these cases, 'angular' distances are used to find the candidate set
        of neighbors with LSH among all indexed objects, and (squared) Euclidean
        distances are subsequently only computed for the candidates.
    memory: int, default = 1GB
        Max memory usage
    recall: float, default = 0.90
        Probability of finding the true nearest neighbors among the candidates
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures
    """
    valid_metrics = ["angular", "cosine", "euclidean", "sqeuclidean", "minkowski",
                     "jaccard",
                     ]
    metric_map = {'euclidean': 'angular',
                  'sqeuclidean': 'angular',
                  'minkowski': 'angular',
                  'cosine': 'angular',
                  }

    def __init__(self, n_candidates: int = 5,
                 metric: str = 'euclidean',
                 memory: int = 1024**3,
                 recall: float = 0.9,
                 n_jobs: int = 1,
                 verbose: int = 0,
                 ):

        if puffinn is None:  # pragma: no cover
            raise ImportError(f'Please install the `puffinn` package, before using this class:\n'
                              f'$ git clone https://github.com/puffinn/puffinn.git\n'
                              f'$ cd puffinn\n'
                              f'$ python3 setup.py build\n'
                              f'$ pip install .\n') from None

        super().__init__(n_candidates=n_candidates,
                         metric=metric,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         )
        self.memory = memory
        self.recall = recall

    def fit(self, X, y=None) -> PuffinnLSH:
        """ Build the puffinn LSH index and insert data from X.

        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored

        Returns
        -------
        self: Puffinn
            An instance of Puffinn with a built index
        """
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
            self.y_train_ = y

        if self.metric not in self.valid_metrics:
            warnings.warn(f'Invalid metric "{self.metric}". Using "euclidean" instead')
            self.metric = 'euclidean'
        try:
            self._effective_metric = self.metric_map[self.metric]
        except KeyError:
            self._effective_metric = self.metric

        # Reduce default memory consumption for unit tests
        if "pytest" in sys.modules:
            memory = 3*1024**2
        else:
            memory = self.memory  # pragma: no cover

        # Construct the index
        index = puffinn.Index(self._effective_metric,
                              X.shape[1],
                              memory,
                              )

        disable_tqdm = False if self.verbose else True
        for v in tqdm(X, desc='Indexing', disable=disable_tqdm):
            index.insert(v.tolist())
        index.rebuild()

        self.index_ = index
        self.n_indexed_ = X.shape[0]
        self.X_indexed_norm_ = np.linalg.norm(X, ord=2, axis=1).reshape(-1, 1)

        return self

    def kneighbors(self, X=None, n_candidates=None, return_distance=True) -> Union[Tuple[np.array, np.array], np.array]:
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
        check_is_fitted(self, 'index_')
        index = self.index_

        if n_candidates is None:
            n_candidates = self.n_candidates
        n_candidates = check_n_candidates(n_candidates)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        if X is None:
            n_query = self.n_indexed_
            X = np.array([index.get(i) for i in range(n_query)])
            search_from_index = True
        else:
            X = check_array(X)
            n_query = X.shape[0]
            search_from_index = False

        dtype = X.dtype

        # If chosen metric is not among the natively supported ones, reorder the neighbors
        reorder = True if self.metric not in ('angular', 'cosine', 'jaccard') else False

        # If fewer candidates than required are found for a query,
        # we save index=-1 and distance=NaN
        neigh_ind = -np.ones((n_query, n_candidates),
                             dtype=np.int32)
        if return_distance or reorder:
            neigh_dist = np.empty_like(neigh_ind,
                                       dtype=dtype) * np.nan
        metric = 'cosine' if self.metric == 'angular' else self.metric

        disable_tqdm = False if self.verbose else True

        if search_from_index:  # search indexed against indexed
            for i in tqdm(range(n_query),
                          desc='Querying',
                          disable=disable_tqdm,
                          ):
                # Find the approximate nearest neighbors.
                # Each of the true `n_candidates` nearest neighbors
                # has at least `recall` chance of being found.
                ind = index.search_from_index(i, n_candidates, self.recall, )

                neigh_ind[i, :len(ind)] = ind
                if return_distance or reorder:
                    X_neigh_denormalized = \
                        X[ind] * self.X_indexed_norm_[ind].reshape(len(ind), -1)
                    neigh_dist[i, :len(ind)] = pairwise_distances(X[i:i+1, :] * self.X_indexed_norm_[i],
                                                                  X_neigh_denormalized,
                                                                  metric=metric,
                                                                  )
        else:  # search new query against indexed
            for i, x in tqdm(enumerate(X),
                             desc='Querying',
                             disable=disable_tqdm,
                             ):
                # Find the approximate nearest neighbors.
                # Each of the true `n_candidates` nearest neighbors
                # has at least `recall` chance of being found.
                ind = index.search(x.tolist(),
                                   n_candidates,
                                   self.recall,
                                   )

                neigh_ind[i, :len(ind)] = ind
                if return_distance or reorder:
                    X_neigh_denormalized =\
                        np.array([index.get(i) for i in ind]) * self.X_indexed_norm_[ind].reshape(len(ind), -1)
                    neigh_dist[i, :len(ind)] = pairwise_distances(x.reshape(1, -1),
                                                                  X_neigh_denormalized,
                                                                  metric=metric,
                                                                  )

        if reorder:
            sort = np.argsort(neigh_dist, axis=1)
            neigh_dist = np.take_along_axis(neigh_dist, sort, axis=1)
            neigh_ind = np.take_along_axis(neigh_ind, sort, axis=1)

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind


class FalconnLSH(ApproximateNearestNeighbor):
    """Wrapper for using falconn LSH

    Falconn is an approximate nearest neighbor library,
    that uses multiprobe locality-sensitive hashing.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    radius: float or None, optional, default = None
        Retrieve neighbors within this radius.
        Can be negative: See Notes.
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "euclidean", "manhattan", "hamming", "dot"
    num_probes: int, default = 50
        The number of buckets the query algorithm probes.
        The higher number of probes is, the better accuracy one gets,
        but the slower queries are.
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures

    Notes
    -----
    From the falconn docs: radius can be negative, and for the distance function
    'negative_inner_product' it actually makes sense.
    """
    valid_metrics = ['euclidean', 'l2', 'minkowski', 'squared_euclidean', 'sqeuclidean',
                     'cosine', 'neg_inner', 'NegativeInnerProduct']

    def __init__(self, n_candidates: int = 5, radius: float = 1., metric: str = 'euclidean', num_probes: int = 50,
                 n_jobs: int = 1, verbose: int = 0):

        if falconn is None:  # pragma: no cover
            raise ImportError(f'Please install the `falconn` package, before using this class:\n'
                              f'$ pip install falconn') from None

        super().__init__(n_candidates=n_candidates,
                         metric=metric,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         )
        self.num_probes = num_probes
        self.radius = radius

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> FalconnLSH:
        """ Setup the LSH index from training data.

        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored

        Returns
        -------
        self: FalconnLSH
            An instance of LSH with a built index
        """
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
        disable_tqdm = False if self.verbose else True
        for i, x in tqdm(enumerate(X),
                         desc='LSH',
                         disable=disable_tqdm,
                         ):
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

    def radius_neighbors(self, X: np.ndarray = None,
                         radius: float = None,
                         return_distance: bool = True) -> Union[Tuple[np.array, np.array], np.array]:
        """ Retrieve neighbors within a certain radius.

        Parameters
        ----------
        X: np.array or None, optional, default = None
            Query objects. If None, search among the indexed objects.
        radius: float or None, optional, default = None
            Retrieve neighbors within this radius.
            Can be negative: See Notes.
        return_distance: bool, default = True
            If return_distance, will return distances and indices to neighbors.
            Else, only return the indices.

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
        disable_tqdm = False if self.verbose else True
        for i, x in tqdm(enumerate(X),
                         desc='LSH',
                         disable=disable_tqdm,
                         ):
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
