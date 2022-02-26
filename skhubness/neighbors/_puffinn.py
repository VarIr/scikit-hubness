# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations

from functools import partial
import multiprocessing as mp
from typing import Tuple, Union
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from tqdm.auto import tqdm
try:
    import puffinn
except ImportError:
    puffinn = None  # pragma: no cover

from .approximate_neighbors import ApproximateNearestNeighbor
from ..utils.check import check_n_candidates

__all__ = [
    "LegacyPuffinn",
    "PuffinnTransformer",
]


class PuffinnTransformer(BaseEstimator, TransformerMixin):
    """ Approximate nearest neighbors retrieval with Puffinn.

    Compatible with sklearn's KNeighborsTransformer.
    "PUFFINN is an easily configurable library for finding the approximate nearest neighbors of arbitrary points.
    The only necessary parameters are the allowed space usage and the recall. Each near neighbor is guaranteed
    to be found with the probability given by the recall, regardless of the difficulty of the query.
    Under the hood PUFFINN uses Locality Sensitive Hashing with an adaptive query mechanism. This means that
    the algorithm works for any similarity measure where a Locality Sensitive Hash family exists.
    Currently Cosine similarity is supported using SimHash or cross-polytope LSH
    and Jaccard similarity is supported using MinHash." (Puffinn authors)

    Parameters
    ----------
    n_neighbors: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = "angular"
        Distance metric, allowed are "angular", "jaccard".
    memory: int, default = None
        Max memory usage [B]. If None, determined heuristically.
    recall: float, default = 0.90
        Probability of finding the true nearest neighbors among the candidates
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show a progress bar on indexing and querying.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures
    """
    valid_metrics = [
        "angular",
        "jaccard",
    ]
    _sklearn_metric = {
        "angular": "cosine",
        "jaccard": "jaccard",
    }

    def __init__(
            self,
            n_neighbors: int = 5,
            metric: str = "angular",
            memory: int = None,
            recall: float = 0.9,
            n_jobs: int = 1,
            verbose: int = 0,
    ):

        if puffinn is None:  # pragma: no cover
            raise ImportError(
                "Please install the puffinn package before using the PuffinnTransformer:\n"
                "git clone https://github.com/puffinn/puffinn.git\n"
                "cd puffinn\n"
                "python setup.py build\n"
                "pip install .\n",
            ) from None

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.memory = memory
        self.recall = recall
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None) -> PuffinnTransformer:
        """ Build the puffinn LSH index and insert data from X.

        Parameters
        ----------
        X: array-like
            Data to be indexed
        y: ignored

        Returns
        -------
        self: PuffinnTransformer
            An instance of PuffinnTransformer with a built index
        """
        X: np.ndarray = check_array(X, accept_sparse=False)  # noqa
        n_samples, n_features = X.shape
        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features

        if self.memory is None:
            # This was found heuristically. Idea being, the index might take
            # some overhead space even for small data sets, plus space for
            # the data itself. Additional optimizations might make sense.
            memory = 1_000_000 + np.multiply(*X.shape) * 8
        else:
            memory = self.memory

        # Construct the index
        index = puffinn.Index(
            self.metric,
            X.shape[1],
            memory,
        )

        # No progress bar here, as time is spent in rebuild(), not the loop
        for v in X:
            index.insert(v.tolist())
        try:
            index.rebuild()
        except ValueError as e:
            # Raise a more helpful error message, when the user supplied puffinn
            # with too little memory.
            if "insufficient memory" in str(e):
                raise ValueError(f"Insufficient memory (got {memory} B): "
                                 f"Increase value for parameter `memory`.") from None
            else:
                raise e

        self.neighbor_index_ = index

        return self

    def transform(self, X) -> csr_matrix:
        """ Create k-neighbors graph for the query objects in X.

        Parameters
        ----------
        X : array-like
            Query objects

        Returns
        -------
        kneighbors_graph : csr_matrix
            The retrieved approximate nearest neighbors in the index for each query.
        """
        check_is_fitted(self, "neighbor_index_")
        X: np.ndarray = check_array(X, accept_sparse=False)  # noqa

        n_samples_transform, n_features_transform = X.shape
        if n_features_transform != self.n_features_in_:
            raise ValueError(f"Shape of X ({n_features_transform} features) does not match "
                             f"shape of fitted data ({self.n_features_in_} features.")

        index = self.neighbor_index_

        tqdm_fmt = partial(
            tqdm,
            desc="puffinn transform",
            disable=self.verbose < 1,
            unit=" vector",
            unit_scale=True,
        )
        indices = []
        for i, x in enumerate(tqdm_fmt(X)):
            # Find the approximate nearest neighbors.
            # Each of the true `n_candidates` nearest neighbors
            # has at least `recall` chance of being found.
            ind = index.search(
                vec=x.tolist(),
                k=self.n_neighbors,
                recall=self.recall,
            )
            ind = np.array(ind)
            indices.append(ind)

        indices = np.vstack(indices)
        distances = np.zeros_like(indices, dtype=X.dtype)
        indptr = np.array([0, *np.cumsum([len(ind) for ind in indices])])

        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_in_),
        )

        return kneighbors_graph


class LegacyPuffinn(BaseEstimator, ApproximateNearestNeighbor):
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
    memory: int, default = None
        Max memory usage. If None, determined heuristically.
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
                 memory: int = None,
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

    def fit(self, X, y=None) -> LegacyPuffinn:
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

        # Larger memory means many iterations (time-recall trade-off)
        memory = max(np.multiply(*X.shape) * 8 * 500, 1024**2)
        if self.memory is not None:
            memory = max(self.memory, memory)

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
