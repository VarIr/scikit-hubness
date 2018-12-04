#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUBNESS package available at
https://github.com/OFAI/hubness/
The HUBNESS package is licensed under the terms of the GNU GPLv3.

(c) 2018, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI) and
University of Vienna, Division of Computational Systems Biology (CUBE)
Contact: <roman.feldbauer@ofai.at>

Adapted from scikit-learn codebase.
"""
from functools import partial
from distutils.version import LooseVersion
import warnings

import numpy as np
from scipy.sparse import issparse

from sklearn.neighbors.base import NeighborsBase as SklearnNeighborsBase
from sklearn.neighbors.base import KNeighborsMixin as SklearnKNeighborsMixin
from sklearn.neighbors.ball_tree import BallTree
from sklearn.neighbors.kd_tree import KDTree
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS, pairwise_distances_chunked
from sklearn.utils import check_array, gen_even_slices
from sklearn.utils.validation import check_is_fitted
try:
    from sklearn.utils import Parallel, delayed, effective_n_jobs
except ImportError:
    from sklearn.externals.joblib import Parallel, delayed, effective_n_jobs

from .lsh import LSH
from .hnsw import HNSW


# from abc import ABCMeta, abstractmethod
#
# from sklearn.base import BaseEstimator
# from sklearn.utils import check_X_y,
# from sklearn.utils.multiclass import check_classification_targets
# from sklearn.externals import six
#
# from sklearn.utils._joblib import __version__ as joblib_version
# from sklearn.exceptions import DataConversionWarning


VALID_METRICS = dict(lsh=LSH.valid_metrics,
                     hnsw=HNSW.valid_metrics,
                     ball_tree=BallTree.valid_metrics,
                     kd_tree=KDTree.valid_metrics,
                     # The following list comes from the
                     # sklearn.metrics.pairwise doc string
                     brute=(list(PAIRWISE_DISTANCE_FUNCTIONS.keys()) +
                            ['braycurtis', 'canberra', 'chebyshev',
                             'correlation', 'cosine', 'dice', 'hamming',
                             'jaccard', 'kulsinski', 'mahalanobis',
                             'matching', 'minkowski', 'rogerstanimoto',
                             'russellrao', 'seuclidean', 'sokalmichener',
                             'sokalsneath', 'sqeuclidean',
                             'yule', 'wminkowski']))


VALID_METRICS_SPARSE = dict(lsh=[],
                            hnsw=[],
                            ball_tree=[],
                            kd_tree=[],
                            brute=PAIRWISE_DISTANCE_FUNCTIONS.keys())


class NeighborsBase(SklearnNeighborsBase):

    def _check_algorithm_metric(self):
        if self.algorithm not in ['auto', 'brute',
                                  'kd_tree', 'ball_tree',
                                  'lsh', 'hnsw']:
            raise ValueError("unrecognized algorithm: '%s'" % self.algorithm)

        if self.algorithm == 'auto':
            if self.metric == 'precomputed':
                alg_check = 'brute'
            # elif self.metric in VALID_METRICS['hnsw']:
            #    alg_check = 'hnsw'
            elif (callable(self.metric) or
                  self.metric in VALID_METRICS['ball_tree']):
                alg_check = 'ball_tree'
            else:
                alg_check = 'brute'
        else:
            alg_check = self.algorithm

        if callable(self.metric):
            if self.algorithm in ['kd_tree', 'lsh', 'hnsw']:
                # callable metric is only valid for brute force and ball_tree
                raise ValueError(f"{self.algorithm} algorithm does not support callable metric '{self.metric}'")
        elif self.metric not in VALID_METRICS[alg_check]:
            raise ValueError(f"Metric '{self.metric}' not valid. Use "
                             f"sorted(hubness.neighbors.VALID_METRICS['{alg_check}']) "
                             f"to get valid options. "
                             f"Metric can also be a callable function.")

        if self.metric_params is not None and 'p' in self.metric_params:
            warnings.warn("Parameter p is found in metric_params. "
                          "The corresponding parameter from __init__ "
                          "is ignored.", SyntaxWarning, stacklevel=3)
            effective_p = self.metric_params['p']
        else:
            effective_p = self.p

        if self.metric in ['wminkowski', 'minkowski'] and effective_p <= 0:
            raise ValueError("p must be greater than zero for minkowski metric")

    def _fit(self, X):
        self._check_algorithm_metric()
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        effective_p = self.effective_metric_params_.get('p', self.p)
        if self.metric in ['wminkowski', 'minkowski']:
            self.effective_metric_params_['p'] = effective_p

        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == 'minkowski':
            p = self.effective_metric_params_.pop('p', 2)
            if p < 1:
                raise ValueError("p must be greater than one "
                                 "for minkowski metric")
            elif p == 1:
                self.effective_metric_ = 'manhattan'
            elif p == 2:
                self.effective_metric_ = 'euclidean'
            elif p == np.inf:
                self.effective_metric_ = 'chebyshev'
            else:
                self.effective_metric_params_['p'] = p

        if isinstance(X, NeighborsBase):
            self._fit_X = X._fit_X
            self._tree = X._tree
            self._fit_method = X._fit_method
            return self

        elif isinstance(X, BallTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'ball_tree'
            return self

        elif isinstance(X, KDTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = 'kd_tree'
            return self

        elif isinstance(X, LSH):
            # TODO implement
            pass

        elif isinstance(X, HNSW):
            # TODO implement
            pass

        X = check_array(X, accept_sparse='csr')

        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError(f"n_samples must be greater than 0 (but was {n_samples}.")

        if issparse(X):
            if self.algorithm not in ('auto', 'brute'):
                warnings.warn("cannot use tree with sparse input: "
                              "using brute force")
            if self.effective_metric_ not in VALID_METRICS_SPARSE['brute'] \
                    and not callable(self.effective_metric_):
                raise ValueError(f"Metric '{self.effective_metric_}' not valid for sparse input. "
                                 f"Use sorted(sklearn.neighbors.VALID_METRICS_SPARSE['brute']) "
                                 f"to get valid options. Metric can also be a callable function.")
            self._fit_X = X.copy()
            self._tree = None
            self._fit_method = 'brute'
            return self

        self._fit_method = self.algorithm
        self._fit_X = X

        if self._fit_method == 'auto':
            # A tree approach is better for small number of neighbors,
            # and KDTree is generally faster when available
            if ((self.n_neighbors is None or
                 self.n_neighbors < self._fit_X.shape[0] // 2) and
                    self.metric != 'precomputed'):
                if self.effective_metric_ in VALID_METRICS['kd_tree']:
                    self._fit_method = 'kd_tree'
                elif (callable(self.effective_metric_) or
                      self.effective_metric_ in VALID_METRICS['ball_tree']):
                    self._fit_method = 'ball_tree'
                else:
                    self._fit_method = 'brute'
            else:
                self._fit_method = 'brute'

        if self._fit_method == 'ball_tree':
            self._tree = BallTree(X, self.leaf_size,
                                  metric=self.effective_metric_,
                                  **self.effective_metric_params_)
        elif self._fit_method == 'kd_tree':
            self._tree = KDTree(X, self.leaf_size,
                                metric=self.effective_metric_,
                                **self.effective_metric_params_)
        elif self._fit_method == 'brute':
            self._tree = None
        elif self._fit_method == 'lsh':
            self._tree = None
        elif self._fit_method == 'hnsw':
            self._tree = None
        else:
            raise ValueError("algorithm = '%s' not recognized"
                             % self.algorithm)

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(
                    "Expected n_neighbors > 0. Got %d" %
                    self.n_neighbors
                )
            else:
                if not np.issubdtype(type(self.n_neighbors), np.integer):
                    raise TypeError(
                        "n_neighbors does not take %s value, "
                        "enter integer value" %
                        type(self.n_neighbors))

        return self


class KNeighborsMixin(SklearnKNeighborsMixin):
    """Mixin for k-neighbors searches.
    NOTE: adapted from scikit-learn. """

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.
        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned
        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        Examples
        --------
        In the following example, we construct a NeighborsClassifier
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]
        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=1)
        >>> neigh.fit(samples) # doctest: +ELLIPSIS
        NearestNeighbors(algorithm='auto', leaf_size=30, ...)
        >>> print(neigh.kneighbors([[1., 1., 1.]])) # doctest: +ELLIPSIS
        (array([[0.5]]), array([[2]]))
        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:
        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False) # doctest: +ELLIPSIS
        array([[1],
               [2]]...)
        """
        check_is_fitted(self, "_fit_method")

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(
                "Expected n_neighbors > 0. Got %d" %
                n_neighbors
            )
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" %
                    type(n_neighbors))

        if X is not None:
            query_is_train = False
            X = check_array(X, accept_sparse='csr')
        else:
            query_is_train = True
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1

        train_size = self._fit_X.shape[0]
        if n_neighbors > train_size:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (train_size, n_neighbors)
            )
        n_samples, _ = X.shape
        sample_range = np.arange(n_samples)[:, None]

        n_jobs = effective_n_jobs(self.n_jobs)
        if self._fit_method == 'brute':

            reduce_func = partial(self._kneighbors_reduce_func,
                                  n_neighbors=n_neighbors,
                                  return_distance=return_distance)

            # for efficiency, use squared euclidean distances
            kwds = ({'squared': True} if self.effective_metric_ == 'euclidean'
                    else self.effective_metric_params_)

            result = pairwise_distances_chunked(
                X, self._fit_X, reduce_func=reduce_func,
                metric=self.effective_metric_, n_jobs=n_jobs,
                **kwds)

        elif self._fit_method in ['ball_tree', 'kd_tree']:
            if issparse(X):
                raise ValueError(
                    "%s does not work with sparse matrices. Densify the data, "
                    "or set algorithm='brute'" % self._fit_method)
            if LooseVersion(joblib_version) < LooseVersion('0.12'):
                # Deal with change of API in joblib
                delayed_query = delayed(self._tree.query,
                                        check_pickle=False)
                parallel_kwargs = {"backend": "threading"}
            else:
                delayed_query = delayed(self._tree.query)
                parallel_kwargs = {"prefer": "threads"}
            result = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(
                    X[s], n_neighbors, return_distance)
                for s in gen_even_slices(X.shape[0], n_jobs)
            )
        else:
            raise ValueError("internal: _fit_method not recognized")

        if return_distance:
            dist, neigh_ind = zip(*result)
            result = np.vstack(dist), np.vstack(neigh_ind)
        else:
            result = np.vstack(result)

        if not query_is_train:
            return result
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                dist, neigh_ind = result
            else:
                neigh_ind = result

            sample_mask = neigh_ind != sample_range

            # Corner case: When the number of duplicates are more
            # than the number of neighbors, the first NN will not
            # be the sample, but a duplicate.
            # In that case mask the first duplicate.
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False

            neigh_ind = np.reshape(
                neigh_ind[sample_mask], (n_samples, n_neighbors - 1))

            if return_distance:
                dist = np.reshape(
                    dist[sample_mask], (n_samples, n_neighbors - 1))
                return dist, neigh_ind

        return neigh_ind
