# -*- coding: utf-8 -*-

""" Base and mixin classes for nearest neighbors.

Adapted from scikit-learn codebase at
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/base.py.
"""

# Authors: Jake Vanderplas <vanderplas@astro.washington.edu>
#          Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Sparseness support by Lars Buitinck
#          Multi-output support by Arnaud Joly <a.joly@ulg.ac.be>
#          Hubness reduction support by Roman Feldbauer <roman.feldbauer@univie.ac.at>
#
# License: BSD 3 clause (C) INRIA, University of Amsterdam

from functools import partial
from distutils.version import LooseVersion
from typing import List, Tuple
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
from joblib import Parallel, delayed, effective_n_jobs, __version__ as joblib_version

from .lsh import LSH
from .hnsw import HNSW
from ..reduction import NoHubnessReduction, LocalScaling, MutualProximity


# from abc import ABCMeta, abstractmethod
#
# from sklearn.base import BaseEstimator
# from sklearn.utils import check_X_y,
# from sklearn.utils.multiclass import check_classification_targets
# from sklearn.externals import six
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


def _check_weights(weights):
    """Check to make sure weights are valid"""
    if weights in (None, 'uniform', 'distance'):
        return weights
    elif callable(weights):
        return weights
    else:
        raise ValueError("weights not recognized: should be 'uniform', "
                         "'distance', or a callable function")


def _get_weights(dist, weights):
    """Get the weights from an array of distances and a parameter ``weights``
    Parameters
    ----------
    dist : ndarray
        The input distances
    weights : {'uniform', 'distance' or a callable}
        The kind of weighting used
    Returns
    -------
    weights_arr : array of the same shape as ``dist``
        if ``weights == 'uniform'``, then returns None
    """
    if weights in (None, 'uniform'):
        return None
    elif weights == 'distance':
        # if user attempts to classify a point that was zero distance from one
        # or more training points, those training points are weighted as 1.0
        # and the other points as 0.0
        if dist.dtype is np.dtype(object):
            for point_dist_i, point_dist in enumerate(dist):
                # check if point_dist is iterable
                # (ex: RadiusNeighborClassifier.predict may set an element of
                # dist to 1e-6 to represent an 'outlier')
                if hasattr(point_dist, '__contains__') and 0. in point_dist:
                    dist[point_dist_i] = point_dist == 0.
                else:
                    dist[point_dist_i] = 1. / point_dist
        else:
            with np.errstate(divide='ignore'):
                dist = 1. / dist
            inf_mask = np.isinf(dist)
            inf_row = np.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
        return dist
    elif callable(weights):
        return weights(dist)
    else:
        raise ValueError("weights not recognized: should be 'uniform', "
                         "'distance', or a callable function")


class NeighborsBase(SklearnNeighborsBase):
    """Base class for nearest neighbors estimators."""

    def __init__(self, n_neighbors=None, radius=None,
                 algorithm='auto', algorithm_params: dict = None,
                 hubness: str = None, hubness_params: dict = None,
                 leaf_size=30, metric='minkowski', p=2, metric_params=None,
                 n_jobs=None, verbose: int = 0, **kwargs):
        super().__init__(n_neighbors=n_neighbors,
                         radius=radius,
                         algorithm=algorithm,
                         leaf_size=leaf_size,
                         metric=metric, p=p, metric_params=metric_params,
                         n_jobs=n_jobs)
        self.algorithm_params = algorithm_params if algorithm_params is not None else {}
        self.hubness_params = hubness_params if hubness_params is not None else {}
        self.hubness = hubness
        # self.mp_distribution = mp_distribution
        # self.ls_method = ls_method
        self.verbose = verbose

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
        self._hubness_reduction_method = self.hubness

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
            self._index = LSH(verbose=self.verbose, **self.algorithm_params)
            self._index.fit(X)
        elif self._fit_method == 'hnsw':
            # TODO implement
            self._index = None
            raise NotImplementedError
        else:
            raise ValueError(f"algorithm = '{self.algorithm}' not recognized")

        if self._hubness_reduction_method is None:
            self._hubness_reduction = NoHubnessReduction()
        else:
            n_candidates = self.algorithm_params['n_candidates']
            neigh_train = self.kcandidates(n_neighbors=n_candidates, return_distance=True)
            # Remove self distances
            neigh_dist_train = neigh_train[0]  # [:, 1:]
            neigh_ind_train = neigh_train[1]  # [:, 1:]
            if self._hubness_reduction_method == 'ls':
                self._hubness_reduction = LocalScaling(verbose=self.verbose, **self.hubness_params)
            elif self._hubness_reduction_method == 'mp':
                self._hubness_reduction = MutualProximity(verbose=self.verbose, **self.hubness_params)
            elif self._hubness_reduction_method == 'snn':
                raise NotImplementedError
            elif self._hubness_reduction_method == 'simhubin':
                raise NotImplementedError
            else:
                raise ValueError(f'Hubness reduction algorithm = "{self._hubness_reduction_method}" not recognized.')
            self._hubness_reduction.fit(neigh_dist_train, neigh_ind_train, assume_sorted=False)

        if self.n_neighbors is not None:
            if self.n_neighbors <= 0:
                raise ValueError(f"Expected n_neighbors > 0. Got {self.n_neighbors:d}")
            else:
                if not np.issubdtype(type(self.n_neighbors), np.integer):
                    raise TypeError(
                        f"n_neighbors does not take {type(self.n_neighbors)} value, "
                        f"enter integer value"
                        )

        return self

    def kcandidates(self, X=None, n_neighbors=None, return_distance=True) -> np.ndarray or (np.ndarray, np.ndarray):
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
        elif self._fit_method in ['lsh']:
            # assume joblib>=0.12
            delayed_query = delayed(self._index.kneighbors)
            parallel_kwargs = {"prefer": "threads"}
            result = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(X[s], n_candidates=n_neighbors, return_distance=True)
                for s in gen_even_slices(X.shape[0], n_jobs)
            )
        elif self._fit_method in ['hnsw']:
            raise NotImplementedError
        else:
            raise ValueError(f"internal: _fit_method not recognized: {self._fit_method}.")

        if return_distance:
            try:
                dist, neigh_ind = zip(*result)
            except ValueError:
                pass  # LSH already passes the correct format
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


class KNeighborsMixin(SklearnKNeighborsMixin):
    """Mixin for k-neighbors searches.
    NOTE: adapted from scikit-learn. """

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """ TODO """

        check_is_fitted(self, ["_fit_method", "_hubness_reduction"])

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(f"n_neighbors does not take {type(n_neighbors)} value, enter integer value")

        # First obtain candidate neighbors
        query_dist, query_ind = self.kcandidates(X, n_neighbors, return_distance)

        # Second, reduce hubness
        hubness_reduced_query_dist, query_ind = self._hubness_reduction.transform(query_dist,
                                                                                  query_ind,
                                                                                  assume_sorted=True,)

        # Third, sort hubness reduced candidate neighbors to get the final k neighbors
        kth = np.arange(n_neighbors)
        mask = np.argpartition(hubness_reduced_query_dist, kth=kth)[:, :n_neighbors]
        hubness_reduced_query_dist = np.take_along_axis(hubness_reduced_query_dist, mask, axis=1)
        query_ind = np.take_along_axis(query_ind, mask, axis=1)

        if return_distance:
            return hubness_reduced_query_dist, query_ind
        else:
            return query_ind

    def kneighbors_old(self, X=None, n_neighbors=None, return_distance=True):
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
        elif self._fit_method in ['lsh']:
            # assume joblib>=0.12
            delayed_query = delayed(self._index.kneighbors)
            parallel_kwargs = {"prefer": "threads"}
            result = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(X[s], return_distance=True) for s in gen_even_slices(X.shape[0], n_jobs)
            )
        elif self._fit_method in ['hnsw']:
            raise NotImplementedError
        else:
            raise ValueError(f"internal: _fit_method not recognized: {self._fit_method}.")

        if return_distance:
            try:
                dist, neigh_ind = zip(*result)
            except ValueError:
                pass  # LSH already passes the correct format
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
