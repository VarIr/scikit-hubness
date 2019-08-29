# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

""" Base and mixin classes for nearest neighbors.

Adapted from scikit-learn codebase at
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/base.py.
"""

# Authors: Jake Vanderplas <vanderplas@astro.washington.edu>
#          Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Sparseness support by Lars Buitinck
#          Multi-output support by Arnaud Joly <a.joly@ulg.ac.be>
#          Hubness reduction and approximate nearest neighbor support by Roman Feldbauer <roman.feldbauer@univie.ac.at>
#
# License: BSD 3 clause (C) INRIA, University of Amsterdam

from functools import partial
import sys
import warnings

import numpy as np
from scipy.sparse import issparse, csr_matrix

from sklearn.neighbors.base import NeighborsBase as SklearnNeighborsBase
from sklearn.neighbors.base import KNeighborsMixin as SklearnKNeighborsMixin
from sklearn.neighbors.base import RadiusNeighborsMixin as SklearnRadiusNeighborsMixin
from sklearn.neighbors.base import UnsupervisedMixin, SupervisedFloatMixin, SupervisedIntegerMixin
from sklearn.neighbors.base import _tree_query_radius_parallel_helper
from sklearn.neighbors.ball_tree import BallTree
from sklearn.neighbors.kd_tree import KDTree
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS, pairwise_distances_chunked
from sklearn.utils import check_array, gen_even_slices
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed, effective_n_jobs

from .hnsw import HNSW
from ..reduction import NoHubnessReduction, LocalScaling, MutualProximity, DisSimLocal

# LSH library falconn does not support Windows
ON_PLATFORM_WINDOWS = sys.platform == 'win32'
if ON_PLATFORM_WINDOWS:
    from .approximate_neighbors import UnavailableANN
    LSH = UnavailableANN
else:
    from .lsh import LSH


__all__ = ['KNeighborsMixin', 'NeighborsBase', 'RadiusNeighborsMixin',
           'SupervisedFloatMixin', 'SupervisedIntegerMixin', 'UnsupervisedMixin',
           'VALID_METRICS', 'VALID_METRICS_SPARSE',
           ]

VALID_METRICS = dict(lsh=LSH.valid_metrics if not ON_PLATFORM_WINDOWS else [],
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
                            brute=(PAIRWISE_DISTANCE_FUNCTIONS.keys()
                                   - {'haversine'}),
                            )


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
        if algorithm_params is None:
            n_candidates = 1 if hubness is None else 100
            algorithm_params = {'n_candidates': n_candidates,
                                'metric': metric}
        self.algorithm_params = algorithm_params
        self.hubness_params = hubness_params if hubness_params is not None else {}
        self.hubness = hubness
        self.verbose = verbose
        self.kwargs = kwargs

    def _check_hubness_algorithm(self):
        if self.hubness not in ['mp', 'mutual_proximity',
                                'ls', 'local_scaling',
                                'dsl', 'dis_sim_loca',
                                None]:
            raise ValueError(f'Unrecognized hubness algorithm: {self.hubness}')

        # Users are allowed to use various identifiers for the algorithms,
        # but here we normalize them to the short abbreviations used downstream
        if self.hubness in ['mp', 'mutual_proximity']:
            self.hubness = 'mp'
        elif self.hubness in ['ls', 'local_scaling']:
            self.hubness = 'ls'
        elif self.hubness in ['dsl', 'dis_sim_local']:
            self.hubness = 'dsl'
        elif self.hubness is None:
            pass
        else:
            raise ValueError(f'Internal error: unknown hubness algorithm: {self.hubness}')

    def _check_algorithm_metric(self):
        if self.algorithm not in ['auto', 'brute',
                                  'kd_tree', 'ball_tree',
                                  'lsh', 'hnsw']:
            raise ValueError("unrecognized algorithm: '%s'" % self.algorithm)

        if self.algorithm == 'auto':
            if self.metric == 'precomputed':
                alg_check = 'brute'
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
                             f"sorted(skhubness.neighbors.VALID_METRICS['{alg_check}']) "
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

    def _check_algorithm_hubness_compatibility(self):
        if self.hubness == 'dsl':
            if self.metric in ['euclidean', 'minkowski']:
                self.metric = 'euclidean'  # DSL input must still be squared Euclidean
                self.hubness_params['squared'] = False
                if self.p != 2:
                    warnings.warn(f'DisSimLocal only supports squared Euclidean distances: Ignoring p={self.p}.')
            elif self.metric in ['sqeuclidean']:
                self.hubness_params['squared'] = True
            else:
                warnings.warn(f'DisSimLocal only supports squared Euclidean distances: Ignoring metric={self.metric}.')
                self.metric = 'euclidean'
                self.hubness_params['squared'] = True

    def _fit(self, X):
        self._check_algorithm_metric()
        self._check_hubness_algorithm()
        self._check_algorithm_hubness_compatibility()
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
            if p <= 0:
                raise ValueError(f"p must be greater than one for minkowski metric, "
                                 f"or in ]0, 1[ for fractional norms.")
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
            self._index = X._index
            self._hubness_reduction = X._hubness_reduction
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

        elif isinstance(X, (LSH, HNSW)):
            self._tree = None
            if isinstance(X, LSH):
                self._fit_X = X.X_train_
                self._fit_method = 'lsh'
            elif isinstance(X, HNSW):
                self._fit_method = 'hnsw'
            self._index = X
            # TODO enable hubness reduction here
            ...
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
            if self.hubness is not None:
                warnings.warn(f'cannot use hubness reduction with tree: disabling hubness reduction.')
                self.hubness = None
            self._hubness_reduction_method = None
            self._hubness_reduction = NoHubnessReduction()
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
            self._index = None

        if self._fit_method == 'ball_tree':
            self._tree = BallTree(X, self.leaf_size,
                                  metric=self.effective_metric_,
                                  **self.effective_metric_params_)
            self._index = None
        elif self._fit_method == 'kd_tree':
            self._tree = KDTree(X, self.leaf_size,
                                metric=self.effective_metric_,
                                **self.effective_metric_params_)
            self._index = None
        elif self._fit_method == 'brute':
            self._tree = None
            self._index = None
        elif self._fit_method == 'lsh':
            self._index = LSH(verbose=self.verbose, **self.algorithm_params)
            self._index.fit(X)
            self._tree = None
        elif self._fit_method == 'hnsw':
            self._index = HNSW(verbose=self.verbose, **self.algorithm_params)
            self._index.fit(X)
            self._tree = None
        else:
            raise ValueError(f"algorithm = '{self.algorithm}' not recognized")

        if self._hubness_reduction_method is None:
            self._hubness_reduction = NoHubnessReduction()
        else:
            n_candidates = self.algorithm_params['n_candidates']
            if 'include_self' in self.kwargs and self.kwargs['include_self']:
                neigh_train = self.kcandidates(X, n_neighbors=n_candidates, return_distance=True)
            else:
                neigh_train = self.kcandidates(n_neighbors=n_candidates, return_distance=True)
            # Remove self distances
            neigh_dist_train = neigh_train[0]  # [:, 1:]
            neigh_ind_train = neigh_train[1]  # [:, 1:]
            if self._hubness_reduction_method == 'ls':
                self._hubness_reduction = LocalScaling(verbose=self.verbose, **self.hubness_params)
            elif self._hubness_reduction_method == 'mp':
                self._hubness_reduction = MutualProximity(verbose=self.verbose, **self.hubness_params)
            elif self._hubness_reduction_method == 'dsl':
                self._hubness_reduction = DisSimLocal(verbose=self.verbose, **self.hubness_params)
            elif self._hubness_reduction_method == 'snn':
                raise NotImplementedError('feature not yet implemented')
            elif self._hubness_reduction_method == 'simhubin':
                raise NotImplementedError('feature not yet implemented')
            else:
                raise ValueError(f'Hubness reduction algorithm = "{self._hubness_reduction_method}" not recognized.')
            self._hubness_reduction.fit(neigh_dist_train, neigh_ind_train, X=X, assume_sorted=False)

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
        X : array-like, shape (n_query, n_features), or (n_query, n_indexed) if metric == 'precomputed'
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
        >>> from skhubness.neighbors import NearestNeighbors
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
            n_neighbors = self.algorithm_params['n_candidates']
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" %
                    type(n_neighbors))

        # The number of candidates must not be less than the number of neighbors used downstream
        if self.n_neighbors is not None:
            if n_neighbors < self.n_neighbors:
                n_neighbors = self.n_neighbors

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
            warnings.warn(f'n_candidates > n_samples. Setting n_candidates = n_samples.')
            n_neighbors = train_size
        n_samples, _ = X.shape
        sample_range = np.arange(n_samples)[:, None]

        n_jobs = effective_n_jobs(self.n_jobs)
        if self._fit_method == 'brute':

            # TODO handle sparse matrices here
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
            # require joblib >= 0.12
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
            # XXX nmslib supports multiple threads natively, so no joblib used here
            # Must pack results into list to match the output format of joblib
            result = self._index.kneighbors(X, n_candidates=n_neighbors, return_distance=True)
            result = [result, ]
        else:
            raise ValueError(f"internal: _fit_method not recognized: {self._fit_method}.")

        if return_distance:
            dist, neigh_ind = zip(*result)
            result = [np.atleast_2d(arr) for arr in [np.vstack(dist), np.vstack(neigh_ind)]]
        else:
            result = np.atleast_2d(np.vstack(result))

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
            neigh_ind = np.atleast_2d(neigh_ind)

            if return_distance:
                dist = np.reshape(
                    dist[sample_mask], (n_samples, n_neighbors - 1))
                dist = np.atleast_2d(dist)
                return dist, neigh_ind

        return neigh_ind


class KNeighborsMixin(SklearnKNeighborsMixin):
    """Mixin for k-neighbors searches.
    NOTE: adapted from scikit-learn. """

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """ TODO """

        check_is_fitted(self, ["_fit_method", "_hubness_reduction"], all_or_any=any)

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(f"Expected n_neighbors > 0. Got {n_neighbors}")
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(f"n_neighbors does not take {type(n_neighbors)} value, enter integer value")

        if X is not None:
            query_is_train = False
            X = check_array(X, accept_sparse='csr')
        else:
            query_is_train = True
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1

        train_size = self._fit_X.shape[0]
        if n_neighbors > train_size:
            raise ValueError(f"Expected n_neighbors <= n_samples, "
                             f"but n_samples = {train_size}, n_neighbors = {n_neighbors}")

        # First obtain candidate neighbors
        query_dist, query_ind = self.kcandidates(X, return_distance=True)
        query_dist = np.atleast_2d(query_dist)
        query_ind = np.atleast_2d(query_ind)

        # Second, reduce hubness
        hubness_reduced_query_dist, query_ind = self._hubness_reduction.transform(query_dist,
                                                                                  query_ind,
                                                                                  X=X,  # required by e.g. DSL
                                                                                  assume_sorted=True,)
        # Third, sort hubness reduced candidate neighbors to get the final k neighbors
        if query_is_train:
            n_neighbors -= 1

        kth = np.arange(n_neighbors)
        mask = np.argpartition(hubness_reduced_query_dist, kth=kth)[:, :n_neighbors]
        hubness_reduced_query_dist = np.take_along_axis(hubness_reduced_query_dist, mask, axis=1)
        query_ind = np.take_along_axis(query_ind, mask, axis=1)

        if return_distance:
            result = hubness_reduced_query_dist, query_ind
        else:
            result = query_ind
        return result


class RadiusNeighborsMixin(SklearnRadiusNeighborsMixin):
    """Mixin for radius-based neighbors searches"""

    def radius_neighbors(self, X=None, radius=None, return_distance=True):
        """Finds the neighbors within a given radius of a point or points.

        Return the indices and distances of each point from the dataset
        lying in a ball with size ``radius`` around the points of the query
        array. Points lying on the boundary are included in the results.

        The result points are *not* necessarily sorted by distance to their
        query point.

        Parameters
        ----------
        X : array-like, (n_samples, n_features), optional
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        radius : float
            Limiting distance of neighbors to return.
            (default is the value passed to the constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array, shape (n_samples,) of arrays
            Array representing the distances to each point, only present if
            return_distance=True. The distance values are computed according
            to the ``metric`` constructor parameter.

        ind : array, shape (n_samples,) of arrays
            An array of arrays of indices of the approximate nearest points
            from the population matrix that lie within a ball of size
            ``radius`` around the query points.

        Examples
        --------
        In the following example, we construct a NeighborsClassifier
        class from an array representing our data set and ask who's
        the closest point to [1, 1, 1]:

        >>> import numpy as np
        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from skhubness.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(radius=1.6)
        >>> neigh.fit(samples) # doctest: +ELLIPSIS
        NearestNeighbors(algorithm='auto', leaf_size=30, ...)
        >>> rng = neigh.radius_neighbors([[1., 1., 1.]])
        >>> print(np.asarray(rng[0][0])) # doctest: +ELLIPSIS
        [1.5 0.5]
        >>> print(np.asarray(rng[1][0])) # doctest: +ELLIPSIS
        [1 2]

        The first array returned contains the distances to all points which
        are closer than 1.6, while the second array returned contains their
        indices.  In general, multiple points can be queried at the same time.

        Notes
        -----
        Because the number of neighbors of each point is not necessarily
        equal, the results for multiple query points cannot be fit in a
        standard data array.
        For efficiency, `radius_neighbors` returns arrays of objects, where
        each object is a 1D array of indices or distances.
        """
        check_is_fitted(self, ["_fit_method", "_fit_X"], all_or_any=any)

        if X is not None:
            query_is_train = False
            X = check_array(X, accept_sparse='csr')
        else:
            query_is_train = True
            X = self._fit_X

        if radius is None:
            radius = self.radius

        if self._fit_method == 'brute':
            # for efficiency, use squared euclidean distances
            if self.effective_metric_ == 'euclidean':
                radius *= radius
                kwds = {'squared': True}
            else:
                kwds = self.effective_metric_params_

            reduce_func = partial(self._radius_neighbors_reduce_func,
                                  radius=radius,
                                  return_distance=return_distance)

            results = pairwise_distances_chunked(
                X, self._fit_X, reduce_func=reduce_func,
                metric=self.effective_metric_, n_jobs=self.n_jobs,
                **kwds)
            if return_distance:
                dist_chunks, neigh_ind_chunks = zip(*results)
                dist_list = sum(dist_chunks, [])
                neigh_ind_list = sum(neigh_ind_chunks, [])
                # See https://github.com/numpy/numpy/issues/5456
                # if you want to understand why this is initialized this way.
                dist = np.empty(len(dist_list), dtype='object')
                dist[:] = dist_list
                neigh_ind = np.empty(len(neigh_ind_list), dtype='object')
                neigh_ind[:] = neigh_ind_list
                results = dist, neigh_ind
            else:
                neigh_ind_list = sum(results, [])
                results = np.empty(len(neigh_ind_list), dtype='object')
                results[:] = neigh_ind_list

        elif self._fit_method in ['ball_tree', 'kd_tree']:
            if issparse(X):
                raise ValueError(f"{self._fit_method} does not work with sparse matrices. "
                                 f"Densify the data, or set algorithm='brute'.")

            n_jobs = effective_n_jobs(self.n_jobs)
            delayed_query = delayed(_tree_query_radius_parallel_helper)
            parallel_kwargs = {"prefer": "threads"}
            results = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(self._tree, X[s], radius, return_distance)
                for s in gen_even_slices(X.shape[0], n_jobs)
            )
            if return_distance:
                # Different order of neigh_ind, dist than usual!
                neigh_ind, dist = tuple(zip(*results))
                results = np.hstack(dist), np.hstack(neigh_ind)
            else:
                results = np.hstack(results)

        elif self._fit_method in ['lsh']:
            # assume joblib>=0.12
            delayed_query = delayed(self._index.radius_neighbors)
            parallel_kwargs = {"prefer": "threads"}
            n_jobs = effective_n_jobs(self.n_jobs)
            results = Parallel(n_jobs, **parallel_kwargs)(
                delayed_query(X[s], radius=radius, return_distance=return_distance)
                for s in gen_even_slices(X.shape[0], n_jobs)
            )

        elif self._fit_method in ['hnsw']:
            raise ValueError(f'nmslib/hnsw does not support radius queries.')

        else:
            raise ValueError(f"internal: _fit_method={self._fit_method} not recognized.")

        if self._fit_method in ['lsh', 'hnsw']:
            if return_distance:
                # dist, neigh_ind = tuple(zip(*results))
                # results = np.hstack(dist), np.hstack(neigh_ind)
                dist, neigh_ind = zip(*results)
                # results = [np.atleast_2d(arr) for arr in [np.hstack(dist), np.hstack(neigh_ind)]]
                results = [np.hstack(dist), np.hstack(neigh_ind)]
            else:
                results = np.hstack(results)

        if not query_is_train:
            return results
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                dist, neigh_ind = results
            else:
                neigh_ind = results

            for ind, ind_neighbor in enumerate(neigh_ind):
                mask = ind_neighbor != ind

                neigh_ind[ind] = ind_neighbor[mask]
                if return_distance:
                    dist[ind] = dist[ind][mask]

            if return_distance:
                return dist, neigh_ind
            return neigh_ind

    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity'):
        """Computes the (weighted) graph of Neighbors for points in X

        Neighborhoods are restricted the points at a distance lower than
        radius.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features], optional
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        radius : float
            Radius of neighborhoods.
            (default is the value passed to the constructor).

        mode : {'connectivity', 'distance'}, optional
            Type of returned matrix: 'connectivity' will return the
            connectivity matrix with ones and zeros, in 'distance' the
            edges are Euclidean distance between points.

        Returns
        -------
        A : sparse matrix in CSR format, shape = [n_samples, n_samples]
            A[i, j] is assigned the weight of edge that connects i to j.

        Examples
        --------
        >>> X = [[0], [3], [1]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(radius=1.5)
        >>> neigh.fit(X) # doctest: +ELLIPSIS
        NearestNeighbors(algorithm='auto', leaf_size=30, ...)
        >>> A = neigh.radius_neighbors_graph(X)
        >>> A.toarray()
        array([[1., 0., 1.],
               [0., 1., 0.],
               [1., 0., 1.]])

        See also
        --------
        kneighbors_graph
        """
        check_is_fitted(self, ["_fit_method", "_fit_X"], all_or_any=any)
        if X is not None:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        n_samples2 = self._fit_X.shape[0]
        if radius is None:
            radius = self.radius

        # construct CSR matrix representation of the NN graph
        if mode == 'connectivity':
            A_ind = self.radius_neighbors(X, radius,
                                          return_distance=False)
            A_data = None
        elif mode == 'distance':
            dist, A_ind = self.radius_neighbors(X, radius,
                                                return_distance=True)
            A_data = np.concatenate(list(dist))
        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                'or "distance" but got %s instead' % mode)

        n_samples1 = A_ind.shape[0]
        n_neighbors = np.array([len(a) for a in A_ind])
        A_ind = np.concatenate(list(A_ind))
        if A_data is None:
            A_data = np.ones(len(A_ind))
        A_indptr = np.concatenate((np.zeros(1, dtype=int),
                                   np.cumsum(n_neighbors)))

        return csr_matrix((A_data, A_ind, A_indptr),
                          shape=(n_samples1, n_samples2))

    def _kneighbors_reduce_func(self, dist, start,
                                n_neighbors, return_distance):
        """Reduce a chunk of distances to the nearest neighbors

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : array of shape (n_samples_chunk, n_samples)
        start : int
            The index in X which the first row of dist corresponds to.
        n_neighbors : int
        return_distance : bool

        Returns
        -------
        dist : array of shape (n_samples_chunk, n_neighbors), optional
            Returned only if return_distance
        neigh : array of shape (n_samples_chunk, n_neighbors)

        Notes
        -----
        This is required until radius_candidates is implemented in addition to kcandiates.
        """
        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :n_neighbors]
        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[
            sample_range, np.argsort(dist[sample_range, neigh_ind])]
        if return_distance:
            if self.effective_metric_ == 'euclidean':
                result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind
            else:
                result = dist[sample_range, neigh_ind], neigh_ind
        else:
            result = neigh_ind
        return result
