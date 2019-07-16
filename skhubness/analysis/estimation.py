#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

"""
This file is part of scikit-hubness.
The package is available at https://github.com/VarIr/scikit-hubness/
and distributed under the terms of the BSD-3 license.

(c) 2018-2019, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI) and
University of Vienna, Division of Computational Systems Biology (CUBE)
Contact: <roman.feldbauer@univie.ac.at>
"""
from __future__ import annotations
from multiprocessing import cpu_count
from tqdm.auto import tqdm
from typing import List
import warnings
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.base import issparse
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_random_state, check_array, check_is_fitted
from skhubness.neighbors import NearestNeighbors

__all__ = ['Hubness', 'VALID_HUBNESS_MEASURES']

VALID_METRICS = ['euclidean',
                 'cosine',
                 'precomputed',
                 ]
VALID_HUBNESS_MEASURES = ['all',
                          'k_skewness',
                          'k_skewness_truncnorm',
                          'atkinson',
                          'gini',
                          'robinhood',
                          'antihubs',
                          'antihub_occurrence',
                          'hubs',
                          'hub_occurrence',
                          'groupie_ratio',
                          'k_neighbors',
                          'k_occurrence',
                          ]


class Hubness(BaseEstimator):
    """ Hubness characteristics of data set.

    Parameters
    ----------
    k : int
        Neighborhood size

    return_value: str, default = "k_skewness"
        Hubness measure to return by :meth:`score`
        By default, this is the skewness of the k-occurrence histogram.
        Use "all" to return a dict of all available measures,
        or check `skhubness.analysis.VALID_HUBNESS_MEASURE`
        for available measures.

    hub_size : float
        Hubs are defined as objects with k-occurrence > hub_size * k.

    metric : string, one of ['euclidean', 'cosine', 'precomputed']
        Metric to use for distance computation. Currently, only
        Euclidean, cosine, and precomputed distances are supported.

    store_k_neighbors : bool
        Whether to save the k-neighbor lists. Requires O(n_test * k) memory.

    store_k_occurrence : bool
        Whether to save the k-occurrence. Requires O(n_test) memory.

    algorithm : {'auto', 'hnsw', 'lsh', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'hnsw' will use :class:`HNSW`
        - 'lsh' will use :class:`LSH`
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    algorithm_params : dict, optional
        Override default parameters of the NN algorithm.
        For example, with algorithm='lsh' and algorithm_params={n_candidates: 100}
        one hundred approximate neighbors are retrieved with LSH.
        If parameter hubness is set, the candidate neighbors are further reordered
        with hubness reduction.
        Finally, n_neighbors objects are used from the (optionally reordered) candidates.

    hubness : {'mutual_proximity', 'local_scaling', 'dis_sim_local', None}, optional
        Hubness reduction algorithm
        # TODO add all supported hubness reduction methods

        - 'mutual_proximity' or 'mp' will use :class:`MutualProximity`
        - 'local_scaling' or 'ls' will use :class:`LocalScaling`
        - 'dis_sim_local' or 'dsl' will use :class:`DisSimLocal`

        If None, no hubness reduction will be performed (=vanilla kNN).

    hubness_params: dict, optional
        Override default parameters of the selected hubness reduction algorithm.
        For example, with hubness='mp' and hubness_params={'method': 'normal'}
        a mutual proximity variant is used, which models distance distributions
        with independent Gaussians.

    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle_equal : bool, optional
        If true and metric='precomputed', shuffle neighbors with identical distances
        to avoid artifact hubness.
        NOTE: This is especially useful for secondary distance measures
        with a finite number of possible values, e.g. SNN or MP empiric.

    n_jobs : int, optional
        CURRENTLY IGNORED.
        Number of processes for parallel computations.
        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs

    verbose : int, optional
        Level of output messages

    Attributes
    ----------
    k_skewness : float
        Hubness, measured as skewness of k-occurrence histogram [1]_

    k_skewness_truncnorm : float
        Hubness, measured as skewness of truncated normal distribution
        fitted with k-occurrence histogram

    atkinson_index : float
        Hubness, measured as the Atkinson index of k-occurrence distribution

    gini_index : float
        Hubness, measured as the Gini index of k-occurrence distribution

    robinhood_index : float
        Hubness, measured as Robin Hood index of k-occurrence distribution [2]_

    antihubs : int
        Indices to antihubs

    antihub_occurrence : float
        Proportion of antihubs in data set

    hubs : int
        Indices to hubs

    hub_occurrence : float
        Proportion of k-nearest neighbor slots occupied by hubs

    groupie_ratio : float
        Proportion of objects with the largest hub in their neighborhood

    k_occurrence : ndarray
        Reverse neighbor count for each object

    k_neighbors : ndarray
        Indices to k-nearest neighbors for each object

    References
    ----------
    .. [1] `Radovanović, M.; Nanopoulos, A. & Ivanovic, M.
            Hubs in space: Popular nearest neighbors in high-dimensional data.
            Journal of Machine Learning Research, 2010, 11, 2487-2531`
    .. [2] `Feldbauer, R.; Leodolter, M.; Plant, C. & Flexer, A.
            Fast approximate hubness reduction for large high-dimensional data.
            IEEE International Conference of Big Knowledge (2018).`
    """

    def __init__(self, k: int = 10, return_value: str = 'k_skewness',
                 hub_size: float = 2., metric='euclidean',
                 store_k_neighbors: bool = False, store_k_occurrence: bool = False,
                 algorithm: str = 'auto', algorithm_params: dict = None,
                 hubness: str = None, hubness_params: dict = None,
                 verbose: int = 0, n_jobs: int = 1, random_state=None,
                 shuffle_equal: bool = True):
        self.k = k
        self.return_value = return_value
        self.hub_size = hub_size
        self.metric = metric
        self.store_k_neighbors = store_k_neighbors
        self.store_k_occurrence = store_k_occurrence
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params
        self.hubness = hubness
        self.hubness_params = hubness_params
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.shuffle_equal = shuffle_equal

    def fit(self, X, y=None) -> Hubness:
        """ Fit indexed objects.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features) or (n_query, n_indexed) if metric=='precomputed'
            Training data vectors or distance matrix, if metric == 'precomputed'.

        y: ignored

        Returns
        -------
        self:
            Fitted instance of :mod:Hubness
        """
        X = check_array(X, accept_sparse=True)

        # Making sure parameters have sensible values
        k = self.k
        if k is None:
            k = 10
        else:
            if k < 1:
                raise ValueError(f"Neighborhood size 'k' must "
                                 f"be >= 1, but is {k}.")
        self.k = k

        store_k_neighbors = self.store_k_neighbors
        if store_k_neighbors is None:
            store_k_neighbors = False
        elif not isinstance(store_k_neighbors, bool):
            raise ValueError(f"k_neighbors must be True or False.")
        self.store_k_neighbors = store_k_neighbors

        store_k_occurrence = self.store_k_occurrence
        if store_k_occurrence is None:
            store_k_occurrence = False
        elif not isinstance(store_k_occurrence, bool):
            raise ValueError(f"k_occurrence must be True or False.")
        self.store_k_occurrence = store_k_occurrence

        return_value = self.return_value
        if return_value is None:
            return_value = 'k_skewness'
        elif return_value not in VALID_HUBNESS_MEASURES:
            raise ValueError(f'Unknown return value: {return_value}. '
                             f'Allowed hubness measures: {VALID_HUBNESS_MEASURES}.')
        elif return_value == 'k_neighbors' and not self.store_k_neighbors:
            warnings.warn(f'Incompatible parameters return_value={return_value} '
                          f'and store_k_neighbors={self.store_k_neighbors}. '
                          f'Overriding store_k_neighbor=True.')
            self.store_k_neighbors = True
        elif return_value == 'k_occurrence' and not self.store_k_occurrence:
            warnings.warn(f'Incompatible parameters return_value={return_value} '
                          f'and store_k_occurrence={self.store_k_occurrence}. '
                          f'Overriding store_k_occurrence=True.')
            self.store_k_occurrence = True
        self.return_value = return_value

        hub_size = self.hub_size
        if hub_size is None:
            hub_size = 2.
        elif hub_size <= 0:
            raise ValueError(f"Hub size must be greater than zero.")
        self.hub_size = hub_size

        metric = self.metric
        if metric is None:
            metric = 'euclidean'
        if metric not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. "
                             f"Must be one of {VALID_METRICS}.")
        self.metric = metric

        n_jobs = self.n_jobs
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"Number of parallel processes 'n_jobs' must be "
                             f"a positive integer, or ``-1`` to use all local"
                             f" CPU cores. Was {n_jobs} instead.")
        self.n_jobs = n_jobs

        verbose = self.verbose
        if verbose is None:
            verbose = 0
        elif verbose < 0:
            verbose = 0
        self.verbose = verbose

        # check random state
        self._random_state = check_random_state(self.random_state)

        shuffle_equal = self.shuffle_equal
        if shuffle_equal is None:
            shuffle_equal = False
        elif not isinstance(shuffle_equal, bool):
            raise ValueError(f'Parameter shuffle_equal must be True or False, '
                             f'but was {shuffle_equal}.')
        self.shuffle_equal = shuffle_equal

        # Fit Hubness to training data: just store as indexed objects
        self.X_train_ = X

        return self

    def _k_neighbors(self, X_test: np.ndarray = None, X_train: np.ndarray = None) -> np.array:
        """ Return indices of nearest neighbors in X_train for each vector in X_test. """
        nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric,
                              algorithm=self.algorithm, algorithm_params=self.algorithm_params,
                              hubness=self.hubness, hubness_params=self.hubness_params)
        nn.fit(X_train)
        # if X_test is None, self distances are ignored
        indices = nn.kneighbors(X_test, return_distance=False)
        return indices

    def _k_neighbors_precomputed(self, D: np.ndarray, kth: np.ndarray, start: int, end: int) -> np.ndarray:
        """ Return indices of nearest neighbors in precomputed distance matrix.

        Note
        ----
        Parameters kth, start, end are used to ensure that objects are
        not returned as their own nearest neighbors.
        """
        n_test, m_test = D.shape
        indices = np.zeros((n_test, self.k), dtype=np.int32)
        if self.verbose:
            range_n_test = tqdm(range(n_test))
        else:
            range_n_test = range(n_test)
        for i in range_n_test:
            d = D[i, :].copy()
            d[~np.isfinite(d)] = np.inf
            if self.shuffle_equal:
                # Randomize equal values in the distance matrix rows to avoid
                # the problem case if all numbers to sort are the same,
                # which would yield high hubness, even if there is none.
                rp = self._random_state.permutation(m_test)
                d2 = d[rp]
                d2idx = np.argpartition(d2, kth=kth)
                indices[i, :] = rp[d2idx[start:end]]
            else:
                d_idx = np.argpartition(d, kth=kth)
                indices[i, :] = d_idx[start:end]
        return indices

    def _k_neighbors_precomputed_sparse(self, X: csr_matrix, n_samples: int = None) -> np.ndarray:
        """ Find nearest neighbors in sparse distance matrix.

        Parameters
        ----------
        X : sparse, shape = [n_test, n_indexed]
            Sparse distance matrix. Only non-zero elements
            may be considered neighbors.

        n_samples : int
            Number of sampled indexed objects, e.g.
            in approximate hubness reduction.
            If None, this is inferred from the first row of X.

        Returns
        -------
        k_neighbors : ndarray
            Flattened array of neighbor indices.
        """
        if not issparse(X):
            raise TypeError(f'Matrix X is not sparse')
        X = X.tocsr()
        if n_samples is None:
            n_samples = X.indptr[1] - X.indptr[0]
        n_test, _ = X.shape
        # To allow different number of explicit entries per row,
        # we need to process the matrix row-by-row.
        if np.all(X.indptr[1:] - X.indptr[:-1] == n_samples) and not self.shuffle_equal:
            min_ind = np.argpartition(X.data.reshape(n_test, n_samples),
                                      kth=np.arange(self.k),
                                      axis=1)[:, :self.k]
            k_neighbors = X.indices[min_ind.ravel() + np.repeat(X.indptr[:-1], repeats=self.k)]
        else:
            k_neighbors = np.empty((n_test,), dtype=object)
            if self.verbose:
                range_n_test = tqdm(range(n_test))
            else:
                range_n_test = range(n_test)
            if self.shuffle_equal:
                for i in range_n_test:
                    x = X.getrow(i)
                    rp = self._random_state.permutation(x.nnz)
                    d2 = x.data[rp]
                    d2idx = np.argpartition(d2, kth=np.arange(self.k))
                    k_neighbors[i] = x.indices[rp[d2idx[:self.k]]]
            else:
                for i in range_n_test:
                    x = X.getrow(i)
                    min_ind = np.argpartition(x.data, kth=np.arange(self.k))[:self.k]
                    k_neighbors[i] = x.indices[min_ind]
            k_neighbors = np.concatenate(k_neighbors)
        return k_neighbors

    @staticmethod
    def _calc_skewness_truncnorm(k_occurrence: np.ndarray) -> float:
        """ Hubness measure; corrected for non-negativity of k-occurrence.

        Hubness as skewness of truncated normal distribution
        estimated from k-occurrence histogram.

        Parameters
        ----------
        k_occurrence : ndarray
            Reverse nearest neighbor count for each object.
        """
        clip_left = 0
        clip_right = np.iinfo(np.int64).max
        k_occurrence_mean = k_occurrence.mean()
        k_occurrence_std = k_occurrence.std(ddof=1)
        a = (clip_left - k_occurrence_mean) / k_occurrence_std
        b = (clip_right - k_occurrence_mean) / k_occurrence_std
        skew_truncnorm = stats.truncnorm(a, b).moment(3)
        return skew_truncnorm

    @staticmethod
    def _calc_gini_index(k_occurrence: np.ndarray, limiting='memory') -> float:
        """ Hubness measure; Gini index

        Parameters
        ----------
        k_occurrence : ndarray
            Reverse nearest neighbor count for each object.
        limiting : 'memory' or 'cpu'
            If 'cpu', use fast implementation with high memory usage,
            if 'memory', use slighly slower, but memory-efficient implementation,
            otherwise use naive implementation (slow, low memory usage)
        """
        n = k_occurrence.size
        if limiting in ['memory', 'space']:
            numerator = np.int(0)
            for i in range(n):
                numerator += np.sum(np.abs(k_occurrence[:] - k_occurrence[i]))
        elif limiting in ['time', 'cpu']:
            numerator = np.sum(np.abs(k_occurrence.reshape(1, -1) - k_occurrence.reshape(-1, 1)))
        else:  # slow naive implementation
            n = k_occurrence.size
            numerator = 0
            for i in range(n):
                for j in range(n):
                    numerator += np.abs(k_occurrence[i] - k_occurrence[j])
        denominator = 2 * n * np.sum(k_occurrence)
        return numerator / denominator

    @staticmethod
    def _calc_robinhood_index(k_occurrence: np.ndarray) -> float:
        """ Hubness measure; Robin hood/Hoover/Schutz index.

        Parameters
        ----------
        k_occurrence : ndarray
            Reverse nearest neighbor count for each object.

        Notes
        -----
        The Robin Hood index was proposed in [1]_ and is especially suited
        for hubness estimation in large data sets. Additionally, it offers
        straight-forward interpretability by answering the question:
        What share of k-occurrence must be redistributed, so that all objects
        are equally often nearest neighbors to others?

        References
        ----------
        .. [1] `Feldbauer, R.; Leodolter, M.; Plant, C. & Flexer, A.
                Fast approximate hubness reduction for large high-dimensional data.
                IEEE International Conference of Big Knowledge (2018).`
        """
        numerator = .5 * float(np.sum(np.abs(k_occurrence - k_occurrence.mean())))
        denominator = float(np.sum(k_occurrence))
        return numerator / denominator

    @staticmethod
    def _calc_atkinson_index(k_occurrence: np.ndarray, eps: float = .5) -> float:
        """ Hubness measure; Atkinson index.

        Parameters
        ----------
        k_occurrence : ndarray
            Reverse nearest neighbor count for each object.
        eps : float, default = 0.5
            'Income' weight. Turns the index into a normative measure.
        """
        if eps == 1:
            term = np.prod(k_occurrence) ** (1. / k_occurrence.size)
        else:
            term = np.mean(k_occurrence ** (1 - eps)) ** (1 / (1 - eps))
        return 1. - 1. / k_occurrence.mean() * term

    @staticmethod
    def _calc_antihub_occurrence(k_occurrence: np.ndarray) -> (np.array, float):
        """Proportion of antihubs in data set.

        Antihubs are objects that are never among the nearest neighbors
        of other objects.

        Parameters
        ----------
        k_occurrence : ndarray
            Reverse nearest neighbor count for each object.
        """
        antihubs = np.argwhere(k_occurrence == 0).ravel()
        antihub_occurrence = antihubs.size / k_occurrence.size
        return antihubs, antihub_occurrence

    @staticmethod
    def _calc_hub_occurrence(k: int, k_occurrence: np.ndarray, n_test: int, hub_size: float = 2):
        """Proportion of nearest neighbor slots occupied by hubs.

        Parameters
        ----------
        k : int
            Specifies the number of nearest neighbors
        k_occurrence : ndarray
            Reverse nearest neighbor count for each object.
        n_test : int
            Number of queries (or objects in a test set)
        hub_size : float
            Factor to determine hubs
        """
        hubs = np.argwhere(k_occurrence >= hub_size * k).ravel()
        hub_occurrence = k_occurrence[hubs].sum() / k / n_test
        return hubs, hub_occurrence

    def score(self, X: np.ndarray = None, y=None, has_self_distances: bool = False) -> List[float, dict]:
        """ Estimate hubness in a data set.

        Hubness is estimated from the distances between all objects in X to all objects in Y.
        If Y is None, all-against-all distances between the objects in X are used.
        If self.metric == 'precomputed', X must be a distance matrix.

        Parameters
        ----------
        X : ndarray, shape (n_query, n_features) or (n_query, n_indexed)
            Array of query vectors, or distance, if self.metric == 'precomputed'

        y : ignored

        has_self_distances : bool, default = False
            Define, whether a precomputed distance matrix contains self distances,
            which need to be excluded.

        Returns
        -------
        hubness_measure : float or dict
            Return the hubness measure as indicated by `return_value`.
            Additional hubness indices are provided as attributes
            (e.g. :func:`robinhood_index_`).
            if return_value is 'all', a dict of all hubness measures is returned.
        """
        check_is_fitted(self, 'X_train_')
        if X is None:
            X_test = self.X_train_
        else:
            X_test = X
        X_test = check_array(X_test, accept_sparse=True)
        X_train = self.X_train_

        kth = np.arange(self.k)
        start = 0
        end = self.k
        if self.metric == 'precomputed':
            if X is not None:
                raise ValueError(f'No X must be passed with metric=="precomputed".')
            n_test, n_train = X_test.shape
            if has_self_distances:
                kth = np.arange(self.k + 1)
                start = 1
                end = self.k + 1
        else:
            if X is None:
                # Self distances do occur in this case
                kth = np.arange(self.k + 1)
                start = 1
                end = self.k + 1
            n_test, m_test = X_test.shape
            n_train, m_train = X_train.shape
            if m_test != m_train:
                raise ValueError(f'Number of features do not match: X_train.shape={self.X_train_.shape}, '
                                 f'X_test.shape={X.shape}.')

        if self.metric == 'precomputed':
            if issparse(X_test):
                k_neighbors = self._k_neighbors_precomputed_sparse(X_test)
            else:
                k_neighbors = self._k_neighbors_precomputed(X_test, kth, start, end)
        else:
            if X is None:
                k_neighbors = self._k_neighbors(X_train=X_test)
            else:
                k_neighbors = self._k_neighbors(X_test=X_test, X_train=X_train)
        if self.store_k_neighbors:
            self.k_neighbors = k_neighbors

        k_occurrence = np.bincount(
            k_neighbors.astype(int).ravel(), minlength=n_train)
        if self.store_k_occurrence:
            self.k_occurrence = k_occurrence

        # traditional skewness measure
        self.k_skewness = stats.skew(k_occurrence)

        # new skewness measure (truncated normal distribution)
        self.k_skewness_truncnorm = self._calc_skewness_truncnorm(k_occurrence)

        # Gini index
        limiting = 'space' if k_occurrence.shape[0] > 10_000 else 'time'
        self.gini_index = self._calc_gini_index(k_occurrence, limiting)

        # Robin Hood index
        self.robinhood_index = self._calc_robinhood_index(k_occurrence)

        # Atkinson index
        self.atkinson_index = self._calc_atkinson_index(k_occurrence)

        # anti-hub occurrence
        self.antihubs, self.antihub_occurrence = \
            self._calc_antihub_occurrence(k_occurrence)

        # hub occurrence
        self.hubs, self.hub_occurrence = \
            self._calc_hub_occurrence(k=self.k, k_occurrence=k_occurrence,
                                      n_test=n_test, hub_size=self.hub_size)

        # Largest hub
        self.groupie_ratio = k_occurrence.max() / n_test / self.k

        # Dictionary of all hubness measures
        self.hubness_measures = {'k_skewness': self.k_skewness,
                                 'k_skewness_truncnorm': self.k_skewness_truncnorm,
                                 'atkinson': self.atkinson_index,
                                 'gini': self.gini_index,
                                 'robinhood': self.robinhood_index,
                                 'antihubs': self.antihubs,
                                 'antihub_occurrence': self.antihub_occurrence,
                                 'hubs': self.hubs,
                                 'hub_occurrence': self.hub_occurrence,
                                 'groupie_ratio': self.groupie_ratio,
                                 }
        if hasattr(self, 'k_neighbors'):
            self.hubness_measures['k_neighbors'] = self.k_neighbors
        if hasattr(self, 'k_occurrence'):
            self.hubness_measures['k_occurrence'] = self.k_occurrence

        if self.return_value == 'all':
            return self.hubness_measures
        else:
            return self.hubness_measures[self.return_value]
