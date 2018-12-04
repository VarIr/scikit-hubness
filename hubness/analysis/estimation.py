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
"""
import logging
from multiprocessing import cpu_count
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.base import issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_random_state

__all__ = ['Hubness']
VALID_METRICS = ['euclidean', 'cosine', 'precomputed']


class Hubness(object):
    """ Hubness characteristics of data set.

    Parameters
    ----------
    k : int
        Neighborhood size
    hub_size : float
        Hubs are defined as objects with k-occurrence > hub_size * k.
    metric : string, one of ['euclidean', 'cosine', 'precomputed']
        Metric to use for distance computation. Currently, only
        Euclidean, cosine, and precomputed distances are supported.
    k_neighbors : bool
        Whether to save the k-neighbor lists. Requires O(n_test * k) memory.
    k_occurrence : bool
        Whether to save the k-occurrence. Requires O(n_test) memory.
    random_state : int, RandomState instance or None, optional
        CURRENTLY IGNORED.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle_equal : bool, optional
        If true, shuffle neighbors with identical distances to avoid
        artifact hubness.
        NOTE: This is especially useful for secondary distance measures
        with a restricted number of possible values, e.g. SNN or MP empiric.
    n_jobs : int, optional
        CURRENTLY IGNORED.
        Number of processes for parallel computations.
        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs
    verbose : int, optional
        Level of output messages

    Attributes
    ----------
    k_skewness_ : float
        Hubness, measured as skewness of k-occurrence histogram [1]_
    k_skewness_truncnom : float
        Hubness, measured as skewness of truncated normal distribution
        fitted with k-occurrence histogram
    atkinson_index_ : float
        Hubness, measured as the Atkinson index of k-occurrence distribution
    gini_index_ : float
        Hubness, measured as the Gini index of k-occurrence distribution
    robinhood_index_ : float
        Hubness, measured as Robin Hood index of k-occurrence distribution [2]_
    antihubs_ : int
        Indices to antihubs
    antihub_occurrence_ : float
        Proportion of antihubs in data set
    hubs_ : int
        Indices to hubs
    hub_occurrence_ : float
        Proportion of k-nearest neighbor slots occupied by hubs
    groupie_ratio_ : float
        Proportion of objects with the largest hub in their neighborhood
    k_occurrence_ : ndarray
        Reverse neighbor count for each object
    k_neighbors_ : ndarray
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

    def __init__(self, k: int = 10, hub_size: float = 2., metric='euclidean',
                 k_neighbors: bool = False,
                 k_occurrence: bool = False,
                 verbose: int = 0, n_jobs: int = 1, random_state=None,
                 shuffle_equal: bool = True, **kwargs):
        self.k = k
        self.hub_size = hub_size
        self.metric = metric
        self.store_k_neighbors = k_neighbors
        self.store_k_occurrence = k_occurrence
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.shuffle_equal = shuffle_equal
        self.kwargs = kwargs

        # Attributes that are set upon fitting
        self.k_neighbors_ = None
        self.antihubs_ = None
        self.hubs_ = None
        self.antihub_occurrence_ = None
        self.atkinson_index_ = None
        self.gini_index_ = None
        self.groupie_ratio_ = None
        self.robinhood_index_ = None
        self.hub_occurrence_ = None
        self.k_occurrence_ = None
        self.k_skewness_ = None
        self.k_skewness_truncnorm_ = None

        # Making sure parameters have sensible values
        if k is not None:
            if k < 1:
                raise ValueError(f"Neighborhood size 'k' must "
                                 f"be >= 1, but is {k}.")
        if hub_size <= 0:
            raise ValueError(f"Hub size must be greater than zero.")
        if metric not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. "
                             f"Must be one of {VALID_METRICS}.")
        if not isinstance(k_neighbors, bool):
            raise ValueError(f"k_neighbors must be True or False.")
        if not isinstance(k_occurrence, bool):
            raise ValueError(f"k_occurrence must be True or False.")
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"Number of parallel processes 'n_jobs' must be "
                             f"a positive integer, or ``-1`` to use all local"
                             f" CPU cores. Was {n_jobs} instead.")
        if verbose < 0:
            raise ValueError(f"Verbosity level 'verbose' must be >= 0, "
                             f"but was {verbose}.")
        return

    def _k_neighbors(self, X: np.ndarray, Y: np.ndarray) -> np.array:
        """ Return indices of nearest neighbors in Y for each vector in X. """
        nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        nn.fit(Y)
        indices = nn.kneighbors(X, return_distance=False)
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
        for i in range(n_test):
            if self.verbose > 1 \
                    or self.verbose and (i % 1000 == 0 or i + 1 == n_test):
                logging.info(f"k neighbors (from distances): "
                             f"{i+1}/{n_test}.", flush=True)
            d = D[i, :].copy()
            d[~np.isfinite(d)] = np.inf
            if self.shuffle_equal:
                # Randomize equal values in the distance matrix rows to avoid
                # the problem case if all numbers to sort are the same,
                # which would yield high hubness, even if there is none.
                rp = np.random.permutation(m_test)
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
            if self.shuffle_equal:
                for i in range(n_test):
                    if self.verbose > 1 or self.verbose and (i % 1000 == 0 or i + 1 == n_test):
                        logging.info(f"k neighbors (from sparse distances): {i+1}/{n_test}.", flush=True)
                    x = X.getrow(i)
                    rp = self.random_state.permutation(x.nnz)
                    d2 = x.data[rp]
                    d2idx = np.argpartition(d2, kth=np.arange(self.k))
                    k_neighbors[i] = x.indices[rp[d2idx[:self.k]]]
            else:
                for i in range(n_test):
                    if self.verbose > 1 or self.verbose and (i % 1000 == 0 or i + 1 == n_test):
                        logging.info(f"k neighbors (from sparse distances): {i+1}/{n_test}.", flush=True)
                    x = X.getrow(i)
                    min_ind = np.argpartition(x.data, kth=np.arange(self.k))[:self.k]
                    k_neighbors[i] = x.indices[min_ind]
            k_neighbors = np.concatenate(k_neighbors)
        return k_neighbors

    @staticmethod
    def skewness_truncnorm(k_occurrence: np.ndarray) -> float:
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
    def gini_index(k_occurrence: np.ndarray, limiting='memory') -> float:
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
        if limiting.lower() in ['memory', 'space']:
            numerator = np.int(0)
            for i in range(n):
                numerator += np.sum(np.abs(k_occurrence[:] - k_occurrence[i]))
        elif limiting.lower() in ['time', 'cpu']:
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
    def robinhood_index(k_occurrence: np.ndarray) -> float:
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
    def atkinson_index(k_occurrence: np.ndarray, eps: float = .5) -> float:
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
    def antihub_occurrence(k_occurrence: np.ndarray) -> (np.array, float):
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
    def hub_occurrence(k: int, k_occurrence: np.ndarray, n_test: int, hub_size: float = 2):
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

    def estimate(self, X: np.ndarray, Y: np.ndarray = None, has_self_distances: bool = False):
        """ Estimate hubness in a data set.

        Hubness is estimated from the distances between all objects in X to all objects in Y.
        If Y is None, all-against-all distances between the objects in X are used.
        If self.metric == 'precomputed', X must be a distance matrix.

        Parameters
        ----------
        X : ndarray, shape (n_query, n_features) or (n_query, n_indexed)
            Array of query vectors, or distance, if self.metric == 'precomputed'
        Y : ndarray, shape (n_indexed, n_features) or None
            Array of indexed vectors. If None, calculate distance between all pairs of
            objects in X.
        has_self_distances : bool, default = False
            Define, whether a precomputed distance matrix contains self distances,
            which need to be excluded.

        Returns
        -------
        self : Hubness
            An instance of class Hubness is returned. Hubness indices are
            provided as attributes (e.g. self.robinhood_index_).
        """
        return self.fit_transform(X, Y, has_self_distances)

    def fit_transform(self, X, Y=None, has_self_distances=False):
        # Let's assume there are no self distances in X
        kth = np.arange(self.k)
        start = 0
        end = self.k
        if self.metric == 'precomputed':
            if Y is not None:
                raise ValueError(
                    f"Y must be None when using precomputed distances.")
            n_test, n_train = X.shape
            if n_test == n_train and has_self_distances:
                kth = np.arange(self.k + 1)
                start = 1
                end = self.k + 1
        else:
            n_test, m_test = X.shape
            if Y is None:
                Y = X
                # Self distances do occur in this case
                kth = np.arange(self.k + 1)
                start = 1
                end = self.k + 1
            n_train, m_train = Y.shape
            assert m_test == m_train, f'Number of features do not match'

        if self.metric == 'precomputed':
            if issparse(X):
                k_neighbors = self._k_neighbors_precomputed_sparse(X)
            else:
                k_neighbors = self._k_neighbors_precomputed(X, kth, start, end)
        else:
            k_neighbors = self._k_neighbors(X, Y)
        if self.store_k_neighbors:
            self.k_neighbors_ = k_neighbors
        k_occurrence = np.bincount(
            k_neighbors.astype(int).ravel(), minlength=n_train)
        if self.store_k_occurrence:
            self.k_occurrence_ = k_occurrence
        # traditional skewness measure
        self.k_skewness_ = stats.skew(k_occurrence)
        # new skewness measure (truncated normal distribution)
        self.k_skewness_truncnorm_ = self.skewness_truncnorm(k_occurrence)
        # Gini index
        if k_occurrence.shape[0] > 10_000:
            limiting = 'space'
        else:
            limiting = 'time'
        self.gini_index_ = self.gini_index(k_occurrence, limiting)
        # Robin Hood index
        self.robinhood_index_ = self.robinhood_index(k_occurrence)
        # Atkinson index
        self.atkinson_index_ = self.atkinson_index(k_occurrence)
        # anti-hub occurrence
        self.antihubs_, self.antihub_occurrence_ = \
            self.antihub_occurrence(k_occurrence)
        # hub occurrence
        self.hubs_, self.hub_occurrence_ = \
            self.hub_occurrence(k=self.k, k_occurrence=k_occurrence,
                                n_test=n_test, hub_size=self.hub_size)
        # Largest hub
        self.groupie_ratio_ = k_occurrence.max() / n_test / self.k

        return self
