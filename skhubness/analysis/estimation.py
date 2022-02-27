#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

"""
This file is part of scikit-hubness.
The package is available at https://github.com/VarIr/scikit-hubness/
and distributed under the terms of the BSD-3 license.

(c) 2018-2022, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI) and
University of Vienna, Division of Computational Systems Biology (CUBE)
Contact: <sci@feldbauer.org>
"""
from __future__ import annotations
from typing import Union
import warnings

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.utils.validation import check_random_state, check_array, check_is_fitted
from tqdm.auto import tqdm

from ..utils.io import validate_verbose
from ..utils.kneighbors_graph import check_kneighbors_graph
from ..utils.multiprocessing import validate_n_jobs

__all__ = [
    "Hubness",
    "VALID_HUBNESS_MEASURES",
]

VALID_METRICS = [
    "euclidean",
    "cosine",
    "precomputed",
]

#: Available hubness measures
VALID_HUBNESS_MEASURES = [
    "all",
    "all_but_gini",
    "antihub_occurrence",
    "atkinson",
    "gini",
    "groupie_ratio",
    "hub_occurrence",
    "k_skewness",
    "k_skewness_truncnorm",
    "robinhood",
]


class Hubness(BaseEstimator):
    """ Examine hubness characteristics of data.

    Parameters
    ----------
    k : int
        Neighborhood size
    return_value : str, default = "k_skewness"
        Hubness measure to return by :meth:`score`
        By default, return the skewness of the k-occurrence histogram.
        Use "all_but_gini" to return all measures except the Gini index,
        which is slow on large datasets.
        Use "all" to return a dict of all available measures,
        or check `skhubness.analysis.VALID_HUBNESS_MEASURE`
        for available measures.
    hub_size : float
        Hubs are defined as objects with k-occurrence > hub_size * k.
    metric : str
        If "precomputed", sparse k-neighbors graphs must be provided in fit and score functions.
        Otherwise, a k-neighbors graph will be computed for the provided vector data.
        In this case, any scikit-learn metric is allowed, e.g. "euclidean" or "cosine".
    p, metric_params : float, dict
        Directly passed to scikit-learn for distance calculations (if metric != "precomputed")
    return_hubs : bool
        Whether to return the list of indices to hub objects
    return_antihubs : bool
        Whether to return the list of indices to antihub objects
    return_k_occurrence: bool
        Whether to save the list of k-occurrences. Requires O(n_test) memory.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle_equal : bool, optional
        If True, shuffle neighbors with identical distances to avoid artifact hubness.
        NOTE: This is especially useful for secondary distance measures
        with a finite number of possible values, e.g. Shared Nearest neighbors
        or Mutual Proximity (empiric).
    n_jobs : int, optional
        Number of processes for parallel computations.

        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs

        Note that not all steps are currently parallelized.
    verbose: int, optional
        Level of output messages

    References
    ----------
    .. [1] `Radovanović, M.; Nanopoulos, A. & Ivanovic, M.
            Hubs in space: Popular nearest neighbors in high-dimensional data.
            Journal of Machine Learning Research, 2010, 11, 2487-2531`
    .. [2] `Feldbauer, R.; Leodolter, M.; Plant, C. & Flexer, A.
            Fast approximate hubness reduction for large high-dimensional data.
            IEEE International Conference of Big Knowledge (2018).`
    """

    def __init__(
            self,
            k: int = 10,
            hub_size: float = 2,
            metric: str = "minkowski",  # "precomputed",
            p: float = 2,
            metric_params: dict = None,
            shuffle_equal: bool = True,
            return_value: str = "k_skewness",
            return_hubs: bool = False,
            return_antihubs: bool = False,
            return_k_occurrence: bool = False,
            verbose: int = 0,
            n_jobs: int = 1,
            random_state=None,

    ):
        self.k = k
        self.return_value = return_value
        self.hub_size = hub_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.shuffle_equal = shuffle_equal
        self.return_hubs = return_hubs
        self.return_antihubs = return_antihubs
        self.return_k_occurrence = return_k_occurrence
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None) -> Hubness:
        """ Fit indexed objects.

        Parameters
        ----------
        X : csr_matrix, shape (n_indexed, n_indexed)
            Distance matrix (k-neighbors graph) of fit/indexed/training objects.
            Must contain at least `k` neighbors per object.

        y : ignored

        Returns
        -------
        self:
            Fitted instance of :mod:Hubness
        """
        if issparse(X):
            X: csr_matrix = check_kneighbors_graph(X)
        else:
            X: np.ndarray = check_array(X, accept_sparse=False)  # noqa
            # In case of vector data, store the index data, and compute
            # the k-neighbors graph for them.
            self.X_indexed_ = X
            # Reduce k for (test) cases with very few objects
            if X.shape[0] <= self.k:
                self.k = X.shape[0] - 1
                if self.k < 1:
                    raise ValueError("Cannot compute hubness as there is only one sample.")
                warnings.warn(f"Parameter k was automatically reduced to {self.k}, "
                              f"because there are only {X.shape[0]} samples.")

        if self.k < 1:
            raise ValueError(f"Neighborhood size 'k' must be >= 1, but is {self.k}")

        # Whether n_"features"_in makes any sense here, idk, but check_estimator requires it
        self.n_samples_in_, self.n_features_in_ = X.shape
        if not issparse(X):
            X = kneighbors_graph(
                X=X,
                n_neighbors=self.k,
                mode="distance",
                metric=self.metric,
                include_self=True,
                n_jobs=self.n_jobs,
            )

        n_neighbors = X.indptr[1]
        if self.k > n_neighbors:
            raise ValueError(f"X does not contain enough neighbors per object "
                             f"(has {n_neighbors} < {self.k}).")

        return_value = self.return_value
        if return_value is None:
            return_value = "k_skewness"
        elif return_value not in VALID_HUBNESS_MEASURES:
            raise ValueError(f"Unknown return value: {return_value}. "
                             f"Allowed hubness measures: {VALID_HUBNESS_MEASURES}.")
        self.return_value = return_value

        hub_size = self.hub_size
        if hub_size is None:
            hub_size = 2.
        elif hub_size <= 0:
            raise ValueError("Hub size must be greater than zero.")
        self.hub_size = hub_size

        metric = self.metric
        if metric is None:
            metric = "euclidean"
        self.metric = metric
        self.n_jobs = validate_n_jobs(self.n_jobs)
        self.verbose = validate_verbose(self.verbose)

        # check random state
        self._random_state = check_random_state(self.random_state)

        if self.shuffle_equal is None:
            self.shuffle_equal = False

        # Fit Hubness to training data: store as indexed objects
        self.kng_indexed_: csr_matrix = X

        return self

    def _k_neighbors(self, X_test: csr_matrix = None) -> np.ndarray:
        """ Return indices of nearest neighbors.

        Parameters
        ----------
        X_test : csr_matrix, optional
            If csr_matrix, assume a k-neighbors graph of query vs. indexed objects.
            If None, find nearest neighbors within the indexed objects.
        """
        k = self.k
        if X_test is None:
            # Filter away the self hits
            X_test = self.kng_indexed_
            start = 1
            end = k + 1
        else:
            start = 0
            end = k
        n_samples = X_test.indptr[1]
        neigh_ind = X_test.indices.reshape(-1, n_samples)[:, start:end]

        if self.shuffle_equal:
            neigh_dist = X_test.data.reshape(-1, n_samples)[:, start:end]
            # Randomize equal values in the distance matrix rows to avoid
            # the problem case if all or many numbers to sort are the same,
            # which would yield high hubness, even if there is none.
            permutation = self._random_state.permutation(k).reshape(-1, k)
            dist_permutated = np.take_along_axis(neigh_dist, permutation, axis=1)
            ind_permutated = np.take_along_axis(neigh_ind, permutation, axis=1)
            sort = np.argsort(dist_permutated, axis=1)
            neigh_ind = np.take_along_axis(ind_permutated, sort, axis=1)

        return neigh_ind

    @staticmethod
    def _calc_skewness_truncnorm(k_occurrence: np.ndarray) -> float:
        """ Hubness measure; corrected for non-negativity of k-occurrence.

        Hubness as skewness of truncated normal distribution estimated from k-occurrence histogram.

        Parameters
        ----------
        k_occurrence : np.ndarray
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
    def _calc_gini_index(k_occurrence: np.ndarray, limiting="memory", verbose: int = 0) -> float:
        """ Hubness measure; Gini index

        Parameters
        ----------
        k_occurrence : np.ndarray
            Reverse nearest neighbor count for each object.
        limiting : "memory" or "cpu"
            If "cpu", use fast implementation with high memory usage,
            if "memory", use slightly slower, but memory-efficient implementation,
            otherwise use naive implementation (slow, low memory usage)
        """
        n = k_occurrence.size
        if limiting in ["memory", "space"]:
            numerator = np.int64(0)
            for i in tqdm(range(n),
                          disable=False if verbose else True,
                          desc="Gini"):
                numerator += np.sum(np.abs(k_occurrence[:] - k_occurrence[i]))
        elif limiting in ["time", "cpu"]:
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
        k_occurrence : np.ndarray
            Reverse nearest neighbor count for each object.

        Notes
        -----
        The Robin Hood index was proposed in [1]_ for hubness estimation and
        is especially suited for large data sets. Additionally, it offers
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
        k_occurrence : np.ndarray
            Reverse nearest neighbor count for each object.
        eps: float, default = 0.5
            "Income" weight. Turns the index into a normative measure.
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
        k_occurrence : np.ndarray
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
            Number of nearest neighbors
        k_occurrence : np.ndarray
            Reverse nearest neighbor count for each object.
        n_test : int
            Number of queries (or objects in a test set)
        hub_size : float
            Factor to determine hubs
        """
        hubs = np.argwhere(k_occurrence >= hub_size * k).ravel()
        hub_occurrence = k_occurrence[hubs].sum() / k / n_test
        return hubs, hub_occurrence

    def score(self, X: csr_matrix = None, y=None) -> Union[float, dict]:
        """ Estimate hubness in a k-neighbors graph.

        Hubness is estimated from the k-neighbors graph provided as `X`,
        in which each row corresponds to one query object, and contains
        the sorted nearest neighbor stored in the index.
        If `X` is None, compute hubness within the indexed objects.

        Parameters
        ----------
        X : csr_matrix, shape (n_query, n_indexed)
            K-neighbors graph of query vs. indexed objects.
            If None, find nearest neighbors within the indexed objects.
        y : ignored

        Returns
        -------
        hubness_measure: float or dict
            Return the hubness measure as indicated by `return_value`.
            Additional hubness indices are provided as attributes
            (e.g. :func:`robinhood_index_`).
            if return_value is "all", a dict of all hubness measures is returned.
        """
        check_is_fitted(self, "kng_indexed_")
        if X is None:
            X = self.kng_indexed_
        else:
            X: csr_matrix = check_array(X, accept_sparse="csr")  # noqa
            if not issparse(X):
                nn = NearestNeighbors(
                    n_neighbors=self.k,
                    metric=self.metric,
                    p=self.p,
                    metric_params=self.metric_params,
                ).fit(self.X_indexed_)
                X = nn.kneighbors_graph(X=X, mode="distance")

        X = check_kneighbors_graph(X, check_sparse=False)
        n_samples_indexed, n_samples_query = X.shape

        k_neighbors = self._k_neighbors(X)
        k_occurrence = np.bincount(
            k_neighbors.astype(int).ravel(),
            minlength=n_samples_query,
        )

        hubness_measures = {}
        calc_all = self.return_value.startswith("all")
        if calc_all or self.return_value == "k_skewness":
            hubness_measures["k_skewness"] = stats.skew(k_occurrence)

        if calc_all or self.return_value == "k_skewness_trunc":
            hubness_measures["k_skewness_truncnorm"] = self._calc_skewness_truncnorm(k_occurrence)

        # don't calc gini in case of "all_but_gini"
        if self.return_value in ["gini", "all"]:
            limiting = "space" if k_occurrence.shape[0] > 10_000 else "time"
            hubness_measures["gini"] = self._calc_gini_index(
                k_occurrence,
                limiting,
                verbose=self.verbose,
            )

        if calc_all or self.return_value == "robinhood":
            hubness_measures["robinhood"] = self._calc_robinhood_index(k_occurrence)

        if calc_all or self.return_value == "atkinson":
            hubness_measures["atkinson"] = self._calc_atkinson_index(k_occurrence)

        if self.return_k_occurrence:
            hubness_measures["k_occurrence"] = k_occurrence

        return_antihub_occurrence = calc_all or self.return_value == "antihub_occurrence"
        if return_antihub_occurrence or self.return_antihubs:
            antihubs, antihub_occurrence = \
                self._calc_antihub_occurrence(k_occurrence)
            if self.return_antihubs:
                hubness_measures["antihubs"] = antihubs
            if return_antihub_occurrence:
                hubness_measures["antihub_occurrence"] = antihub_occurrence

        return_hub_occurrence = calc_all or self.return_value == "hub_occurrence"
        if return_hub_occurrence or self.return_hubs:
            hubs, hub_occurrence = self._calc_hub_occurrence(
                k=self.k,
                k_occurrence=k_occurrence,
                n_test=n_samples_query,
                hub_size=self.hub_size,
            )
            if self.return_hubs:
                hubness_measures["hubs"] = hubs
            if return_hub_occurrence:
                hubness_measures["hub_occurrence"] = hub_occurrence

        if calc_all or self.return_value == "groupie_ratio":
            hubness_measures["groupie_ratio"] = k_occurrence.max() / n_samples_indexed / self.k

        # If there is only one measure, return the value only
        if len(hubness_measures) == 1:
            hubness_measures = hubness_measures.get(self.return_value, None)
            if hubness_measures is None:
                raise ValueError(f"Internal error: could not retrieve {self.return_value}")
        # Otherwise, return a dict of all values
        return hubness_measures
