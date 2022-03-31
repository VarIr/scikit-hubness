# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from collections import namedtuple
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from tqdm.auto import tqdm

from ._base import HubnessReduction
from skhubness.utils.kneighbors_graph import check_kneighbors_graph, check_matching_n_indexed
from skhubness.utils.kneighbors_graph import hubness_reduced_k_neighbors_graph


class LocalScaling(HubnessReduction, TransformerMixin, BaseEstimator):
    """ Hubness reduction with Local Scaling [1]_ in an sklearn-compatible kneighbors_graph.

    Parameters
    ----------
    k: int, default = 5
        Number of neighbors to consider for the rescaling
    method: 'standard' or 'nicdm', default = 'standard'
        Perform local scaling with the specified variant:

        - 'standard' or 'ls' rescale distances using the distance to the k-th neighbor
        - 'nicdm' rescales distances using a statistic over distances to k neighbors
    verbose: int, default = 0
        If verbose > 0, show progress bar.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """

    def __init__(self, k: int = 5, *, method: str = 'standard', verbose: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.method = method
        self.effective_method_ = method
        self.verbose = verbose

    def fit(self, X: csr_matrix, y=None, **kwargs) -> LocalScaling:
        """ Extract local scaling parameters.

        Parameters
        ----------
        X : sparse matrix
            Sorted sparse neighbors graph, such as obtained from sklearn.neighbors.kneighbors_graph.
            Each row i must contain the neighbors of X_i in order of increasing distances,
            that is, nearest neighbor first.
        y : ignored

        Returns
        -------
        self

        Notes
        -----
        Ensure sorting when using custom (approximate) neighbors implementations.
        """
        X = check_kneighbors_graph(X)

        if self.effective_method_ in ["ls", "standard", "nicdm"]:
            self.effective_method_ = "ls" if self.effective_method_ == "standard" else self.effective_method_
        else:
            raise ValueError(f"Unknown local scaling method: {self.effective_method_}. "
                             f"Must be one of: 'ls', 'standard', 'nicdm'.")
        k = self.k
        if (k >= (stored_neigh := X.indptr[1] - X.indptr[0])) or (k < 1):
            raise ValueError(f"Local scaling neighbor parameter k={k} must be in "
                             f"[1, {stored_neigh}), that is, less than n_neighbors "
                             f"in `X`.")
        local_statistic = self._local_statistic(X, k, include_self=True)
        self.r_dist_indexed_ = local_statistic.dist
        self.r_ind_indexed_ = local_statistic.indices
        self.n_indexed_ = X.shape[0]

        return self

    def _local_statistic(
            self,
            X: csr_matrix,
            k: int,
            include_self: bool,
            return_indices: bool = True,
    ):
        # If self distances are included, the nearest "neighbor" must be excluded
        include_self = int(include_self)

        # Find distances to the k nearest neighbors and their indices.
        # For standard LS, the single k-th nearest neighbor suffices.
        if self.effective_method_ == "nicdm":
            start = include_self  # i.e., 0 or 1
        else:  # "ls", "standard"
            start = k - 1 + include_self
        end = k + include_self

        n_samples = X.shape[0]
        dist = X.data.reshape(n_samples, -1)[:, start:end]
        if return_indices:
            indices = X.indices.reshape(n_samples, -1)[:, start:end]
        else:
            indices = None

        return namedtuple("LocalStatistic", ["dist", "indices"])(dist=dist, indices=indices)

    def transform(self, X, y=None, **kwargs) -> csr_matrix:
        """ Transform distance between query and indexed data with Local Scaling.

        Parameters
        ----------
        X : sparse matrix of shape (n_query, n_indexed)
            Sorted sparse neighbors graph, such as obtained from sklearn.neighbors.kneighbors_graph.
            Each row i must contain the neighbors of X_i in order of increasing distances,
            that is, nearest neighbor first.
        y : ignored

        Returns
        -------
        A : sparse matrix of shape (n_query, n_indexed)
            Hubness-reduced graph where A[i, j] is assigned the weight of edge that connects i to j.
            The matrix is of CSR format.
        """
        check_is_fitted(self, ["r_dist_indexed_", "r_ind_indexed_", "n_indexed_"])
        X = check_kneighbors_graph(X)
        check_matching_n_indexed(X, self.n_indexed_)

        n_query, n_indexed = X.shape
        n_neighbors = X.getrow(0).indices.size

        if n_neighbors == 1:
            warnings.warn("Cannot perform hubness reduction with a single neighbor per query. "
                          "Skipping hubness reduction, and returning untransformed neighbor graph.")
            return X

        # increment to account for self neighborhood in pos. 0
        k = self.k + 1

        # Find distances to the k-th neighbor (standard LS) or the k neighbors (NICDM)
        r_dist_query = self._local_statistic(X, k, return_indices=False, include_self=False).dist

        # Calculate LS or NICDM
        hub_reduced_dist = np.empty_like(X.data, shape=(n_query, n_neighbors))

        # Optionally show progress of local scaling transformation loop
        range_n_test = tqdm(
            range(n_query),
            desc=f"{self.effective_method_.upper()} trafo",
            disable=self.verbose < 1,
        )

        # Perform standard local scaling...
        if self.effective_method_ == "ls":
            r_indexed = self.r_dist_indexed_[:, -1]
            r_query = r_dist_query[:, -1]
            # XXX acceleration via numba/pythran/etc desirable
            for i in range_n_test:
                X_i = X.getrow(i)
                dist_i = X_i.data ** 2
                dist_i *= -1
                dist_i /= (r_query[i] * r_indexed[X_i.indices])
                dist_i = 1. - dist_i
                hub_reduced_dist[i, :] = dist_i
        # ...or use non-iterative contextual dissimilarity measure
        elif self.effective_method_ == "nicdm":
            r_indexed = self.r_dist_indexed_.mean(axis=1)
            r_query = r_dist_query.mean(axis=1)
            # XXX acceleration?
            for i in range_n_test:
                X_i = X.getrow(i)
                denominator = r_query[i] * r_indexed[X_i.indices]
                denominator **= 0.5
                hub_reduced_dist[i, :] = X_i.data / denominator
        else:
            raise ValueError(f"Internal: Invalid method {self.effective_method_}. Try 'ls' or 'nicdm'.")

        # Sort neighbors according to hubness-reduced distances, and create sparse kneighbors graph
        kng = hubness_reduced_k_neighbors_graph(hub_reduced_dist, original_X=X, sort_distances=True)
        return kng
