# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from collections import namedtuple
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from tqdm.auto import tqdm

from .base import HubnessReduction, GraphHubnessReduction
from ..utils.helper import k_neighbors_graph


class GraphLocalScaling(GraphHubnessReduction, TransformerMixin):
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
        self.effective_method_ = "nicdm" if method.lower() == "nicdm" else "ls"
        self.verbose = verbose

    def fit(self, X: csr_matrix, y=None) -> GraphLocalScaling:
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
        # check_kneighbors_graph(kng)  # TODO

        k = self.k
        X = X.tocsr()
        local_statistic = self._local_statistic(X, k)
        self.r_dist_indexed_ = local_statistic.dist
        self.r_ind_indexed_ = local_statistic.indices
        self.n_indexed_ = X.shape[0]

        return self

    def _local_statistic(self, X, k, return_indices: bool = True):
        # Find distances to the k nearest neighbors and their indices.
        # For standard LS, the single k-th nearest neighbor suffices.
        if self.effective_method_ == "nicdm":
            n_local_statistic = k + 1
            start = 1
        else:  # "ls", "standard"
            n_local_statistic = 1
            start = k
        end = k + 1

        n_samples = X.shape[0]
        dist = np.empty_like(X.data, shape=(n_samples, n_local_statistic))
        indices = np.empty_like(X.indices, shape=(n_samples, n_local_statistic))
        for i in tqdm(
            range(n_samples),
            disable=self.verbose < 2,
            desc=f'LS ({self.method}) stat'
        ):
            row = X.getrow(i)
            dist[i, :] = row.data[start:end]
            indices[i, :] = row.indices[start:end]

        # XXX better not extract them at all above
        if not return_indices:
            indices = None

        return namedtuple("LocalStatistic", ["dist", "indices"])(dist=dist, indices=indices)

    def transform(self, X, y=None) -> csr_matrix:
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

        X: csr_matrix = X.tocsr()
        n_query, n_indexed = X.shape
        n_neighbors = X.getrow(0).indices.size

        if n_neighbors == 1:
            warnings.warn(f'Cannot perform hubness reduction with a single neighbor per query. '
                          f'Skipping hubness reduction, and returning untransformed neighbor graph.')
            return X

        # increment to account for self neighborhood in pos. 0
        k = self.k + 1

        # Find distances to the k-th neighbor (standard LS) or the k neighbors (NICDM)
        r_dist_query = self._local_statistic(X, k, return_indices=False).dist

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
        kng = k_neighbors_graph(hub_reduced_dist, original_X=X, sort_distances=True)
        return kng


class LocalScaling(HubnessReduction):
    """ Hubness reduction with Local Scaling [1]_.

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

    def __init__(self, k: int = 5, method: str = 'standard', verbose: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.method = method
        self.verbose = verbose

    def fit(self, neigh_dist, neigh_ind, X=None, assume_sorted: bool = True, *args, **kwargs) -> LocalScaling:
        """ Fit the model using neigh_dist and neigh_ind as training data.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_samples, n_neighbors)
            Distance matrix of training objects (rows) against their
            individual k nearest neighbors (colums).

        neigh_ind: np.ndarray, shape (n_samples, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist.

        X: ignored

        assume_sorted: bool, default = True
            Assume input matrices are sorted according to neigh_dist.
            If False, these are sorted here.
        """
        # Check equal number of rows and columns
        check_consistent_length(neigh_ind, neigh_dist)
        check_consistent_length(neigh_ind.T, neigh_dist.T)

        # increment to include the k-th element in slicing
        k = self.k + 1

        # Find distances to the k-th neighbor (standard LS) or the k neighbors (NICDM)
        if assume_sorted:
            self.r_dist_train_ = neigh_dist[:, :k]
            self.r_ind_train_ = neigh_ind[:, :k]
        else:
            kth = np.arange(self.k)
            mask = np.argpartition(neigh_dist, kth=kth)[:, :k]
            self.r_dist_train_ = np.take_along_axis(neigh_dist, mask, axis=1)
            self.r_ind_train_ = np.take_along_axis(neigh_ind, mask, axis=1)

        return self

    def transform(self, neigh_dist, neigh_ind, X=None,
                  assume_sorted: bool = True, *args, **kwargs) -> (np.ndarray, np.ndarray):
        """ Transform distance between test and training data with Mutual Proximity.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_query, n_neighbors)
            Distance matrix of test objects (rows) against their individual
            k nearest neighbors among the training data (columns).

        neigh_ind: np.ndarray, shape (n_query, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist

        X: ignored

        assume_sorted: bool, default = True
            Assume input matrices are sorted according to neigh_dist.
            If False, these are partitioned here.

            NOTE: The returned matrices are never sorted.

        Returns
        -------
        hub_reduced_dist, neigh_ind
            Local scaling distances, and corresponding neighbor indices

        Notes
        -----
        The returned distances are NOT sorted! If you use this class directly,
        you will need to sort the returned matrices according to hub_reduced_dist.
        Classes from :mod:`skhubness.neighbors` do this automatically.
        """
        check_is_fitted(self, 'r_dist_train_')

        n_test, n_indexed = neigh_dist.shape

        if n_indexed == 1:
            warnings.warn(f'Cannot perform hubness reduction with a single neighbor per query. '
                          f'Skipping hubness reduction, and returning untransformed distances.')
            return neigh_dist, neigh_ind

        # increment to include the k-th element in slicing
        k = self.k + 1

        # Find distances to the k-th neighbor (standard LS) or the k neighbors (NICDM)
        if assume_sorted:
            r_dist_test = neigh_dist[:, :k]
        else:
            kth = np.arange(self.k)
            mask = np.argpartition(neigh_dist, kth=kth)[:, :k]
            r_dist_test = np.take_along_axis(neigh_dist, mask, axis=1)

        # Calculate LS or NICDM
        hub_reduced_dist = np.empty_like(neigh_dist)

        # Optionally show progress of local scaling loop
        disable_tqdm = False if self.verbose else True
        range_n_test = tqdm(range(n_test),
                            desc=f'LS {self.method}',
                            disable=disable_tqdm,
                            )

        # Perform standard local scaling...
        if self.method in ['ls', 'standard']:
            r_train = self.r_dist_train_[:, -1]
            r_test = r_dist_test[:, -1]
            for i in range_n_test:
                hub_reduced_dist[i, :] = \
                    1. - np.exp(-1 * neigh_dist[i] ** 2 / (r_test[i] * r_train[neigh_ind[i]]))
        # ...or use non-iterative contextual dissimilarity measure
        elif self.method == 'nicdm':
            r_train = self.r_dist_train_.mean(axis=1)
            r_test = r_dist_test.mean(axis=1)
            for i in range_n_test:
                hub_reduced_dist[i, :] = neigh_dist[i] / np.sqrt((r_test[i] * r_train[neigh_ind[i]]))
        else:
            raise ValueError(f"Internal: Invalid method {self.method}. Try 'ls' or 'nicdm'.")

        # Return the hubness reduced distances
        # These must be sorted downstream
        return hub_reduced_dist, neigh_ind
