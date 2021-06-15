# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import warnings

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_consistent_length, check_array
from tqdm.auto import tqdm

from .base import GraphHubnessReduction, HubnessReduction
from ..utils.helper import k_neighbors_graph


class GraphMutualProximity(GraphHubnessReduction, TransformerMixin):
    """ Hubness reduction with Mutual Proximity [1]_ in an sklearn-compatible kneighbors_graph.

    Parameters
    ----------
    method: "normal" or "empiric", default = "normal"
        Model distance distribution with "method".

        - "normal" (="gaussi") models distance distributions with independent Gaussians (fast)
        - "empiric" (="exact") models distances with the empiric distributions (slow)

    verbose: int, default = 0
        If verbose > 0, show progress bar.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """

    def __init__(self, method: str = "normal", verbose: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.effective_method_ = "normal" if method.lower() in "normal gaussi" else "empiric"
        self.verbose = verbose

    def fit(self, X: csr_matrix, y=None, **kwargs) -> GraphHubnessReduction:
        """ Extract mutual proximity parameters.

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

        X = X.tocsr()
        n_indexed = X.shape[0]
        self.n_indexed_ = n_indexed

        if self.effective_method_ == "empiric":
            self.X_indexed_ = X
        elif self.effective_method_ == "normal":
            self.mu_indexed_ = np.nanmean(X.data.reshape(n_indexed, -1), axis=1)
            self.sd_indexed_ = np.nanstd(X.data.reshape((n_indexed, -1)), axis=1, ddof=0)
        else:
            raise ValueError(f'Mutual proximity method "{self.method}" not recognized. Try "normal" or "empiric".')

        return self

    def transform(self, X, y=None, **kwargs) -> csr_matrix:
        """ Transform distance between query and indexed data with Mutual Proximity.

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

        Notes
        -----
        This mutual proximity implementation assumes symmetric dissimilarities.
        """
        check_is_fitted(self, ['mu_indexed_', 'sd_indexed_', 'X_indexed_'], all_or_any=any)
        # check_kneighbors_graph(kng)  # TODO

        X: csr_matrix = X.tocsr()
        n_query, n_indexed = X.shape
        n_neighbors = X.getrow(0).data.size

        if n_neighbors == 1:
            warnings.warn(f'Cannot perform hubness reduction with a single neighbor per query. '
                          f'Skipping hubness reduction, and returning untransformed distances.')
            return X

        # Initialize all values to 1, so that we can inplace subtract mutual proximity (similarity) scores later on
        hub_reduced_dist = np.ones_like(X.data, shape=(n_query, n_neighbors), dtype=X.data.dtype)

        # Show progress in hubness reduction transformation loop
        range_n_query = tqdm(
            range(n_query),
            desc=f"MP ({self.effective_method_.lower()}) trafo",
            disable=self.verbose < 1,
        )

        # Calculate MP with independent Gaussians
        if self.effective_method_ == "normal":
            mu_indexed = self.mu_indexed_
            sd_indexed = self.sd_indexed_
            for i in range_n_query:
                row = X.getrow(i)
                j_mom = row.indices
                mu = np.nanmean(row.data)
                sd = np.nanstd(row.data, ddof=0)
                p1 = stats.norm.sf(row.data, mu, sd)
                p2 = stats.norm.sf(row.data, mu_indexed[j_mom], sd_indexed[j_mom])
                hub_reduced_dist[i, :] -= (p1 * p2).ravel()
        # Calculate MP empiric (slow)
        elif self.effective_method_ == "empiric":
            n_samples = self.X_indexed_.shape[1]
            for x in range_n_query:
                d_xj = X.getrow(x)
                for idx_y, (y, d_xy) in enumerate(zip(d_xj.indices, d_xj.data)):
                    d_yj = self.X_indexed_.getrow(y)

                    # Since X comes sorted, we already know all d_{x,j} > d_{x,y}, i.e., all afterwards
                    pos_jx_greater = idx_y + 1
                    j_jx_greater = d_xj.indices[pos_jx_greater:]
                    # Also consider all k neighbors of y (but not those objects that are not NN to x or y)
                    d_yj_greater = np.setdiff1d(d_yj.indices, d_xj.indices, assume_unique=True)
                    j_jx_greater = np.union1d(j_jx_greater, d_yj_greater)

                    # Keep at maximum MP-distance (=1), because intersection(A, B) is necessarily empty, if A={}
                    if len(j_jx_greater) == 0:
                        continue

                    # Finding d_{y,j} > d_{y,x} requires searching for d_{y,x} in the indexed neighbors graph.
                    # Actually, we have no way of knowing d_{y,x} (query x not available during indexing, and query
                    # graph is not computed twice for both directions). Instead, we have to use d_{x,y} as a substitute,
                    # which is equivalent for symmetric dissimilarities, and might still be acceptable
                    # in case of mildly asymmetric dissimilarities.
                    d_yj_max = d_yj.data[-1]

                    # Keep at maximum MP-distance (=1), because intersection(A, B) is necessarily empty, if B={}
                    if d_xy >= d_yj_max:
                        continue

                    # Only compute the intersection, if both A and B are non-empty
                    pos_yj_greater = np.searchsorted(a=d_yj.data, v=d_xy, side="right")
                    j_yj_greater = d_yj.indices[pos_yj_greater:]
                    d_xj_greater = np.setdiff1d(d_xj.indices, d_yj.indices, assume_unique=True)
                    j_yj_greater = np.union1d(j_yj_greater, d_xj_greater)

                    # MP(d_{x,y}) := fraction of objects j with distance to x and y greater than d_{x,y} among all obj.
                    j_both_greater: np.ndarray = np.intersect1d(j_jx_greater, j_yj_greater, assume_unique=True)  # noqa
                    # n("All") objects: number of stored neighbors in the graph, which is also the highest possible n(j)
                    hub_reduced_dist[x, idx_y] -= j_both_greater.size / n_samples
        else:
            raise ValueError(f"Internal: Invalid method {self.effective_method_}.")

        # Return the sorted hubness reduced kneighbors graph
        return k_neighbors_graph(hub_reduced_dist, original_X=X, sort_distances=True)


class MutualProximity(HubnessReduction):
    """ Hubness reduction with Mutual Proximity [1]_.

    Parameters
    ----------
    method: 'normal' or 'empiric', default = 'normal'
        Model distance distribution with 'method'.

        - 'normal' or 'gaussi' model distance distributions with independent Gaussians (fast)
        - 'empiric' or 'exact' model distances with the empiric distributions (slow)

    verbose: int, default = 0
        If verbose > 0, show progress bar.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """

    def __init__(self, method: str = 'normal', verbose: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.verbose = verbose

    def fit(self, neigh_dist, neigh_ind, X=None, assume_sorted=None, *args, **kwargs) -> MutualProximity:
        """ Fit the model using neigh_dist and neigh_ind as training data.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_samples, n_neighbors)
            Distance matrix of training objects (rows) against their
            individual k nearest neighbors (columns).

        neigh_ind: np.ndarray, shape (n_samples, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist.

        X: ignored

        assume_sorted: ignored
        """
        # Check equal number of rows and columns
        check_consistent_length(neigh_ind, neigh_dist)
        check_consistent_length(neigh_ind.T, neigh_dist.T)
        check_array(neigh_dist, force_all_finite=False)
        check_array(neigh_ind)

        self.n_train = neigh_dist.shape[0]

        if self.method in ['exact', 'empiric']:
            self.method = 'empiric'
            self.neigh_dist_train_ = neigh_dist
            self.neigh_ind_train_ = neigh_ind
        elif self.method in ['normal', 'gaussi']:
            self.method = 'normal'
            self.mu_train_ = np.nanmean(neigh_dist, axis=1)
            self.sd_train_ = np.nanstd(neigh_dist, axis=1, ddof=0)
        else:
            raise ValueError(f'Mutual proximity method "{self.method}" not recognized. Try "normal" or "empiric".')

        return self

    def transform(self, neigh_dist, neigh_ind, X=None, assume_sorted=None, *args, **kwargs):
        """ Transform distance between test and training data with Mutual Proximity.

        Parameters
        ----------
        neigh_dist: np.ndarray
            Distance matrix of test objects (rows) against their individual
            k nearest neighbors among the training data (columns).

        neigh_ind: np.ndarray
            Neighbor indices corresponding to the values in neigh_dist

        X: ignored

        assume_sorted: ignored

        Returns
        -------
        hub_reduced_dist, neigh_ind
            Mutual Proximity distances, and corresponding neighbor indices

        Notes
        -----
        The returned distances are NOT sorted! If you use this class directly,
        you will need to sort the returned matrices according to hub_reduced_dist.
        Classes from :mod:`skhubness.neighbors` do this automatically.
        """
        check_is_fitted(self, ['mu_train_', 'sd_train_', 'neigh_dist_train_', 'neigh_ind_train_'], all_or_any=any)
        check_array(neigh_dist, force_all_finite='allow-nan')
        check_array(neigh_ind)

        n_test, n_indexed = neigh_dist.shape

        if n_indexed == 1:
            warnings.warn(f'Cannot perform hubness reduction with a single neighbor per query. '
                          f'Skipping hubness reduction, and returning untransformed distances.')
            return neigh_dist, neigh_ind

        hub_reduced_dist = np.empty_like(neigh_dist)

        # Show progress in hubness reduction loop
        disable_tqdm = False if self.verbose else True
        range_n_test = tqdm(range(n_test),
                            desc=f'MP ({self.method})',
                            disable=disable_tqdm,
                            )

        # Calculate MP with independent Gaussians
        if self.method == 'normal':
            mu_train = self.mu_train_
            sd_train = self.sd_train_
            for i in range_n_test:
                j_mom = neigh_ind[i]
                mu = np.nanmean(neigh_dist[i])
                sd = np.nanstd(neigh_dist[i], ddof=0)
                p1 = stats.norm.sf(neigh_dist[i, :], mu, sd)
                p2 = stats.norm.sf(neigh_dist[i, :], mu_train[j_mom], sd_train[j_mom])
                hub_reduced_dist[i, :] = (1 - p1 * p2).ravel()
        # Calculate MP empiric (slow)
        elif self.method == 'empiric':
            max_ind = self.neigh_ind_train_.max()
            for i in range_n_test:
                dI = neigh_dist[i, :][np.newaxis, :]  # broadcasted afterwards
                dJ = np.zeros((dI.size, n_indexed))
                for j in range(n_indexed):
                    tmp = np.zeros(max_ind + 1) + (self.neigh_dist_train_[neigh_ind[i, j], -1] + 1e-6)
                    tmp[self.neigh_ind_train_[neigh_ind[i, j]]] = self.neigh_dist_train_[neigh_ind[i, j]]
                    dJ[j, :] = tmp[neigh_ind[i]]
                # dJ = self.neigh_dist_train_[neigh_ind[i], :n_indexed]
                d = dI.T
                hub_reduced_dist[i, :] = 1. - (np.sum((dI > d) & (dJ > d), axis=1) / n_indexed)
        else:
            raise ValueError(f"Internal: Invalid method {self.method}.")

        # Return the hubness reduced distances
        # These must be sorted downstream
        return hub_reduced_dist, neigh_ind
