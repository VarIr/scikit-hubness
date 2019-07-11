# -*- coding: utf-8 -*-
from __future__ import annotations
import warnings

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from tqdm.auto import tqdm


class LocalScaling:
    """ Hubness reduction with local scaling. """

    def __init__(self, k: int = 5, method: str = 'standard', verbose: int = 0):
        self.k = k
        self.method = method
        self.verbose = verbose

    def fit(self, neigh_dist, neigh_ind, assume_sorted: bool = True) -> LocalScaling:
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

    def transform(self, neigh_dist, neigh_ind, assume_sorted: bool = True, *args, **kwargs) -> (np.ndarray, np.ndarray):
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
        if self.verbose:
            range_n_test = tqdm(range(n_test),
                                total=n_test,
                                desc=f'LS {self.method}')
        else:
            range_n_test = range(n_test)

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
