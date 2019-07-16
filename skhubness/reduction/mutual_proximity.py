# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import warnings

import numpy as np
from scipy import stats
from sklearn.utils.validation import check_is_fitted, check_consistent_length, check_array
from tqdm.auto import tqdm

from .base import HubnessReduction


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
            individual k nearest neighbors (colums).

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
        n_train = self.n_train

        if n_indexed == 1:
            warnings.warn(f'Cannot perform hubness reduction with a single neighbor per query. '
                          f'Skipping hubness reduction, and returning untransformed distances.')
            return neigh_dist, neigh_ind

        hub_reduced_dist = np.empty_like(neigh_dist)
        # Show progress in hubness reduction loop
        if self.verbose:
            range_n_test = tqdm(range(n_test), total=n_test, desc=f'MP ({self.method})')
        else:
            range_n_test = range(n_test)

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
            for i in range_n_test:
                dI = neigh_dist[i, :][np.newaxis, :]  # broadcasted afterwards
                dJ = self.neigh_dist_train_[neigh_ind[i], :n_indexed]
                d = dI.T
                # div by n
                n_pts = n_train
                hub_reduced_dist[i, :] = 1. - (np.sum((dI > d) & (dJ > d), axis=1) / n_pts)
        else:
            raise ValueError(f"Internal: Invalid method {self.method}.")

        # Return the hubness reduced distances
        # These must be sorted downstream
        return hub_reduced_dist, neigh_ind
