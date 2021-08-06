# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from abc import ABC, abstractmethod
import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import euclidean_distances
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_is_fitted, check_consistent_length, check_array
from tqdm.auto import tqdm


__all__ = [
    "MutualProximity",
    "LocalScaling",
    "DisSimLocal",
    "NoHubnessReduction",
]


def _sort_neighbors(dist, ind):
    mask = np.argsort(dist)
    dist = np.take_along_axis(dist, mask, axis=1)
    ind = np.take_along_axis(ind, mask, axis=1)
    return dist, ind


class HubnessReduction(ABC):
    """ Base class for hubness reduction. """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, neigh_dist, neigh_ind, X, assume_sorted, *args, **kwargs):
        pass  # pragma: no cover

    @abstractmethod
    def transform(self, neigh_dist, neigh_ind, X, assume_sorted, return_distance=True):
        pass  # pragma: no cover

    def fit_transform(self, neigh_dist, neigh_ind, X, assume_sorted=True, return_distance=True, *args, **kwargs):
        """ Equivalent to call .fit().transform() """
        self.fit(neigh_dist, neigh_ind, X, assume_sorted, *args, **kwargs)
        return self.transform(neigh_dist, neigh_ind, X, assume_sorted, return_distance)


class NoHubnessReduction(HubnessReduction):
    """ Compatibility class for neighbor search without hubness reduction. """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):
        pass  # pragma: no cover

    def transform(self, neigh_dist, neigh_ind, X, assume_sorted=True, return_distance=True, *args, **kwargs):
        """ Equivalent to call .fit().transform() """
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind


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

        # Return the sorted hubness reduced distances
        return _sort_neighbors(hub_reduced_dist, neigh_ind)


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

        # Return the sorted hubness reduced distances
        return _sort_neighbors(hub_reduced_dist, neigh_ind)


class DisSimLocal(HubnessReduction):
    """ Hubness reduction with DisSimLocal [1]_.

    Parameters
    ----------
    k: int, default = 5
        Number of neighbors to consider for the local centroids

    squared: bool, default = True
        DisSimLocal operates on squared Euclidean distances.
        If True, return (quasi) squared Euclidean distances;
        if False, return (quasi) Eucldean distances.

    References
    ----------
    .. [1] Hara K, Suzuki I, Kobayashi K, Fukumizu K, Radovanović M (2016)
           Flattening the density gradient for eliminating spatial centrality to reduce hubness.
           In: Proceedings of the 30th AAAI conference on artificial intelligence, pp 1659–1665.
           https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12055
    """
    def __init__(self, k: int = 5, squared: bool = True, *args, **kwargs):
        super().__init__()
        self.k = k
        self.squared = squared

    def fit(self, neigh_dist: np.ndarray, neigh_ind: np.ndarray, X: np.ndarray,
            assume_sorted: bool = True, *args, **kwargs) -> DisSimLocal:
        """ Fit the model using X, neigh_dist, and neigh_ind as training data.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_samples, n_neighbors)
            Distance matrix of training objects (rows) against their
            individual k nearest neighbors (colums).

        neigh_ind: np.ndarray, shape (n_samples, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist.

        X: np.ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of vectors,
            and n_features their dimensionality (number of features).

        assume_sorted: bool, default = True
            Assume input matrices are sorted according to neigh_dist.
            If False, these are sorted here.
        """
        # Check equal number of rows and columns
        check_consistent_length(neigh_ind, neigh_dist)
        check_consistent_length(neigh_ind.T, neigh_dist.T)
        X: np.ndarray = check_array(X)  # noqa
        try:
            if self.k <= 0:
                raise ValueError(f"Expected k > 0. Got {self.k}")
        except TypeError:
            raise TypeError(f'Expected k: int > 0. Got {self.k}')

        k = self.k
        if k > neigh_ind.shape[1]:
            warnings.warn(f'Neighborhood parameter k larger than provided neighbors in neigh_dist, neigh_ind. '
                          f'Will reduce to k={neigh_ind.shape[1]}.')
            k = neigh_ind.shape[1]

        # Calculate local neighborhood centroids among the training points
        if assume_sorted:
            knn = neigh_ind[:, :k]
        else:
            mask = np.argpartition(neigh_dist, kth=k-1)[:, :k]
            knn = np.take_along_axis(neigh_ind, mask, axis=1)
        centroids = X[knn].mean(axis=1)
        dist_to_cent = row_norms(X - centroids, squared=True)

        self.X_train_ = X
        self.X_train_centroids_ = centroids
        self.X_train_dist_to_centroids_ = dist_to_cent

        return self

    def transform(self, neigh_dist: np.ndarray, neigh_ind: np.ndarray, X: np.ndarray,
                  assume_sorted: bool = True, *args, **kwargs) -> (np.ndarray, np.ndarray):
        """ Transform distance between test and training data with DisSimLocal.

        Parameters
        ----------
        neigh_dist: np.ndarray, shape (n_query, n_neighbors)
            Distance matrix of test objects (rows) against their individual
            k nearest neighbors among the training data (columns).

        neigh_ind: np.ndarray, shape (n_query, n_neighbors)
            Neighbor indices corresponding to the values in neigh_dist

        X: np.ndarray, shape (n_query, n_features)
            Test data, where n_query is the number of vectors,
            and n_features their dimensionality (number of features).

        assume_sorted: ignored

        Returns
        -------
        hub_reduced_dist, neigh_ind
            DisSimLocal distances, and corresponding neighbor indices

        Notes
        -----
        The returned distances are NOT sorted! If you use this class directly,
        you will need to sort the returned matrices according to hub_reduced_dist.
        Classes from :mod:`skhubness.neighbors` do this automatically.
        """
        check_is_fitted(self, ['X_train_', 'X_train_centroids_', 'X_train_dist_to_centroids_'])
        if X is None:
            X = self.X_train_
        else:
            X = check_array(X)

        n_test, n_indexed = neigh_dist.shape

        if n_indexed == 1:
            warnings.warn(f'Cannot perform hubness reduction with a single neighbor per query. '
                          f'Skipping hubness reduction, and returning untransformed distances.')
            return neigh_dist, neigh_ind

        k = self.k
        if k > neigh_ind.shape[1]:
            warnings.warn(f'Neighborhood parameter k larger than provided neighbors in neigh_dist, neigh_ind. '
                          f'Will reduce to k={neigh_ind.shape[1]}.')
            k = neigh_ind.shape[1]

        # Calculate local neighborhood centroids for test objects among training objects
        mask = np.argpartition(neigh_dist, kth=k-1)
        # neigh_dist = np.take_along_axis(neigh_dist, mask, axis=1)
        for i, ind in enumerate(neigh_ind):
            neigh_dist[i, :] = euclidean_distances(X[i].reshape(1, -1), self.X_train_[ind], squared=True)
        neigh_ind = np.take_along_axis(neigh_ind, mask, axis=1)
        knn = neigh_ind[:, :k]
        centroids = self.X_train_centroids_[knn].mean(axis=1)

        X_test = X - centroids
        X_test **= 2
        X_test_dist_to_centroids = X_test.sum(axis=1)
        X_train_dist_to_centroids = self.X_train_dist_to_centroids_[neigh_ind]

        hub_reduced_dist = neigh_dist.copy()
        hub_reduced_dist -= X_test_dist_to_centroids[:, np.newaxis]
        hub_reduced_dist -= X_train_dist_to_centroids

        # DisSimLocal can yield negative dissimilarities, which can cause problems with
        # certain scikit-learn routines (e.g. in metric='precomputed' usages).
        # We, therefore, shift dissimilarities to non-negative values, if necessary.
        min_dist = hub_reduced_dist.min()
        if min_dist < 0.:
            hub_reduced_dist += (-min_dist)

        # Return Euclidean or squared Euclidean distances?
        if not self.squared:
            hub_reduced_dist **= (1 / 2)

        # Return the sorted hubness reduced distances
        return _sort_neighbors(hub_reduced_dist, neigh_ind)
