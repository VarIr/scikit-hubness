# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_is_fitted, check_array

from ._base import HubnessReduction
from skhubness.utils.kneighbors_graph import check_kneighbors_graph, check_matching_n_indexed
from skhubness.utils.kneighbors_graph import hubness_reduced_k_neighbors_graph


class DisSimLocal(HubnessReduction, TransformerMixin):
    """ Hubness reduction with DisSimLocal [1]_ in an sklearn-compatible kneighbors_graph.

    Parameters
    ----------
    k: int, default = 5
        Number of neighbors to consider for the local centroids
    return_squared_distances: bool, default = True
        DisSimLocal operates on squared Euclidean distances.
        If True, also return (quasi) squared Euclidean distances;
        if False, return (quasi) Euclidean distances instead.

    References
    ----------
    .. [1] Hara K, Suzuki I, Kobayashi K, Fukumizu K, Radovanović M (2016)
           Flattening the density gradient for eliminating spatial centrality to reduce hubness.
           In: Proceedings of the 30th AAAI conference on artificial intelligence, pp 1659–1665.
           https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12055
    """
    def __init__(
            self,
            k: int = 5,
            return_squared_distances: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.k = k
        self.return_squared = return_squared_distances

    def fit(self, X: csr_matrix, y=None, **kwargs) -> DisSimLocal:
        """ Extract DisSimLocal parameters.

        Parameters
        ----------
        X : sparse matrix of shape (n_indexed, n_indexed)
            Squared Euclidean distances in sorted sparse neighbors graph,
            such as obtained from sklearn.neighbors.kneighbors_graph.
            Each row i must contain the neighbors of X_i in order of increasing distances,
            that is, nearest neighbor first.
        y : ignored
        vectors : array-like of shape (n_indexed, n_features)
            Indexed objects in vector representation.
            DisSimLocal is formulated specifically for Euclidean distances, and requires access to the objects
            in the original vector space. (In contrast to other hubness reduction methods, like mutual proximity
            or local scaling, that operate on dissimilarity data).

        Returns
        -------
        self

        Notes
        -----
        Ensure sorting when using custom (approximate) neighbors implementations.
        DisSimLocal strictly requires squared Euclidean distances, and may return undefined values otherwise.
        """
        X = check_kneighbors_graph(X)
        n_indexed = X.shape[0]
        n_neighbors = X.indptr[1]

        vectors_indexed = kwargs.get("vectors", None)
        if vectors_indexed is None:
            raise ValueError("DisSimLocal requires vector data in addition to the k-neighbors graph. "
                             "Please provide them as: fit(kng, vectors=X_indexed).")
        vectors_indexed: np.ndarray = check_array(vectors_indexed)  # noqa
        if vectors_indexed.shape[0] != n_indexed:
            raise ValueError("Number of objects in `vectors` does not match number of objects in `X`.")

        try:
            if self.k <= 0:
                raise ValueError(f"Expected k > 0. Got {self.k}")
        except TypeError:  # Why?
            raise TypeError(f'Expected k: int > 0. Got {self.k}')
        k = self.k

        if k > n_neighbors:
            k = n_neighbors
            warnings.warn(f'Neighborhood parameter k larger than number of provided neighbors in X. Reducing to k={k}.')

        # Calculate local neighborhood centroids among the training points
        ind_knn = X.indices.reshape(n_indexed, -1)[:, :k]
        centroids_indexed = vectors_indexed[ind_knn].mean(axis=1)
        dist_to_cent = row_norms(vectors_indexed - centroids_indexed, squared=True)

        self.centroids_indexed_ = centroids_indexed
        self.dist_to_centroids_indexed_ = dist_to_cent
        self.n_indexed_ = n_indexed

        return self

    def transform(self, X: csr_matrix, y=None, **kwargs) -> csr_matrix:
        """ Transform distance between query and indexed data with DisSimLocal.

        Parameters
        ----------
        X : sparse matrix of shape (n_query, n_indexed)
            Squared Euclidean distances in sorted sparse neighbors graph,
            such as obtained from sklearn.neighbors.kneighbors_graph.
            Each row i must contain the neighbors of X_i in order of increasing distances,
            that is, nearest neighbor first.
        y : ignored
        vectors : array-like of shape (n_query, n_features)
            Query objects in vector representation.
            DisSimLocal is formulated specifically for Euclidean distances, and requires access to the objects
            in the original vector space. (In contrast to other hubness reduction methods, like mutual proximity
            or local scaling, that operate on dissimilarity data).

        Returns
        -------
        A : sparse matrix of shape (n_query, n_indexed)
            Hubness-reduced graph where A[i, j] is assigned the weight of edge that connects i to j.
            The matrix is of CSR format.

        Notes
        -----
        Ensure sorting when using custom (approximate) neighbors implementations.
        DisSimLocal strictly requires squared Euclidean distances, and returns undefined values otherwise.
        """
        check_is_fitted(self, ["centroids_indexed_", "dist_to_centroids_indexed_"])
        vectors_query = kwargs.get("vectors", None)
        if vectors_query is None:
            raise ValueError("DisSimLocal requires vector data in addition to the k-neighbors graph. "
                             "Please provide them as: transform(kng, vectors=X_query).")
        vectors_query: np.ndarray = check_array(vectors_query, copy=True)  # noqa
        X_query: csr_matrix = check_kneighbors_graph(X)
        check_matching_n_indexed(X_query, self.n_indexed_)

        n_query, n_indexed = X_query.shape
        n_neighbors = X_query.indptr[1]

        if vectors_query.shape[0] != n_query:
            raise ValueError("Number of objects in `vectors` does not match number of objects in `X`.")

        if n_neighbors == 1:
            warnings.warn("Cannot perform hubness reduction with a single neighbor per query. "
                          "Skipping hubness reduction, and returning untransformed distances.")
            return X_query

        k = self.k
        if k > n_neighbors:
            k = n_neighbors
            warnings.warn(f'Neighborhood parameter k larger than number of provided neighbors in X. Reducing to k={k}.')

        # Calculate local neighborhood centroids for query objects among indexed objects
        neigh_dist = X_query.data.reshape(n_query, n_neighbors)
        neigh_ind = X_query.indices.reshape(n_query, -1)
        knn = neigh_ind[:, :k]
        centroids_indexed = self.centroids_indexed_[knn].mean(axis=1)

        vectors_query -= centroids_indexed
        vectors_query **= 2
        X_query_dist_to_centroids = vectors_query.sum(axis=1)
        X_indexed_dist_to_centroids = self.dist_to_centroids_indexed_[neigh_ind]

        hub_reduced_dist = neigh_dist
        hub_reduced_dist -= X_query_dist_to_centroids[:, np.newaxis]
        hub_reduced_dist -= X_indexed_dist_to_centroids

        # DisSimLocal can yield negative dissimilarities, which can cause problems with
        # certain scikit-learn routines (e.g. in metric='precomputed' usages).
        # We, therefore, shift dissimilarities to non-negative values, if necessary.
        min_dist = hub_reduced_dist.min(initial=0.)
        if min_dist < 0.:
            hub_reduced_dist -= min_dist

        # Return Euclidean or squared Euclidean distances?
        if not self.return_squared:
            np.sqrt(hub_reduced_dist, out=hub_reduced_dist)

        # Return the sorted hubness reduced kneighbors graph
        return hubness_reduced_k_neighbors_graph(hub_reduced_dist, original_X=X_query, sort_distances=True)
