# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_raises

from skhubness.reduction import MutualProximity
from skhubness.reduction.tests.reference_algorithms import ReferenceMutualProximity as ReferenceMutualProximity, _sort_neighbors

METHODS = [
    "normal",
    "exact",
]
ALLOWED_METHODS = [
    "exact",
    "empiric",
    "normal",
    "gaussi",
]


@pytest.mark.parametrize("method", ["normal", "empiric"])
def test_mp_kneighbors_graph_equals_mp_reference(method):
    if method == "empiric":
        pytest.xfail("Graph-based MP empiric implementation is known to "
                     "yield different results than reference. Need to investigate.")
    X, y = make_classification(
        n_samples=120,
        n_features=1000,
        random_state=1234,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20)

    # Reference MP
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(X_train, y_train)
    neigh_dist_train, neigh_ind_train = nn.kneighbors()
    neigh_dist_test, neigh_ind_test = nn.kneighbors(X_test)
    mp = ReferenceMutualProximity(method=method)
    mp.fit(neigh_dist_train, neigh_ind_train, X=None, assume_sorted=True)
    mp_dist_test, mp_ind_test = mp.transform(neigh_dist_test, neigh_ind_test, X=None, assume_sorted=True)

    # K-neighbors graph MP
    nn_graph = NearestNeighbors(n_neighbors=20)
    nn_graph.fit(X_train)
    graph_train = nn_graph.kneighbors_graph(mode="distance")
    graph_test = nn_graph.kneighbors_graph(X_test, mode="distance")
    mp_graph = MutualProximity(method=method)
    mp_graph.fit(graph_train)
    mp_graph_test = mp_graph.transform(graph_test)

    del X, X_test, X_train, y, y_test, y_train
    del graph_test, graph_train, mp, mp_graph, nn, nn_graph
    del neigh_ind_test, neigh_ind_train, neigh_dist_test, neigh_dist_train

    # Check correct neighbors
    np.testing.assert_array_equal(
        x=mp_ind_test.ravel(),
        y=mp_graph_test.indices.ravel(),
    )
    # Check (near) identical distances
    np.testing.assert_array_almost_equal(
        x=mp_dist_test.ravel(),
        y=mp_graph_test.data.ravel(),
        decimal=6,
    )


def test_reference_correct_mp_empiric():
    X, y = make_classification(n_samples=120, n_features=10, random_state=1234, )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20)
    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(X_train, y_train)
    neigh_dist_train, neigh_ind_train = nn.kneighbors()
    neigh_dist_test, neigh_ind_test = nn.kneighbors(X_test)

    # Calculate MP with fast vectorized routines
    mp = ReferenceMutualProximity(method="empiric")
    mp.fit(neigh_dist_train, neigh_ind_train, X=None, assume_sorted=True)
    mp_dist_test, mp_ind_test = mp.transform(neigh_dist_test, neigh_ind_test, X=None, assume_sorted=True)

    # Calculate MP in slow, naive loops
    mp_dist_test_correct = np.empty_like(neigh_dist_test, dtype=float)
    mp_ind_test_correct = np.empty_like(neigh_ind_test, dtype=int)
    n_test, n_train = neigh_ind_test.shape

    # Loop over all test distances
    for x in range(n_test):
        for y in range(n_train):
            idx = neigh_ind_test[x, y]
            d_xy = neigh_dist_test[x, y]
            set1 = set()
            set2 = set()
            # P(X > d_xy), i.e. how many distances from query x to indexed objects j
            # are greater than distance between x and y?
            for j, d_xj in zip(neigh_ind_test[x, :], neigh_dist_test[x, :]):
                if d_xj > d_xy:
                    set1.add(j)
            # P(Y > d_yx), i.e. how many distances from indexed object y to other indexed objects j
            # are greater than distance between y and x?
            for j in neigh_ind_test[x, :]:
                k = np.argwhere(neigh_ind_train[idx] == j).ravel()
                # Since we don't store all distances between all pairs of indexed objects,
                # this is approximated by setting all distance to not-nearest neighbors
                # to the distance to the k-th neighbor plus some epsilon
                d_yj = neigh_dist_train[idx, k] if k.size else neigh_dist_train[idx, -1] + 1e-6
                if d_yj > d_xy:
                    set2.add(j)
            mp_dist_test_correct[x, y] = 1 - (len(set1.intersection(set2)) / n_train)
            mp_ind_test_correct[x, y] = idx
    mp_dist_test_correct, mp_ind_test_correct = _sort_neighbors(mp_dist_test_correct, mp_ind_test_correct)
    np.testing.assert_array_almost_equal(mp_dist_test, mp_dist_test_correct)
    np.testing.assert_array_equal(mp_ind_test, mp_ind_test_correct)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("verbose", [0, 1])
def test_reference_mp_runs_without_error(method, verbose):
    X, y = make_classification(n_samples=20, n_features=10)
    nn = NearestNeighbors()
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    mp = ReferenceMutualProximity(method=method, verbose=verbose)
    _ = mp.fit(neigh_dist, neigh_ind, X, assume_sorted=True)\
          .transform(neigh_dist, neigh_ind, X, assume_sorted=True)


@pytest.mark.parametrize("method", ["invalid", None])
def test_reference_invalid_method(method):
    X, y = make_classification(n_samples=10, )
    nn = NearestNeighbors()
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    mp = ReferenceMutualProximity(method=method)
    with assert_raises(ValueError):
        mp.fit(neigh_dist, neigh_ind, X, assume_sorted=True)
