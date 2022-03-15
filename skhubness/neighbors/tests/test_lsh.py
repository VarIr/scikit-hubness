# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
import sys

from sklearn.datasets import make_classification
from sklearn.preprocessing import Normalizer
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator
from skhubness.neighbors import LegacyPuffinn, PuffinnTransformer

# Exclude libraries that are not available on specific platforms
LSH_LEGACY_KNN = []
LSH_LEGACY_RADIUS = []
LSH_TRAFO_KNN = []
LSH_TRAFO_RADIUS = []
if sys.platform == "win32":
    pass  # Currently, none available
elif sys.platform == "darwin":
    # Work-around for imprecise Puffinn on Mac: disable tests for now
    pass
elif sys.platform == "linux":
    LSH_LEGACY_KNN.append(LegacyPuffinn)
    LSH_TRAFO_KNN.append(PuffinnTransformer)
LSH_LEGACY = set(LSH_LEGACY_KNN + LSH_LEGACY_RADIUS)
LSH_TRAFO = set(LSH_TRAFO_KNN + LSH_TRAFO_RADIUS)
LSH_ALL = LSH_LEGACY.union(LSH_TRAFO)


@pytest.mark.parametrize("LSH", LSH_ALL)
def test_estimator(LSH):
    check_estimator(LSH())


@pytest.mark.parametrize("LSH", LSH_LEGACY_KNN)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("n_jobs", [-1, 1, None])
@pytest.mark.parametrize("verbose", [0, 1])
def test_kneighbors_with_or_without_self_hit(LSH, metric, n_jobs, verbose):
    X, y = make_classification(random_state=234)
    X = Normalizer().fit_transform(X)
    lsh = LSH(metric=metric, n_jobs=n_jobs, verbose=verbose)
    lsh.fit(X, y)
    neigh_dist, neigh_ind = lsh.kneighbors(return_distance=True)
    neigh_dist_self, neigh_ind_self = lsh.kneighbors(X, return_distance=True)

    ind_only = lsh.kneighbors(return_distance=False)
    ind_only_self = lsh.kneighbors(X, return_distance=False)

    assert_array_equal(neigh_ind, ind_only)
    assert_array_equal(neigh_ind_self, ind_only_self)

    assert (neigh_ind - neigh_ind_self).mean() <= .01, "More than 1% of neighbors mismatch"
    assert ((neigh_dist - neigh_dist_self) < 0.0001).mean() <= 0.01,\
        "Not almost equal to 4 decimals in more than 1% of neighbor slots"


@pytest.mark.parametrize("LSH", LSH_LEGACY_RADIUS)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("n_jobs", [-1, 1, None])
@pytest.mark.parametrize("verbose", [0, 1])
def test_radius_neighbors_with_or_without_self_hit(LSH, metric, n_jobs, verbose):
    X, y = make_classification()
    X = Normalizer().fit_transform(X)
    lsh = LSH(metric=metric, n_jobs=n_jobs, verbose=verbose)
    lsh.fit(X, y)
    radius = lsh.kneighbors(n_candidates=3)[0][:, 2].max()
    neigh_dist, neigh_ind = lsh.radius_neighbors(return_distance=True, radius=radius)
    neigh_dist_self, neigh_ind_self = lsh.radius_neighbors(X, return_distance=True, radius=radius)

    ind_only = lsh.radius_neighbors(return_distance=False, radius=radius)
    ind_only_self = lsh.radius_neighbors(X, return_distance=False, radius=radius)

    assert len(neigh_ind) == len(neigh_ind_self) == len(neigh_dist) == len(neigh_dist_self)
    for i in range(len(neigh_ind)):
        assert_array_equal(neigh_ind[i], ind_only[i])
        assert_array_equal(neigh_ind_self[i], ind_only_self[i])

        assert_array_equal(neigh_ind[i][:3],
                           neigh_ind_self[i][1:4])
        assert_array_almost_equal(neigh_dist[i][:3],
                                  neigh_dist_self[i][1:4])


@pytest.mark.parametrize("LSH", LSH_LEGACY)
def test_squared_euclidean_same_neighbors_as_euclidean(LSH):
    X, y = make_classification(random_state=234)
    X = Normalizer().fit_transform(X)
    lsh = LSH(metric="minkowski")
    lsh.fit(X, y)
    neigh_dist_eucl, neigh_ind_eucl = lsh.kneighbors()

    lsh_sq = LSH(metric="sqeuclidean")
    lsh_sq.fit(X, y)
    neigh_dist_sqeucl, neigh_ind_sqeucl = lsh_sq.kneighbors()

    assert_array_equal(neigh_ind_eucl, neigh_ind_sqeucl)
    assert_array_almost_equal(neigh_dist_eucl ** 2, neigh_dist_sqeucl)

    if LSH in LSH_LEGACY_RADIUS:
        radius = neigh_dist_eucl[:, 2].max()
        rad_dist_eucl, rad_ind_eucl = lsh.radius_neighbors(radius=radius)
        rad_dist_sqeucl, rad_ind_sqeucl = lsh_sq.radius_neighbors(radius=radius**2)
        for i in range(len(rad_ind_eucl)):
            assert_array_equal(rad_ind_eucl[i], rad_ind_sqeucl[i])
            assert_array_almost_equal(rad_dist_eucl[i] ** 2, rad_dist_sqeucl[i])


@pytest.mark.parametrize("LSH", LSH_LEGACY)
@pytest.mark.parametrize("metric", ["invalid", "manhattan", "l1", "chebyshev"])
def test_warn_on_invalid_metric(LSH, metric):
    X, y = make_classification(random_state=24643)
    X = Normalizer().fit_transform(X)
    lsh = LSH(metric="euclidean")
    lsh.fit(X, y)
    neigh_dist, neigh_ind = lsh.kneighbors()

    lsh.metric = metric
    with pytest.warns(UserWarning):
        lsh.fit(X, y)
    neigh_dist_inv, neigh_ind_inv = lsh.kneighbors()

    assert_array_equal(neigh_ind, neigh_ind_inv)
    assert_array_almost_equal(neigh_dist, neigh_dist_inv)


@pytest.mark.parametrize("LSH", LSH_TRAFO)
@pytest.mark.parametrize("metric", ["invalid", None])
def test_invalid_metric(LSH, metric):
    X = np.empty((10, 100))
    lsh = LSH(metric=metric)
    with pytest.raises((ValueError, TypeError)):
        lsh.fit(X)


@pytest.mark.skipif(sys.platform == "win32", reason="Puffinn not supported on Windows.")
def test_puffinn_lsh_custom_memory():
    # If user decides to set memory, this value should be selected,
    # if it is higher than what the heuristic yields.
    X, y = make_classification(n_samples=10)
    memory = 2*1024**2
    lsh = LegacyPuffinn(n_candidates=2,
                        memory=memory)
    lsh.fit(X, y)
    assert lsh.memory == memory


@pytest.mark.parametrize("metric", ["angular", "jaccard"])
def test_transformer_vs_legacy_puffinn(metric):
    X, y = make_classification(random_state=123)
    if metric == "jaccard":
        X /= X.max() / 2
        X = X.astype(np.bool)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    memory = 20_000_000 + np.multiply(*X.shape) * 8 if metric == "jaccard" else None
    legacy_puffinn = LegacyPuffinn(metric=metric, memory=memory)
    legacy_puffinn.fit(X_train, y_train)
    hnsw_neigh_dist, hnsw_neigh_ind = legacy_puffinn.kneighbors(X_test, return_distance=True)

    puffinn_trafo = PuffinnTransformer(metric=metric, memory=memory)
    puffinn_trafo.fit(X_train, y_train)
    nms_graph = puffinn_trafo.transform(X_test)

    # Check that both old and new Puffinn wrapper yield identical nearest neighbors and distances
    np.testing.assert_array_equal(hnsw_neigh_ind.ravel(), nms_graph.indices.ravel())
    np.testing.assert_array_almost_equal(hnsw_neigh_dist.ravel(), nms_graph.data.ravel())
