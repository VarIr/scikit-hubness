# SPDX-License-Identifier: BSD-3-Clause
import sys
import pytest
import numpy as np
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils._testing import assert_raises
from skhubness.neighbors import LegacyNNG, NGTTransformer
from sklearn.neighbors import NearestNeighbors


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
@pytest.mark.parametrize("NGT", [NGTTransformer, LegacyNNG])
def test_is_valid_sklearn_estimator_in_persistent_memory(NGT):
    check_estimator(NGT())


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
@pytest.mark.parametrize("n_candidates", [1, 2, 5, 99, 100, 1000, ])
@pytest.mark.parametrize("set_in_constructor", [True, False])
@pytest.mark.parametrize("return_distance", [True, False])
@pytest.mark.parametrize("search_among_indexed", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
def test_return_correct_number_of_neighbors(n_candidates: int,
                                            set_in_constructor: bool,
                                            return_distance: bool,
                                            search_among_indexed: bool,
                                            verbose: bool):
    n_samples = 100
    X, y = make_classification(n_samples=n_samples)
    ann = LegacyNNG(n_candidates=n_candidates, verbose=verbose)\
        if set_in_constructor else LegacyNNG(verbose=verbose)
    ann.fit(X, y)
    X_query = None if search_among_indexed else X
    neigh = ann.kneighbors(X_query, return_distance=return_distance) if set_in_constructor\
        else ann.kneighbors(X_query, n_candidates=n_candidates, return_distance=return_distance)

    if return_distance:
        dist, neigh = neigh
        assert dist.shape == neigh.shape, "Shape of distances and indices matrices do not match."
        if n_candidates > n_samples:
            assert np.all(np.isnan(dist[:, n_samples:])), "Returned distances for invalid neighbors"

    assert neigh.shape[1] == n_candidates, "Wrong number of neighbors returned."
    if n_candidates > n_samples:
        assert np.all(neigh[:, n_samples:] == -1), "Returned indices for invalid neighbors"


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
@pytest.mark.parametrize("NGT", [NGTTransformer, LegacyNNG])
@pytest.mark.parametrize("metric", ["invalid", None])
def test_invalid_metric(metric, NGT):
    X, y = make_classification(n_samples=10, n_features=10)
    ann = NGT(metric=metric)
    with assert_raises(ValueError):
        _ = ann.fit(X, y)


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
@pytest.mark.parametrize("metric", LegacyNNG.valid_metrics)
@pytest.mark.parametrize("n_jobs", [-1, 1, None])
@pytest.mark.parametrize("verbose", [0, 1])
def test_kneighbors_with_or_without_distances(metric, n_jobs, verbose):
    n_samples = 100
    X = np.random.RandomState(1235232).rand(n_samples, 2)
    ann = LegacyNNG(metric=metric,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    )
    ann.fit(X)
    neigh_dist_self, neigh_ind_self = ann.kneighbors(X, return_distance=True)
    ind_only_self = ann.kneighbors(X, return_distance=False)

    # Identical neighbors retrieved, whether dist or not
    assert_array_equal(neigh_ind_self, ind_only_self)

    # Is the first hit always the object itself?
    # Less strict test for inaccurate distances
    if metric in ["Hamming", "Jaccard", "Normalized Cosine", "Normalized Angle"]:
        assert np.intersect1d(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self))).size >= 75
    else:
        assert_array_equal(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self)))

    if metric in ["Hamming", "Jaccard"]:  # quite inaccurate...
        assert neigh_dist_self[:, 0].mean() <= 0.016
    elif metric in ["Normalized Angle"]:
        assert_array_almost_equal(neigh_dist_self[:, 0], np.zeros(len(neigh_dist_self)), decimal=3)
    else:  # distances in [0, inf]
        assert_array_almost_equal(neigh_dist_self[:, 0], np.zeros(len(neigh_dist_self)))


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
@pytest.mark.parametrize("metric", LegacyNNG.valid_metrics)
def test_kneighbors_with_or_without_self_hit(metric):
    X = np.random.RandomState(1245544).rand(50, 2)
    n_candidates = 5
    ann = LegacyNNG(metric=metric,
                    n_candidates=n_candidates,
                    )
    ann.fit(X)
    ind_self = ann.kneighbors(X, n_candidates=n_candidates+1, return_distance=False)
    ind_no_self = ann.kneighbors(n_candidates=n_candidates, return_distance=False)

    if metric in ["Hamming", "Jaccard"]:  # just inaccurate...
        assert (ind_self[:, 0] == np.arange(len(ind_self))).sum() >= 46
        assert np.setdiff1d(ind_self[:, 1:], ind_no_self).size <= 10
    else:
        assert_array_equal(ind_self[:, 0], np.arange(len(ind_self)))
        assert_array_equal(ind_self[:, 1:], ind_no_self)


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
def test_squared_euclidean_same_neighbors_as_euclidean():
    X, y = make_classification()
    ann = LegacyNNG(metric="euclidean")
    ann.fit(X, y)
    neigh_dist_eucl, neigh_ind_eucl = ann.kneighbors(X)

    ann = LegacyNNG(metric="sqeuclidean")
    ann.fit(X, y)
    neigh_dist_sqeucl, neigh_ind_sqeucl = ann.kneighbors(X)

    assert_array_equal(neigh_ind_eucl, neigh_ind_sqeucl)
    assert_array_almost_equal(neigh_dist_eucl ** 2, neigh_dist_sqeucl)


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
def test_same_neighbors_as_with_exact_nn_search():
    X = np.random.RandomState(42).randn(10, 2)

    nn = NearestNeighbors()
    nn_dist, nn_neigh = nn.fit(X).kneighbors(return_distance=True)

    ann = LegacyNNG()
    ann_dist, ann_neigh = ann.fit(X).kneighbors(return_distance=True)

    assert_array_almost_equal(ann_dist, nn_dist, decimal=5)
    assert_array_almost_equal(ann_neigh, nn_neigh, decimal=0)


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
@pytest.mark.xfail(reason="ngtpy.Index can not be pickled as of v1.12.2")
@pytest.mark.parametrize("NGT", [LegacyNNG, NGTTransformer])
def test_is_valid_estimator_in_main_memory(NGT):
    arg = "index_dir" if issubclass(NGT, LegacyNNG) else "mmap_dir"
    check_estimator(NGT(**{arg: None}))


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
@pytest.mark.parametrize("NGT", [LegacyNNG, NGTTransformer])
@pytest.mark.parametrize("dir_", [tuple(), 0, "auto", "/dev/shm", "/tmp", None])
def test_memory_mapped(dir_, NGT):
    n_samples: int = 10
    # Note that we expect one more neighbor downstream, b/c of KNeighborsTransformer convention
    n_neighbors: int = 6
    X, y = make_classification(n_samples=n_samples,
                               n_features=5,
                               random_state=123,
                               )
    if issubclass(NGT, LegacyNNG):
        kwargs = {
            "index_dir": dir_,
            "n_candidates": n_neighbors + 1,
        }
    else:
        kwargs = {
            "mmap_dir": dir_,
            "n_neighbors": n_neighbors,
        }
    ann = NGT(**kwargs)
    if isinstance(dir_, str) or dir_ is None:
        ann.fit(X, y)
        if issubclass(NGT, LegacyNNG):
            neigh_dist, neigh_ind = ann.kneighbors(X)
            # Look for the self-neighbors
            np.testing.assert_array_equal(neigh_dist[:, 0], 0)
            np.testing.assert_array_equal(neigh_ind[:, 0], np.arange(len(neigh_ind)))
            # Check there are no self-neighbors
            neigh_dist, neigh_ind = ann.kneighbors()
            np.testing.assert_array_less(0, neigh_dist)
            assert not np.any(neigh_ind[:, 0] == np.arange(len(neigh_ind)))
        else:
            graph = ann.transform(X)
            assert sparse.issparse(graph)
            assert graph.shape == (n_samples, n_samples)
            assert graph.nnz == (n_neighbors + 1) * n_samples
            np.testing.assert_array_equal(graph.diagonal(), 0)
    else:
        with np.testing.assert_raises(TypeError):
            ann.fit(X, y)


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows.")
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_transformer_vs_legacy_ngt(metric):
    X, y = make_classification(random_state=123)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    n_neighbors_transformer = 5
    n_neighbors_legacy = n_neighbors_transformer + 1
    ngt_legacy = LegacyNNG(metric=metric, n_candidates=n_neighbors_legacy)
    ngt_legacy.fit(X_train, y_train)
    neigh_dist, neigh_ind = ngt_legacy.kneighbors(X_test, return_distance=True)

    ngt_trafo = NGTTransformer(metric=metric, n_neighbors=n_neighbors_transformer)
    ngt_trafo.fit(X_train, y_train)
    graph = ngt_trafo.transform(X_test)

    # Check that both old and new NGT wrappers yield identical nearest neighbors and distances
    np.testing.assert_array_equal(neigh_ind.ravel(), graph.indices.ravel())
    np.testing.assert_array_almost_equal(neigh_dist.ravel(), graph.data.ravel())


@pytest.mark.skipif(sys.platform == "win32", reason="NGT not supported on Windows")
@pytest.mark.skip(reason="PERFORMANCE. This takes >1min. Enable again with better default params...")
def test_nng_optimization():
    n_samples = 150
    n_neighbors = 10
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        random_state=123,
    )
    ann = NGTTransformer(
        n_neighbors=n_neighbors,
        mmap_dir="/dev/shm",
        optimize=True,
        edge_size_for_search=40,
        edge_size_for_creation=10,
        num_incoming=2,
        num_outgoing=2,
        epsilon=0.1,
        n_jobs=2,
    )
    ann.fit(X, y)
    graph = ann.transform(X)
    assert sparse.issparse(graph)
    assert graph.shape == (n_samples, n_samples)
    assert graph.nnz == n_neighbors * n_samples
    np.testing.assert_array_equal(graph.diagonal(), 0)
