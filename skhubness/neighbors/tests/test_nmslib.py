# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
from scipy.sparse import csr_matrix, random as sparse_random
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils._testing import assert_raises
from sklearn.utils.estimator_checks import check_estimator
from skhubness.neighbors import LegacyHNSW, NMSlibTransformer


def matrix_to_str_array(sparse_matrix: csr_matrix):
    """ Helper function to test str/object NMSlib datatypes"""
    res = []
    indptr = sparse_matrix.indptr
    indices = sparse_matrix.indices
    for row in range(sparse_matrix.shape[0]):
        arr = [k for k in indices[indptr[row]: indptr[row + 1]]]
        arr.sort()
        res.append(' '.join([str(k) for k in arr]))
    return res


def test_sklearn_estimator():
    # Note that LegacyHNSW is known to fail check_estimator, and always will
    check_estimator(NMSlibTransformer())


@pytest.mark.parametrize('metric', ['invalid', None])
@pytest.mark.parametrize("HubnessReduction", [LegacyHNSW, NMSlibTransformer])
def test_invalid_metric(metric, HubnessReduction):
    X, y = make_classification(n_samples=10, n_features=10)
    hnsw = HubnessReduction(metric=metric)
    with assert_raises(ValueError):
        _ = hnsw.fit(X, y)


def test_fail_kneighbors_without_data():
    X, y = make_classification(n_samples=10, n_features=10)
    hnsw = LegacyHNSW()
    hnsw.fit(X, y)
    with assert_raises(NotImplementedError):
        hnsw.kneighbors()


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
@pytest.mark.parametrize('n_jobs', [-1, 1, None])
@pytest.mark.parametrize('verbose', [0, 1])
def test_kneighbors_with_or_without_self_hit(metric, n_jobs, verbose):
    X, y = make_classification()
    hnsw = LegacyHNSW(metric=metric, n_jobs=n_jobs, verbose=verbose)
    hnsw.fit(X, y)
    neigh_dist_self, neigh_ind_self = hnsw.kneighbors(X, return_distance=True)
    ind_only_self = hnsw.kneighbors(X, return_distance=False)

    assert_array_equal(neigh_ind_self, ind_only_self)
    assert_array_equal(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self)))

    # distances in [0, inf]
    assert_array_almost_equal(neigh_dist_self[:, 0], np.zeros(len(neigh_dist_self)))


def test_squared_euclidean_same_neighbors_as_euclidean():
    X, y = make_classification()
    hnsw = LegacyHNSW(metric='minkowski')
    hnsw.fit(X, y)
    neigh_dist_eucl, neigh_ind_eucl = hnsw.kneighbors(X)

    hnsw = LegacyHNSW(metric='sqeuclidean')
    hnsw.fit(X, y)
    neigh_dist_sqeucl, neigh_ind_sqeucl = hnsw.kneighbors(X)

    assert_array_equal(neigh_ind_eucl, neigh_ind_sqeucl)
    assert_array_almost_equal(neigh_dist_eucl ** 2, neigh_dist_sqeucl)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_transformer_vs_legacy_hnsw(metric):
    X, y = make_classification(random_state=123)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    hnsw = LegacyHNSW(metric=metric)
    hnsw.fit(X_train, y_train)
    hnsw_neigh_dist, hnsw_neigh_ind = hnsw.kneighbors(X_test, return_distance=True)

    nms = NMSlibTransformer(metric=metric)
    nms.fit(X_train, y_train)
    nms_graph = nms.transform(X_test)

    # Check that both old and new HNSW wrapper yield identical nearest neighbors and distances
    np.testing.assert_array_equal(hnsw_neigh_ind.ravel(), nms_graph.indices.ravel())
    np.testing.assert_array_almost_equal(hnsw_neigh_dist.ravel(), nms_graph.data.ravel())


@pytest.mark.parametrize("metric", NMSlibTransformer.valid_metrics)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.uint8, np.int32, np.int64])
def test_all_metrics(metric, dtype):
    sparse = False
    if "_sparse" in metric:
        sparse = True
    kwargs = {}
    if metric.startswith("lp"):
        kwargs.update({"p": 1.5})
    # This will only be required once string-based metrics are enabled
    convert_to_str = False

    n, m = (20, 100)
    if sparse:
        X = sparse_random(n, m, density=0.01, format="csr", dtype=dtype, random_state=123)
        if convert_to_str:
            X = matrix_to_str_array(X)
    else:
        X = np.random.rand(n, m)
        X *= 10.
        X = X.astype(dtype)
    nms = NMSlibTransformer(metric=metric, **kwargs)
    graph = nms.fit_transform(X)
    assert graph.shape == (20, 20)
    # scipy.sparse does not support float16
    if not sparse:
        assert graph.dtype == dtype
