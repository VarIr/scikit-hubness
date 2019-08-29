# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from skhubness.reduction import DisSimLocal
from skhubness.neighbors import NearestNeighbors, KNeighborsClassifier


def test_squared():
    X, y = make_classification()
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    hr_squared = DisSimLocal(k=5, squared=True)
    hr = DisSimLocal(k=5, squared=False)

    dist_squared, _ = hr_squared.fit_transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    dist, _ = hr.fit_transform(neigh_dist, neigh_ind, X, assume_sorted=True)

    assert_array_almost_equal(dist_squared, dist ** 2)


@pytest.mark.parametrize('squared', [True, False])
@pytest.mark.parametrize('k', [1, 5, 10])
def test_fit_sorted_and_fit_transform(squared, k):
    X, y = make_classification()
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    hr = DisSimLocal(k=k, squared=squared)

    nd_sorted, ni_sorted = hr.fit(neigh_dist, neigh_ind, X, assume_sorted=True)\
                             .transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    nd_unsort, ni_unsort = hr.fit(neigh_dist, neigh_ind, X, assume_sorted=False)\
                             .transform(neigh_dist, neigh_ind, X, assume_sorted=False)

    assert_array_equal(ni_sorted, ni_unsort)
    assert_array_almost_equal(nd_sorted, nd_unsort)

    nd_sorted_fit_tr, ni_sorted_fit_tr = hr.fit_transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    nd_unsorted_fit_tr, ni_unsorted_fit_tr = hr.fit_transform(neigh_dist, neigh_ind, X, assume_sorted=False)

    assert_array_almost_equal(nd_sorted, nd_sorted_fit_tr)
    assert_array_equal(ni_sorted, ni_sorted_fit_tr)
    assert_array_almost_equal(nd_unsort, nd_unsorted_fit_tr)
    assert_array_equal(ni_unsort, ni_unsorted_fit_tr)


@pytest.mark.parametrize('k', ['invalid', None, -1, 0])
def test_invalid_k(k):
    X, y = make_classification(n_samples=10, )
    nn = NearestNeighbors()
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    hr = DisSimLocal(k=k)
    with assert_raises(Exception):
        hr.fit(neigh_dist, neigh_ind, X, assume_sorted=True)


@pytest.mark.parametrize('k', [9, 10, 11])
def test_warning_on_too_large_k(k, n_samples=10):
    X, y = make_classification(n_samples=n_samples)
    nn = NearestNeighbors(n_neighbors=n_samples-1)
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    hr = DisSimLocal(k=k)
    if k < n_samples:
        hr.fit(neigh_dist, neigh_ind, X, assume_sorted=True)
        _ = hr.transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    else:
        with pytest.warns(Warning):
            hr.fit(neigh_dist, neigh_ind, X, assume_sorted=True)
        with pytest.warns(Warning):
            _ = hr.transform(neigh_dist, neigh_ind, X, assume_sorted=True)


@pytest.mark.parametrize('k', [9, 10, 11])
def test_warning_on_too_few_neighbors(k, n_samples=10):
    X, y = make_classification(n_samples=n_samples)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    hr = DisSimLocal(k=k)
    with pytest.warns(Warning):
        hr.fit(neigh_dist, neigh_ind, X, assume_sorted=True)
    with pytest.warns(Warning):
        _ = hr.transform(neigh_dist, neigh_ind, X, assume_sorted=True)


def test_dsl_knn_with_various_metrics():
    X, y = make_classification()
    algorithm_params = {'n_candidates': X.shape[0]-1}
    knn = KNeighborsClassifier(hubness='dsl', metric='euclidean', algorithm_params=algorithm_params)
    knn.fit(X, y)
    y_pred_eucl = knn.predict(X)
    knn = KNeighborsClassifier(hubness='dsl', metric='sqeuclidean', algorithm_params=algorithm_params)
    knn.fit(X, y)
    y_pred_sqeucl = knn.predict(X)
    knn = KNeighborsClassifier(hubness='dsl', metric='manhattan', algorithm_params=algorithm_params)
    with pytest.warns(UserWarning):
        knn.fit(X, y)
    y_pred_other = knn.predict(X)

    assert_array_equal(y_pred_eucl, y_pred_sqeucl)
    assert_array_equal(y_pred_sqeucl, y_pred_other)
