#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import euclidean_distances
from sklearn.utils.estimator_checks import check_estimator


from skhubness import Hubness

DIST = squareform(np.array([.2, .1, .8, .4, .3, .5, .7, 1., .6, .9]))


def test_estimator():
    check_estimator(Hubness)


@pytest.mark.parametrize('verbose', [-1, 0, 1, 2, None])
def test_hubness(verbose):
    """Test hubness against ground truth calc on spreadsheet"""
    HUBNESS_TRUE = -0.2561204163  # Hubness truth: S_k=5, skewness calculated with bias
    hub = Hubness(k=2, metric='precomputed', verbose=verbose)
    hub.fit(DIST)
    Sk2 = hub.score(has_self_distances=True)
    np.testing.assert_almost_equal(Sk2, HUBNESS_TRUE, decimal=10)


def test_all_params_none():
    X, _ = make_classification()
    hub = Hubness(k=None, return_value=None, hub_size=None, metric=None,
                  store_k_neighbors=None, store_k_occurrence=None,
                  algorithm_params=None,
                  hubness=None, hubness_params=None,
                  verbose=None, n_jobs=None, random_state=None,
                  shuffle_equal=None)
    hub.fit(X)
    _ = hub.score()


@pytest.mark.parametrize('store_k_neighbors', [True, False])
def test_return_k_neighbors(store_k_neighbors):
    X, _ = make_classification()
    hub = Hubness(return_value='k_neighbors', store_k_neighbors=store_k_neighbors)
    if store_k_neighbors:
        hub.fit(X)
    else:
        with pytest.warns(UserWarning):
            hub.fit(X)
    _ = hub.score()


@pytest.mark.parametrize('store_k_occurrence', [True, False])
def test_return_k_occurrence(store_k_occurrence):
    X, _ = make_classification()
    hub = Hubness(return_value='k_occurrence', store_k_occurrence=store_k_occurrence)
    if store_k_occurrence:
        hub.fit(X)
    else:
        with pytest.warns(UserWarning):
            hub.fit(X)
    _ = hub.score()


def test_limiting_factor():
    X, _ = make_classification()
    hub = Hubness(store_k_occurrence=True, return_value='k_occurrence')
    hub.fit(X)
    k_occ = hub.score()

    gini_space = hub._calc_gini_index(k_occ, limiting='space')
    gini_time = hub._calc_gini_index(k_occ, limiting='time')
    gini_naive = hub._calc_gini_index(k_occ, limiting=None)

    assert gini_space == gini_time == gini_naive


@pytest.mark.parametrize('verbose', [True, False])
def test_shuffle_equal(verbose):
    # for this data set there shouldn't be any equal distances,
    # and shuffle should make no difference
    X, _ = make_classification(random_state=12354)
    dist = euclidean_distances(X)
    skew_shuffle, skew_no_shuffle = \
        [Hubness(metric='precomputed', shuffle_equal=v, verbose=verbose)
         .fit(dist).score() for v in [True, False]]
    assert skew_no_shuffle == skew_shuffle


@pytest.mark.parametrize('verbose', [True, False])
@pytest.mark.parametrize('shuffle_equal', [True, False])
def test_sparse_equal_dense(verbose, shuffle_equal):
    X, _ = make_classification()
    dist_dense = euclidean_distances(X)
    dist_sparse = csr_matrix(dist_dense)

    hub = Hubness(metric='precomputed',
                  shuffle_equal=shuffle_equal,
                  verbose=verbose)
    hub.fit(dist_dense)
    skew_dense = hub.score(has_self_distances=True)

    hub.fit(dist_sparse)
    skew_sparse = hub.score(has_self_distances=True)

    np.testing.assert_almost_equal(skew_dense, skew_sparse)


@pytest.mark.parametrize('shuffle_equal', [True, False])
def test_sparse_equal_dense_if_variable_hits_per_row(shuffle_equal):
    X, _ = make_classification(random_state=123)
    dist = euclidean_distances(X)
    dist[0, 1:3] = 999
    dist[1:3, 0] = 999
    dist[1, 1:5] = 999
    dist[1:5, 1] = 999
    sparse = dist.copy()
    sparse[0, 1:3] = 0
    sparse[1:3, 0] = 0
    sparse[1, 1:5] = 0
    sparse[1:5, 1] = 0
    sparse = csr_matrix(sparse)

    hub = Hubness(metric='precomputed',
                  shuffle_equal=shuffle_equal,
                  random_state=123)
    hub.fit(dist)
    skew_dense = hub.score(has_self_distances=True)

    hub = Hubness(metric='precomputed',
                  shuffle_equal=shuffle_equal,
                  random_state=123)
    hub.fit(sparse)
    skew_sparse = hub.score(has_self_distances=True)

    np.testing.assert_almost_equal(skew_dense, skew_sparse, decimal=2)


def test_atkinson():
    X, _ = make_classification(random_state=123)
    hub = Hubness(return_value='k_occurrence').fit(X)
    k_occ = hub.score()

    atkinson_0999 = hub._calc_atkinson_index(k_occ, eps=.999)
    atkinson_1000 = hub._calc_atkinson_index(k_occ, eps=1)

    np.testing.assert_almost_equal(atkinson_0999, atkinson_1000, decimal=3)


@pytest.mark.parametrize('seed', [0, 626])
@pytest.mark.parametrize('n_samples', [20, 200])
@pytest.mark.parametrize('n_features', [2, 500])
@pytest.mark.parametrize('k', [1, 5, 10])
def test_hubness_return_values_are_self_consistent(n_samples, n_features, k, seed):
    """Test that the three returned values fit together"""
    np.random.seed(seed)
    vectors = 99. * (np.random.rand(n_samples, n_features) - 0.5)
    k = 10
    hub = Hubness(k=k, metric='euclidean', store_k_neighbors=True, store_k_occurrence=True)
    hub.fit(vectors)
    skew = hub.score()
    neigh = hub.k_neighbors
    occ = hub.k_occurrence
    # Neighbors are just checked for correct shape
    assert neigh.shape == (n_samples, k)
    # Count k-occurrence (different method than in module)
    neigh = neigh.ravel()
    occ_true = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        occ_true[i] = (neigh == i).sum()
    np.testing.assert_array_equal(occ, occ_true)
    # Calculate skewness (different method than in module)
    x0 = occ - occ.mean()
    s2 = (x0 ** 2).mean()
    m3 = (x0 ** 3).mean()
    skew_true = m3 / (s2 ** 1.5)
    np.testing.assert_equal(skew, skew_true)


@pytest.mark.parametrize('dist', [DIST])
@pytest.mark.parametrize('n_jobs', [-1, 1, 2, 4])
@pytest.mark.filterwarnings('ignore:divide by zero')
@pytest.mark.filterwarnings('ignore:invalid value encountered')
def test_parallel_hubness_equal_serial_hubness_distance_based(dist, n_jobs):
    # Parallel
    hub = Hubness(k=4, metric='precomputed', store_k_occurrence=True, store_k_neighbors=True, n_jobs=n_jobs)
    hub.fit(dist)
    skew_p = hub.score(has_self_distances=True)
    neigh_p = hub.k_neighbors
    occ_p = hub.k_occurrence

    # Sequential
    hub = Hubness(k=4, metric='precomputed', store_k_occurrence=True, store_k_neighbors=True, n_jobs=1)
    hub.fit(dist)
    skew_s = hub.score(has_self_distances=True)
    neigh_s = hub.k_neighbors
    occ_s = hub.k_occurrence

    np.testing.assert_array_almost_equal(skew_p, skew_s, decimal=7)
    np.testing.assert_array_almost_equal(neigh_p, neigh_s, decimal=7)
    np.testing.assert_array_almost_equal(occ_p, occ_s, decimal=7)


@pytest.mark.parametrize('has_self_distances', [True, False])
def test_hubness_against_distance(has_self_distances):
    """Test hubness class against distance-based methods."""

    np.random.seed(123)
    X = np.random.rand(100, 50)
    D = euclidean_distances(X)
    verbose = 1

    hub = Hubness(k=10, metric='precomputed',
                  store_k_occurrence=True,
                  store_k_neighbors=True,
                  )
    hub.fit(D)
    skew_d = hub.score(has_self_distances=has_self_distances)
    neigh_d = hub.k_neighbors
    occ_d = hub.k_occurrence

    hub = Hubness(k=10, metric='euclidean',
                  store_k_neighbors=True,
                  store_k_occurrence=True,
                  verbose=verbose)
    hub.fit(X)
    skew_v = hub.score(X if not has_self_distances else None)
    neigh_v = hub.k_neighbors
    occ_v = hub.k_occurrence

    np.testing.assert_allclose(skew_d, skew_v, atol=1e-7)
    np.testing.assert_array_equal(neigh_d, neigh_v)
    np.testing.assert_array_equal(occ_d, occ_v)


@pytest.mark.parametrize('hubness_measure', ['k_skewness', 'k_skewness_truncnorm', 'robinhood', 'gini', 'atkinson'])
def test_hubness_independent_on_data_set_size(hubness_measure):
    """ New measures should pass, traditional skewness should fail. """
    thousands = 3
    n_objects = thousands * 1_000
    rng = np.random.RandomState(1247)
    X = rng.rand(n_objects, 128)
    N_SAMPLES_LIST = np.arange(1, thousands + 1) * 1_000
    value = np.empty(N_SAMPLES_LIST.size)
    for i, n_samples in enumerate(N_SAMPLES_LIST):
        ind = rng.permutation(n_objects)[:n_samples]
        X_sample = X[ind, :]
        hub = Hubness(return_value='all')
        hub.fit(X_sample)
        measures = hub.score()
        if hubness_measure == 'k_skewness':
            value[i] = hub.k_skewness
        elif hubness_measure == 'k_skewness_truncnorm':
            value[i] = hub.k_skewness_truncnorm
        elif hubness_measure == 'robinhood':
            value[i] = hub.robinhood_index
        elif hubness_measure == 'gini':
            value[i] = hub.gini_index
        elif hubness_measure == 'atkinson':
            value[i] = hub.atkinson_index
        assert value[i] == measures[hubness_measure]
        if i > 0:
            if hubness_measure == 'k_skewness':
                with np.testing.assert_raises(AssertionError):
                    np.testing.assert_allclose(value[i], value[i-1], rtol=0.1)
            else:
                np.testing.assert_allclose(
                    value[i], value[i - 1], rtol=2e-1,
                    err_msg=(f'Hubness measure is too dependent on data set '
                             f'size with S({N_SAMPLES_LIST[i]}) = x '
                             f'and S({N_SAMPLES_LIST[i-1]}) = y.'))
    if hubness_measure == 'k_skewness':
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(value[-1], value[0], rtol=0.1)
    else:
        np.testing.assert_allclose(value[-1], value[0], rtol=2e-1)


# def test_hubness_from_sparse_precomputed_matrix(self):
#     # Generate high-dimensional data
#     X, y = make_classification(n_samples=1000,
#                                n_features=100,
#                                n_informative=100,
#                                n_redundant=0,
#                                n_repeated=0,
#                                random_state=123)
#     X = X.astype(np.float32)
#     y = y.astype(np.int32)
#     for hr_algorithm in ['mpg', 'ls', 'dsl']:
#         for sampling_algorithm in ['hnsw', 'lsh']:  # ['hnsw', 'lsh']:#
#             for n_samples in [50, 100]:
#                 print(f'Test {hr_algorithm}, {sampling_algorithm}, '
#                       f'with {n_samples} samples.')
#                 self.hubness_from_sparse_precomputed_matrix(
#                     X, y, hr_algorithm, sampling_algorithm, n_samples)


# def hubness_from_sparse_precomputed_matrix(self, X, y, hr,
#                                            sample, n_samples):
#     # Make train-test split
#     X_train, X_test, y_train, _ = train_test_split(X, y)
#     # Obtain a sparse distance matrix
#     ahr = ApproximateHubnessReduction(
#         hr_algorithm=hr, sampling_algorithm=sample, n_samples=n_samples)
#     ahr.fit(X_train, y_train)
#     _ = ahr.transform(X_test)
#     D_test_csr = ahr.sec_dist_sparse_
#     # Hubness in sparse matrix
#     hub = Hubness(k=10,
#                   metric='precomputed',
#                   return_k_neighbors=True,
#                   shuffle_equal=False,
#                   verbose=self.verbose)
#     hub.score(D_test_csr)
#     Sk_trunc_sparse = hub.k_skewness_truncnorm_
#     Sk_sparse = hub.k_skewness_
#     k_neigh_sparse = hub.k_neighbors_
#     # Hubness in dense matrix
#     try:
#         D_test_dense = D_test_csr.toarray()
#     except AttributeError:
#         return  # Without sampling, the distance matrix is not sparse
#     D_test_dense[D_test_dense == 0] = np.finfo(np.float32).max
#     hub_dense = Hubness(k=10,
#                         metric='precomputed',
#                         return_k_neighbors=True,
#                         shuffle_equal=False)
#     hub_dense.score(D_test_dense)
#     Sk_trunc_dense = hub_dense.k_skewness_truncnorm_
#     Sk_dense = hub_dense.k_skewness_
#     k_neigh_dense = hub_dense.k_neighbors_
#     if hr in ['MP', 'MPG']:
#         decimal = 1
#     else:
#         decimal = 5
#     try:
#         np.testing.assert_array_equal(
#             k_neigh_dense.ravel(), k_neigh_sparse)
#     except AssertionError:
#         s1 = k_neigh_dense.sum()
#         s2 = k_neigh_sparse.sum()
#         sm = max(s1, s2)
#         print(f'k_neighbors not identical, but close: '
#               f'{s1}, {s2}, {s1/s2}.')
#         np.testing.assert_allclose(s2 / sm, s1 / sm, rtol=1e-2)
#     np.testing.assert_array_almost_equal(
#         Sk_sparse, Sk_dense, decimal=decimal)
#     np.testing.assert_array_almost_equal(
#         Sk_trunc_sparse, Sk_trunc_dense, decimal=decimal)
