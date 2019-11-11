# SPDX-License-Identifier: BSD-3-Clause
import sys
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils.testing import assert_raises
from skhubness.neighbors import NNG, NearestNeighbors


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
@pytest.mark.parametrize('n_candidates', [1, 2, 5, 99, 100, 1000, ])
@pytest.mark.parametrize('set_in_constructor', [True, False])
@pytest.mark.parametrize('return_distance', [True, False])
@pytest.mark.parametrize('search_among_indexed', [True, False])
@pytest.mark.parametrize('verbose', [True, False])
def test_return_correct_number_of_neighbors(n_candidates: int,
                                            set_in_constructor: bool,
                                            return_distance: bool,
                                            search_among_indexed: bool,
                                            verbose: bool):
    n_samples = 100
    X, y = make_classification(n_samples=n_samples)
    ann = NNG(n_candidates=n_candidates, verbose=verbose)\
        if set_in_constructor else NNG(verbose=verbose)
    ann.fit(X, y)
    X_query = None if search_among_indexed else X
    neigh = ann.kneighbors(X_query, return_distance=return_distance) if set_in_constructor\
        else ann.kneighbors(X_query, n_candidates=n_candidates, return_distance=return_distance)

    if return_distance:
        dist, neigh = neigh
        assert dist.shape == neigh.shape, f'Shape of distances and indices matrices do not match.'
        if n_candidates > n_samples:
            assert np.all(np.isnan(dist[:, n_samples:])), f'Returned distances for invalid neighbors'

    assert neigh.shape[1] == n_candidates, f'Wrong number of neighbors returned.'
    if n_candidates > n_samples:
        assert np.all(neigh[:, n_samples:] == -1), f'Returned indices for invalid neighbors'


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
@pytest.mark.parametrize('metric', ['invalid', None])
def test_invalid_metric(metric):
    X, y = make_classification(n_samples=10, n_features=10)
    ann = NNG(metric=metric)
    with assert_raises(ValueError):
        _ = ann.fit(X, y)


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
@pytest.mark.parametrize('metric', NNG.valid_metrics)
@pytest.mark.parametrize('n_jobs', [-1, 1, None])
@pytest.mark.parametrize('verbose', [0, 1])
def test_kneighbors_with_or_without_distances(metric, n_jobs, verbose):
    n_samples = 100
    X = np.random.RandomState(1235232).rand(n_samples, 2)
    ann = NNG(metric=metric,
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
    if metric in ['Hamming', 'Jaccard', 'Normalized Cosine', 'Normalized Angle']:
        assert np.intersect1d(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self))).size >= 75
    else:
        assert_array_equal(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self)))

    if metric in ['Hamming', 'Jaccard']:  # quite inaccurate...
        assert neigh_dist_self[:, 0].mean() <= 0.016
    elif metric in ['Normalized Angle']:
        assert_array_almost_equal(neigh_dist_self[:, 0], np.zeros(len(neigh_dist_self)), decimal=3)
    else:  # distances in [0, inf]
        assert_array_almost_equal(neigh_dist_self[:, 0], np.zeros(len(neigh_dist_self)))


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
@pytest.mark.parametrize('metric', NNG.valid_metrics)
def test_kneighbors_with_or_without_self_hit(metric):
    X = np.random.RandomState(1245544).rand(50, 2)
    n_candidates = 5
    ann = NNG(metric=metric,
              n_candidates=n_candidates,
              )
    ann.fit(X)
    ind_self = ann.kneighbors(X, n_candidates=n_candidates+1, return_distance=False)
    ind_no_self = ann.kneighbors(n_candidates=n_candidates, return_distance=False)

    if metric in ['Hamming', 'Jaccard']:  # just inaccurate...
        assert (ind_self[:, 0] == np.arange(len(ind_self))).sum() >= 46
        assert np.setdiff1d(ind_self[:, 1:], ind_no_self).size <= 10
    else:
        assert_array_equal(ind_self[:, 0], np.arange(len(ind_self)))
        assert_array_equal(ind_self[:, 1:], ind_no_self)


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
def test_squared_euclidean_same_neighbors_as_euclidean():
    X, y = make_classification()
    ann = NNG(metric='euclidean')
    ann.fit(X, y)
    neigh_dist_eucl, neigh_ind_eucl = ann.kneighbors(X)

    ann = NNG(metric='sqeuclidean')
    ann.fit(X, y)
    neigh_dist_sqeucl, neigh_ind_sqeucl = ann.kneighbors(X)

    assert_array_equal(neigh_ind_eucl, neigh_ind_sqeucl)
    assert_array_almost_equal(neigh_dist_eucl ** 2, neigh_dist_sqeucl)


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
def test_same_neighbors_as_with_exact_nn_search():
    X = np.random.RandomState(42).randn(10, 2)

    nn = NearestNeighbors()
    nn_dist, nn_neigh = nn.fit(X).kneighbors(return_distance=True)

    ann = NNG()
    ann_dist, ann_neigh = ann.fit(X).kneighbors(return_distance=True)

    assert_array_almost_equal(ann_dist, nn_dist, decimal=5)
    assert_array_almost_equal(ann_neigh, nn_neigh, decimal=0)


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
def test_is_valid_estimator_in_persistent_memory():
    check_estimator(NNG)


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
@pytest.mark.xfail(reason='ngtpy.Index can not be pickled as of v1.7.6')
def test_is_valid_estimator_in_main_memory():
    check_estimator(NNG(index_dir=None))


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows.')
@pytest.mark.parametrize('index_dir', [tuple(), 0, 'auto', '/dev/shm', '/tmp', None])
def test_memory_mapped(index_dir):
    X, y = make_classification(n_samples=10,
                               n_features=5,
                               random_state=123,
                               )
    ann = NNG(index_dir=index_dir)
    if isinstance(index_dir, str) or index_dir is None:
        ann.fit(X, y)
        _ = ann.kneighbors(X)
        _ = ann.kneighbors()
    else:
        with np.testing.assert_raises(TypeError):
            ann.fit(X, y)


@pytest.mark.skipif(sys.platform == 'win32', reason='NGT not supported on Windows')
def test_nng_optimization():
    X, y = make_classification(n_samples=150,
                               n_features=2,
                               n_redundant=0,
                               random_state=123,
                               )
    ann = NNG(index_dir='/dev/shm',
              optimize=True,
              edge_size_for_search=40,
              edge_size_for_creation=10,
              epsilon=0.1,
              n_jobs=2,
              )
    ann.fit(X, y)
    _ = ann.kneighbors(X, )
    _ = ann.kneighbors()
