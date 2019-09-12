# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils.testing import assert_raises
from skhubness.neighbors import RandomProjectionTree, NearestNeighbors


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
    ann = RandomProjectionTree(n_candidates=n_candidates, verbose=verbose)\
        if set_in_constructor else RandomProjectionTree(verbose=verbose)
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


@pytest.mark.parametrize('metric', ['invalid', None])
def test_invalid_metric(metric):
    X, y = make_classification(n_samples=10, n_features=10)
    ann = RandomProjectionTree(metric=metric)
    with assert_raises(Exception):  # annoy raises ValueError or TypeError
        _ = ann.fit(X, y)


@pytest.mark.parametrize('metric', RandomProjectionTree.valid_metrics)
@pytest.mark.parametrize('n_jobs', [-1, 1, None])
@pytest.mark.parametrize('verbose', [0, 1])
def test_kneighbors_with_or_without_distances(metric, n_jobs, verbose):
    n_samples = 100
    X, y = make_classification(n_samples=n_samples,
                               random_state=123,
                               )
    ann = RandomProjectionTree(metric=metric,
                               n_jobs=n_jobs,
                               verbose=verbose,
                               )
    ann.fit(X, y)
    neigh_dist_self, neigh_ind_self = ann.kneighbors(X, return_distance=True)
    ind_only_self = ann.kneighbors(X, return_distance=False)

    # Identical neighbors retrieved, whether dist or not
    assert_array_equal(neigh_ind_self, ind_only_self)

    # Is the first hit always the object itself?
    # Less strict test for dot/hamming distances
    if metric in ['dot']:
        assert np.setdiff1d(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self))).size <= n_samples // 10
    elif metric in ['hamming']:
        assert np.setdiff1d(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self))).size <= n_samples // 100
    else:
        assert_array_equal(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self)))

    if metric in ['dot', 'angular']:
        pass  # does not guarantee self distance 0
    else:  # distances in [0, inf]
        assert_array_almost_equal(neigh_dist_self[:, 0], np.zeros(len(neigh_dist_self)))


@pytest.mark.parametrize('metric', RandomProjectionTree.valid_metrics)
def test_kneighbors_with_or_without_self_hit(metric):
    X, y = make_classification(random_state=1234435)
    n_candidates = 5
    ann = RandomProjectionTree(metric=metric,
                               n_candidates=n_candidates,
                               )
    ann.fit(X, y)
    ind_self = ann.kneighbors(X, n_candidates=n_candidates+1, return_distance=False)
    ind_no_self = ann.kneighbors(n_candidates=n_candidates, return_distance=False)

    if metric in ['dot']:  # dot is just inaccurate...
        assert (ind_self[:, 0] == np.arange(len(ind_self))).sum() > 92
        assert np.setdiff1d(ind_self[:, 1:], ind_no_self).size <= 10
    else:
        assert_array_equal(ind_self[:, 0], np.arange(len(ind_self)))
        assert_array_equal(ind_self[:, 1:], ind_no_self)


def test_squared_euclidean_same_neighbors_as_euclidean():
    X, y = make_classification()
    ann = RandomProjectionTree(metric='euclidean')
    ann.fit(X, y)
    neigh_dist_eucl, neigh_ind_eucl = ann.kneighbors(X)

    ann = RandomProjectionTree(metric='sqeuclidean')
    ann.fit(X, y)
    neigh_dist_sqeucl, neigh_ind_sqeucl = ann.kneighbors(X)

    assert_array_equal(neigh_ind_eucl, neigh_ind_sqeucl)
    assert_array_almost_equal(neigh_dist_eucl ** 2, neigh_dist_sqeucl)


def test_same_neighbors_as_with_exact_nn_search():
    X = np.random.RandomState(42).randn(10, 2)

    nn = NearestNeighbors()
    nn_dist, nn_neigh = nn.fit(X).kneighbors(return_distance=True)

    ann = RandomProjectionTree()
    ann_dist, ann_neigh = ann.fit(X).kneighbors(return_distance=True)

    assert_array_almost_equal(ann_dist, nn_dist, decimal=5)
    assert_array_almost_equal(ann_neigh, nn_neigh, decimal=0)


def test_is_valid_estimator():
    check_estimator(RandomProjectionTree)


@pytest.mark.parametrize('mmap_dir', [None, 'auto', '/dev/shm', '/tmp'])
def test_memory_mapped(mmap_dir):
    X, y = make_classification(n_samples=10,
                               n_features=5,
                               random_state=123,
                               )
    ann = RandomProjectionTree(mmap_dir=mmap_dir)
    ann.fit(X, y)
    _ = ann.kneighbors(X)
    _ = ann.kneighbors()
