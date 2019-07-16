# SPDX-License-Identifier: BSD-3-Clause

import pytest
import sys
from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from skhubness.neighbors import LSH


@pytest.mark.skipif(sys.platform == 'win32', reason='Currently no LSH supported on Windows.')
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
@pytest.mark.parametrize('n_jobs', [-1, 1, None])
@pytest.mark.parametrize('verbose', [0, 1])
def test_kneighbors_with_or_without_self_hit(metric, n_jobs, verbose):
    X, y = make_classification()
    lsh = LSH(metric=metric, n_jobs=n_jobs, verbose=verbose)
    lsh.fit(X, y)
    neigh_dist, neigh_ind = lsh.kneighbors(return_distance=True)
    neigh_dist_self, neigh_ind_self = lsh.kneighbors(X, return_distance=True)

    ind_only = lsh.kneighbors(return_distance=False)
    ind_only_self = lsh.kneighbors(X, return_distance=False)

    assert_array_equal(neigh_ind, ind_only)
    assert_array_equal(neigh_ind_self, ind_only_self)

    assert_array_equal(neigh_ind[:, :-1],
                       neigh_ind_self[:, 1:])
    assert_array_almost_equal(neigh_dist[:, :-1],
                              neigh_dist_self[:, 1:])


@pytest.mark.skipif(sys.platform == 'win32', reason='Currently no LSH supported on Windows.')
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
@pytest.mark.parametrize('n_jobs', [-1, 1, None])
@pytest.mark.parametrize('verbose', [0, 1])
def test_radius_neighbors_with_or_without_self_hit(metric, n_jobs, verbose):
    X, y = make_classification()
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


@pytest.mark.skipif(sys.platform == 'win32', reason='Currently no LSH supported on Windows.')
def test_squared_euclidean_same_neighbors_as_euclidean():
    X, y = make_classification()
    lsh = LSH(metric='minkowski')
    lsh.fit(X, y)
    neigh_dist_eucl, neigh_ind_eucl = lsh.kneighbors()
    radius = neigh_dist_eucl[:, 2].max()
    rad_dist_eucl, rad_ind_eucl = lsh.radius_neighbors(radius=radius)

    lsh = LSH(metric='sqeuclidean')
    lsh.fit(X, y)
    neigh_dist_sqeucl, neigh_ind_sqeucl = lsh.kneighbors()
    rad_dist_sqeucl, rad_ind_sqeucl = lsh.radius_neighbors(radius=radius**2)

    assert_array_equal(neigh_ind_eucl, neigh_ind_sqeucl)
    assert_array_almost_equal(neigh_dist_eucl ** 2, neigh_dist_sqeucl)
    for i in range(len(rad_ind_eucl)):
        assert_array_equal(rad_ind_eucl[i], rad_ind_sqeucl[i])
        assert_array_almost_equal(rad_dist_eucl[i] ** 2, rad_dist_sqeucl[i])


@pytest.mark.skipif(sys.platform == 'win32', reason='Currently no LSH supported on Windows.')
@pytest.mark.parametrize('metric', ['invalid', 'manhattan', 'l1', 'chebyshev'])
def test_warn_on_invalid_metric(metric):
    X, y = make_classification()
    lsh = LSH(metric='euclidean')
    lsh.fit(X, y)
    neigh_dist, neigh_ind = lsh.kneighbors()

    lsh.metric = metric
    with pytest.warns(UserWarning):
        lsh.fit(X, y)
    neigh_dist_inv, neigh_ind_inv = lsh.kneighbors()

    assert_array_equal(neigh_ind, neigh_ind_inv)
    assert_array_almost_equal(neigh_dist, neigh_dist_inv)
