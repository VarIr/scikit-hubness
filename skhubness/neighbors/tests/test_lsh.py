# SPDX-License-Identifier: BSD-3-Clause

import pytest
import sys
from sklearn.datasets import make_classification
from sklearn.preprocessing import Normalizer
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.estimator_checks import check_estimator
from skhubness.neighbors import FalconnLSH, PuffinnLSH

# Exclude libraries that are not available on specific platforms
if sys.platform == 'win32':  # pragma: no cover
    LSH_METHODS = ()
    LSH_WITH_RADIUS = ()
elif sys.platform == 'darwin':  # pragma: no cover
    # Work-around for imprecise Puffinn on Mac: disable tests for now
    LSH_METHODS = (FalconnLSH, )
    LSH_WITH_RADIUS = (FalconnLSH, )
else:
    LSH_METHODS = (FalconnLSH, PuffinnLSH, )
    LSH_WITH_RADIUS = (FalconnLSH, )


@pytest.mark.parametrize('LSH', LSH_METHODS)
def test_estimator(LSH):
    if LSH in [FalconnLSH]:
        pytest.xfail(f'Falconn does not support pickling its index.')
    check_estimator(LSH)


@pytest.mark.parametrize('LSH', LSH_METHODS)
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
@pytest.mark.parametrize('n_jobs', [-1, 1, None])
@pytest.mark.parametrize('verbose', [0, 1])
def test_kneighbors_with_or_without_self_hit(LSH: callable, metric, n_jobs, verbose):
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

    assert (neigh_ind - neigh_ind_self).mean() <= .01, f'More than 1% of neighbors mismatch'
    assert ((neigh_dist - neigh_dist_self) < 0.0001).mean() <= 0.01,\
        f'Not almost equal to 4 decimals in more than 1% of neighbor slots'


@pytest.mark.parametrize('LSH', LSH_WITH_RADIUS)
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
@pytest.mark.parametrize('n_jobs', [-1, 1, None])
@pytest.mark.parametrize('verbose', [0, 1])
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


@pytest.mark.parametrize('LSH', LSH_METHODS)
def test_squared_euclidean_same_neighbors_as_euclidean(LSH):
    X, y = make_classification(random_state=234)
    X = Normalizer().fit_transform(X)
    lsh = LSH(metric='minkowski')
    lsh.fit(X, y)
    neigh_dist_eucl, neigh_ind_eucl = lsh.kneighbors()

    lsh_sq = LSH(metric='sqeuclidean')
    lsh_sq.fit(X, y)
    neigh_dist_sqeucl, neigh_ind_sqeucl = lsh_sq.kneighbors()

    assert_array_equal(neigh_ind_eucl, neigh_ind_sqeucl)
    assert_array_almost_equal(neigh_dist_eucl ** 2, neigh_dist_sqeucl)

    if LSH in LSH_WITH_RADIUS:
        radius = neigh_dist_eucl[:, 2].max()
        rad_dist_eucl, rad_ind_eucl = lsh.radius_neighbors(radius=radius)
        rad_dist_sqeucl, rad_ind_sqeucl = lsh_sq.radius_neighbors(radius=radius**2)
        for i in range(len(rad_ind_eucl)):
            assert_array_equal(rad_ind_eucl[i], rad_ind_sqeucl[i])
            assert_array_almost_equal(rad_dist_eucl[i] ** 2, rad_dist_sqeucl[i])


@pytest.mark.parametrize('LSH', LSH_METHODS)
@pytest.mark.parametrize('metric', ['invalid', 'manhattan', 'l1', 'chebyshev'])
def test_warn_on_invalid_metric(LSH, metric):
    X, y = make_classification(random_state=24643)
    X = Normalizer().fit_transform(X)
    lsh = LSH(metric='euclidean')
    lsh.fit(X, y)
    neigh_dist, neigh_ind = lsh.kneighbors()

    lsh.metric = metric
    with pytest.warns(UserWarning):
        lsh.fit(X, y)
    neigh_dist_inv, neigh_ind_inv = lsh.kneighbors()

    assert_array_equal(neigh_ind, neigh_ind_inv)
    assert_array_almost_equal(neigh_dist, neigh_dist_inv)
