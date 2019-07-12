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
