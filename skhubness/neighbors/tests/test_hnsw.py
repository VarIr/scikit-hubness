# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils.testing import assert_raises
from skhubness.neighbors import HNSW


@pytest.mark.parametrize('metric', ['invalid', None])
def test_invalid_metric(metric):
    X, y = make_classification(n_samples=10, n_features=10)
    hnsw = HNSW(metric=metric)
    with assert_raises(ValueError):
        _ = hnsw.fit(X, y)


def test_fail_kneighbors_without_data():
    X, y = make_classification(n_samples=10, n_features=10)
    hnsw = HNSW()
    hnsw.fit(X, y)
    with assert_raises(NotImplementedError):
        hnsw.kneighbors()


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
@pytest.mark.parametrize('n_jobs', [-1, 1, None])
@pytest.mark.parametrize('verbose', [0, 1])
def test_kneighbors_with_or_without_self_hit(metric, n_jobs, verbose):
    X, y = make_classification()
    hnsw = HNSW(metric=metric, n_jobs=n_jobs, verbose=verbose)
    hnsw.fit(X, y)
    neigh_dist_self, neigh_ind_self = hnsw.kneighbors(X, return_distance=True)
    ind_only_self = hnsw.kneighbors(X, return_distance=False)

    assert_array_equal(neigh_ind_self, ind_only_self)
    assert_array_equal(neigh_ind_self[:, 0], np.arange(len(neigh_ind_self)))
    if metric in ['cosine']:  # similarities in [0, 1]
        assert_array_almost_equal(neigh_dist_self[:, 0], np.ones(len(neigh_dist_self)))
    else:  # distances in [0, inf]
        assert_array_almost_equal(neigh_dist_self[:, 0], np.zeros(len(neigh_dist_self)))


def test_squared_euclidean_same_neighbors_as_euclidean():
    X, y = make_classification()
    hnsw = HNSW(metric='minkowski')
    hnsw.fit(X, y)
    neigh_dist_eucl, neigh_ind_eucl = hnsw.kneighbors(X)

    hnsw = HNSW(metric='sqeuclidean')
    hnsw.fit(X, y)
    neigh_dist_sqeucl, neigh_ind_sqeucl = hnsw.kneighbors(X)

    assert_array_equal(neigh_ind_eucl, neigh_ind_sqeucl)
    assert_array_almost_equal(neigh_dist_eucl ** 2, neigh_dist_sqeucl)
