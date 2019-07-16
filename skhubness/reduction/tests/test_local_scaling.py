# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from skhubness.reduction import LocalScaling
from skhubness.neighbors import NearestNeighbors

LS_METHODS = ['standard', 'nicdm',
              ]


@pytest.mark.parametrize('method', LS_METHODS)
@pytest.mark.parametrize('verbose', [0, 1])
def test_fit_sorted(method, verbose):
    X, y = make_classification()
    nn = NearestNeighbors()
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    ls = LocalScaling(method=method, verbose=verbose)

    nd_sorted, ni_sorted = ls.fit(neigh_dist, neigh_ind, X, assume_sorted=True)\
                             .transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    nd_unsort, ni_unsort = ls.fit(neigh_dist, neigh_ind, X, assume_sorted=False)\
                             .transform(neigh_dist, neigh_ind, X, assume_sorted=False)

    assert_array_almost_equal(nd_sorted, nd_unsort)
    assert_array_equal(ni_sorted, ni_unsort)


@pytest.mark.parametrize('method', ['invalid', None])
def test_invalid_method(method):
    X, y = make_classification(n_samples=10, )
    nn = NearestNeighbors()
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    ls = LocalScaling(method=method)
    ls.fit(neigh_dist, neigh_ind, X, assume_sorted=True)
    with assert_raises(ValueError):
        _ = ls.transform(neigh_dist, neigh_ind, X, assume_sorted=True)
