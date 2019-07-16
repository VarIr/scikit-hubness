# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raises
from skhubness.reduction.shared_neighbors import SharedNearestNeighbors, SimhubIn
from skhubness.neighbors import NearestNeighbors


@pytest.mark.parametrize('method', [SharedNearestNeighbors, SimhubIn])
def test_snn(method):
    X, y = make_classification()
    nn = NearestNeighbors()
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    snn = method()
    with assert_raises(NotImplementedError):
        snn.fit(neigh_dist, neigh_ind, X, assume_sorted=True)

    with assert_raises(NotFittedError):
        snn.transform(neigh_dist, neigh_ind, X, assume_sorted=True)
