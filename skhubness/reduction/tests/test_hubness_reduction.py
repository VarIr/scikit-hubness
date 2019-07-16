from itertools import product
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_array_equal
from skhubness.analysis import Hubness
from skhubness.data import load_dexter
from skhubness.neighbors import kneighbors_graph, NearestNeighbors
from skhubness.reduction import NoHubnessReduction


HUBNESS_ALGORITHMS = ('mp',
                      'ls',
                      )
MP_PARAMS = tuple({'method': method} for method in ['normal', 'empiric'])
LS_PARAMS = tuple({'method': method} for method in ['standard', 'nicdm'])
HUBNESS_ALGORITHMS_WITH_PARAMS = ((*product(['mp'], MP_PARAMS),
                                   *product(['ls'], LS_PARAMS),
                                   ))


@pytest.mark.parametrize('hubness_param', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_neighbors_dexter(hubness_param, metric):
    hubness, param = hubness_param
    X, y = load_dexter()

    # Hubness in standard spaces
    hub = Hubness(k=10, metric=metric)
    hub.fit(X)
    k_skew_orig = hub.score()

    # Hubness in secondary distance spaces (after hub. red.)
    graph = kneighbors_graph(X, n_neighbors=10, metric=metric,
                             hubness=hubness, hubness_params=param)
    hub = Hubness(k=10, metric='precomputed')
    hub.fit(graph)
    k_skew_hr = hub.score(has_self_distances=True)

    assert k_skew_hr < k_skew_orig * 8/10,\
        f'k-occurrence skewness was not reduced by at least 20% for dexter with {hubness}'


def test_same_indices():
    X, y = make_classification()
    nn = NearestNeighbors()
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()
    hr = NoHubnessReduction()
    _, neigh_ind_hr = hr.fit_transform(neigh_dist, neigh_ind, X, return_distance=True)
    neigh_ind_ht_no_dist = hr.fit_transform(neigh_dist, neigh_ind, X, return_distance=False)
    assert_array_equal(neigh_ind, neigh_ind_hr)
    assert_array_equal(neigh_ind_hr, neigh_ind_ht_no_dist)
