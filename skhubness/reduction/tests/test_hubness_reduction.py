# SPDX-License-Identifier: BSD-3-Clause

from itertools import product
import pytest
from sklearn.datasets import make_classification
from sklearn.utils._testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors

from skhubness.analysis import Hubness
from skhubness.data import load_dexter
from skhubness.reduction import LocalScaling, MutualProximity, DisSimLocal
from skhubness.reduction.tests.reference_algorithms import ReferenceNoHubnessReduction


HUBNESS_REDUCTION = (
    LocalScaling, MutualProximity, DisSimLocal,
)
MP_PARAMS = tuple({"method": method} for method in ["normal", "empiric"])
LS_PARAMS = tuple({"method": method} for method in ["standard", "nicdm"])
HUBNESS_REDUCTION_WITH_PARAMS = ((
    *product([MutualProximity], MP_PARAMS),
    *product([LocalScaling], LS_PARAMS),
    (DisSimLocal, {}),
))


@pytest.mark.parametrize("hubness_param", HUBNESS_REDUCTION_WITH_PARAMS)
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_neighbors_dexter(hubness_param, metric):
    HubnessReduction, param = hubness_param
    if HubnessReduction is DisSimLocal and metric != "euclidean":
        pytest.skip("DisSimLocal works only with Euclidean distances")
    X, y = load_dexter()

    # Hubness in standard spaces
    hub = Hubness(k=10, metric=metric)
    hub.fit(X)
    k_skew_orig = hub.score()

    # Hubness in secondary distance spaces (after hub. red.)
    nn = NearestNeighbors(n_neighbors=50, metric=metric)
    graph = nn.fit(X).kneighbors_graph(X, mode="distance")
    hub_red = HubnessReduction(method=param.get("method"))
    if HubnessReduction is DisSimLocal:
        graph = hub_red.fit(graph, vectors=X).transform(graph, vectors=X)
    else:
        graph = hub_red.fit(graph).transform(graph)
    hub = Hubness(k=10, metric="precomputed")
    hub.fit(graph)
    k_skew_hr = hub.score()

    assert k_skew_hr < k_skew_orig * 8/10,\
        f"k-occurrence skewness was not reduced by at least 20% for dexter with {HubnessReduction}"


def test_same_indices():
    X, y = make_classification()
    nn = NearestNeighbors()
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()
    hr = ReferenceNoHubnessReduction()
    _, neigh_ind_hr = hr.fit_transform(neigh_dist, neigh_ind, X, return_distance=True)
    neigh_ind_ht_no_dist = hr.fit_transform(neigh_dist, neigh_ind, X, return_distance=False)
    assert_array_equal(neigh_ind, neigh_ind_hr)
    assert_array_equal(neigh_ind_hr, neigh_ind_ht_no_dist)
