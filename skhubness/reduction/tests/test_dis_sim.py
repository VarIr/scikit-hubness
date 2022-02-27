# SPDX-License-Identifier: BSD-3-Clause
import warnings

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_raises

from skhubness.reduction import DisSimLocal
from skhubness.reduction.tests.reference_algorithms import ReferenceDisSimLocal


def test_squared_vs_nonsquared_and_reference_vs_transformer_base():
    X, y = make_classification(random_state=123)
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()
    # DSL requires squared Euclidean distance matrix
    neigh_graph = nn.kneighbors_graph(mode="distance")
    neigh_graph.data **= 2

    hr_squared = ReferenceDisSimLocal(k=5, squared=True)
    hr = ReferenceDisSimLocal(k=5, squared=False)
    hr_graph_squared = DisSimLocal(k=5, return_squared_distances=True)
    hr_graph = DisSimLocal(k=5, return_squared_distances=False)

    # Are squared-DSL and DSL^2 equivalent in the reference?
    dist_squared, _ = hr_squared.fit_transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    dist, ind = hr.fit_transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    assert_array_almost_equal(dist_squared, dist ** 2)

    # Equivalence of reference to KNeighborsTransformer-based implementation
    dsl_graph: csr_matrix = hr_graph.fit_transform(neigh_graph, vectors=X)
    assert_array_equal(ind.ravel(), dsl_graph.indices)
    assert_array_almost_equal(dist.ravel(), dsl_graph.data)
    dsl_graph_squared: csr_matrix = hr_graph_squared.fit_transform(neigh_graph, vectors=X)
    assert_array_equal(dsl_graph.indices, dsl_graph_squared.indices)
    assert_array_almost_equal(dsl_graph.data ** 2, dsl_graph_squared.data)


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "cosine", "cityblock", "seuclidean"])
def test_warn_on_non_squared_euclidean_distances(metric):
    X = np.random.rand(3, 10)
    nn = NearestNeighbors(n_neighbors=2, metric=metric)
    nn.fit(X)
    graph = nn.kneighbors_graph(mode="distance")
    dsl = DisSimLocal(k=1)
    if metric != "sqeuclidean":
        with pytest.warns(match="not defined for other"):
            dsl.fit_transform(graph, vectors=X)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dsl.fit_transform(graph, vectors=X)


@pytest.mark.parametrize("squared", [True, False])
@pytest.mark.parametrize("k", [1, 5, 10])
def test_fit_sorted_and_fit_transform(squared, k):
    X, y = make_classification()
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    hr = ReferenceDisSimLocal(k=k, squared=squared)

    nd_sorted, ni_sorted = hr.fit(neigh_dist, neigh_ind, X, assume_sorted=True)\
                             .transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    nd_unsort, ni_unsort = hr.fit(neigh_dist, neigh_ind, X, assume_sorted=False)\
                             .transform(neigh_dist, neigh_ind, X, assume_sorted=False)

    assert_array_equal(ni_sorted, ni_unsort)
    assert_array_almost_equal(nd_sorted, nd_unsort)

    nd_sorted_fit_tr, ni_sorted_fit_tr = hr.fit_transform(neigh_dist, neigh_ind, X, assume_sorted=True)
    nd_unsorted_fit_tr, ni_unsorted_fit_tr = hr.fit_transform(neigh_dist, neigh_ind, X, assume_sorted=False)

    assert_array_almost_equal(nd_sorted, nd_sorted_fit_tr)
    assert_array_equal(ni_sorted, ni_sorted_fit_tr)
    assert_array_almost_equal(nd_unsort, nd_unsorted_fit_tr)
    assert_array_equal(ni_unsort, ni_unsorted_fit_tr)


@pytest.mark.parametrize("k", ["invalid", None, -1, 0])
@pytest.mark.parametrize("DisSimLocalClass", [DisSimLocal, ReferenceDisSimLocal])
def test_invalid_k(k, DisSimLocalClass):
    X, y = make_classification(n_samples=10, )
    nn = NearestNeighbors()
    nn.fit(X, y)

    hr = DisSimLocalClass(k=k)
    if isinstance(hr, DisSimLocal):
        graph = nn.kneighbors_graph(mode="distance")
        kwargs = {"X": graph, "vectors": X}
    else:
        neigh_dist, neigh_ind = nn.kneighbors()
        kwargs = {"neigh_dist": neigh_dist, "neigh_ind": neigh_ind, "X": X, "assume_sorted": True}

    expected_exception = ValueError if isinstance(k, int) else TypeError
    with assert_raises(expected_exception):
        hr.fit(**kwargs)


@pytest.mark.parametrize("k", [9, 10, 11])
@pytest.mark.parametrize("DisSimLocalClass", [DisSimLocal, ReferenceDisSimLocal])
def test_warning_on_too_large_k(k, DisSimLocalClass, n_samples=10):
    X, y = make_classification(n_samples=n_samples)
    nn = NearestNeighbors(n_neighbors=n_samples-1, metric="sqeuclidean")
    nn.fit(X, y)

    hr = DisSimLocalClass(k=k)
    if isinstance(hr, DisSimLocal):
        graph = nn.kneighbors_graph(mode="distance")
        kwargs = {"X": graph, "vectors": X}
    else:
        neigh_dist, neigh_ind = nn.kneighbors()
        kwargs = {"neigh_dist": neigh_dist, "neigh_ind": neigh_ind, "X": X, "assume_sorted": True}

    if k < n_samples:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hr.fit(**kwargs)
            _ = hr.transform(**kwargs)
    else:
        with pytest.warns(Warning):
            hr.fit(**kwargs)
        with pytest.warns(Warning):
            _ = hr.transform(**kwargs)


@pytest.mark.parametrize("k", [9, 10, 11])
def test_warning_on_too_few_neighbors(k, n_samples=10):
    # ReferenceDisSimLocal only
    X, y = make_classification(n_samples=n_samples)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X, y)
    neigh_dist, neigh_ind = nn.kneighbors()

    hr = ReferenceDisSimLocal(k=k)
    with pytest.warns(Warning):
        hr.fit(neigh_dist, neigh_ind, X, assume_sorted=True)
    with pytest.warns(Warning):
        _ = hr.transform(neigh_dist, neigh_ind, X, assume_sorted=True)
