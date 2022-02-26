#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import euclidean_distances
from sklearn.utils.estimator_checks import check_estimator

from skhubness import Hubness
from skhubness.analysis.estimation import VALID_HUBNESS_MEASURES

DIST = csr_matrix(squareform(np.array([.2, .1, .8, .4, .3, .5, .7, 1., .6, .9])))


def test_dev_Hubness(hubness_k: int = 10):
    X, y = make_classification(
        n_samples=1_000,
        n_features=100,
        n_informative=80,
    )
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=123,
        shuffle=True,
        stratify=y,
    )
    from sklearn.neighbors import NearestNeighbors
    ann = NearestNeighbors(n_neighbors=5, n_jobs=16)
    ann.fit(X_train)
    for n_neighbors in [1, 5, 10]:
        kng_train = ann.kneighbors_graph(
            n_neighbors=n_neighbors,
            mode="distance",
        )
        kng_test = ann.kneighbors_graph(
            X_test,
            n_neighbors=n_neighbors,
            mode="distance",
        )
        hub = Hubness(k=hubness_k)
        if n_neighbors < hubness_k:
            with pytest.raises(ValueError):
                hub.fit(kng_train)
            continue
        else:
            hub.fit(kng_train)
        score = hub.score(kng_test)
        print(score)


@pytest.mark.parametrize("Class", [Hubness, ])
def test_estimator(Class):
    """ Check that Class is a valid scikit-learn estimator. """
    check_estimator(Class())


@pytest.mark.parametrize("verbose", [-1, 0, 1, 2, 3, None])
def test_hubness(verbose):
    """Test hubness against ground truth calc on spreadsheet"""
    HUBNESS_TRUE = -0.2561204163  # Hubness truth: skewness calculated with bias
    hub = Hubness(k=2, metric="precomputed", verbose=verbose)
    hub.fit(DIST)
    Sk2 = hub.score(DIST)  # has_self_distances=True)
    np.testing.assert_almost_equal(Sk2, HUBNESS_TRUE, decimal=10)


def test_return_k_neighbors():
    """ This was only available in legacy Hubness estimation before v.30"""
    X, _ = make_classification()
    hub = Hubness(return_value="k_neighbors")
    with pytest.raises(ValueError):
        hub.fit(X)


@pytest.mark.parametrize("return_value", ["all", "k_skewness"])
@pytest.mark.parametrize("return_k_occurrence", [True, False])
def test_return_k_occurrence(return_value, return_k_occurrence):
    X, _ = make_classification()
    hub = Hubness(
        return_value=return_value,
        return_k_occurrence=return_k_occurrence,
    )
    hub.fit(X)
    result = hub.score()
    if return_k_occurrence:
        k_occ = result["k_occurrence"]
        assert k_occ.shape == (X.shape[0], )
    else:
        ExpectedError = KeyError if return_value == "all" else TypeError
        with pytest.raises(ExpectedError):
            _ = result["k_occurrence"]


@pytest.mark.parametrize("return_value", ["all", "k_skewness"])
@pytest.mark.parametrize("return_hubs", [True, False])
def test_return_hubs(return_value, return_hubs):
    X, _ = make_classification(random_state=123)
    hub = Hubness(
        return_value=return_value,
        return_hubs=return_hubs,
    )
    hub.fit(X)
    result = hub.score()
    if return_hubs:
        hubs = result["hubs"]
        # TOFU hub number for `make_classification(random_state=123)`
        assert hubs.shape == (8, )
    else:
        ExpectedError = KeyError if return_value == "all" else TypeError
        with pytest.raises(ExpectedError):
            _ = result["hubs"]


@pytest.mark.parametrize("return_value", ["all", "k_skewness"])
@pytest.mark.parametrize("return_antihubs", [True, False])
def test_return_hubs(return_value, return_antihubs):
    X, _ = make_classification(
        random_state=123,
    )
    hub = Hubness(
        return_value=return_value,
        return_antihubs=return_antihubs,
    )
    hub.fit(X)
    result = hub.score()
    if return_antihubs:
        antihubs = result["antihubs"]
        # TOFU anti-hub number for `make_classification(random_state=123)`
        assert antihubs.shape == (0, )
    else:
        ExpectedError = KeyError if return_value == "all" else TypeError
        with pytest.raises(ExpectedError):
            _ = result["antihubs"]


def test_limiting_factor():
    """ Different implementations of Gini index calculation should give the same result. """
    X, _ = make_classification()
    hub = Hubness(return_k_occurrence=True)
    hub.fit(X)
    k_occ = hub.score().get("k_occurrence")

    gini = {
        str(x): hub._calc_gini_index(k_occ, limiting=x)
        for x in ["space", "time", None]
    }

    assert gini["space"] == gini["time"] == gini["None"]


def test_all_but_gini():
    X, _ = make_classification()
    hub = Hubness(
        return_k_occurrence=True,
        return_antihubs=True,
        return_hubs=True,
        return_value="all_but_gini",
    )
    hub.fit(X)
    measures = hub.score()

    hit_gini = False
    for m in VALID_HUBNESS_MEASURES:
        if m in ["all", "all_but_gini"]:
            continue
        elif m == "gini":
            assert m not in measures
            hit_gini = True
        else:
            assert m in measures
    assert hit_gini


@pytest.mark.parametrize('verbose', [True, False])
def test_shuffle_equal(verbose):
    # for this data set there shouldn't be any equal distances,
    # and shuffle should make no difference
    X, _ = make_classification(random_state=12354)
    dist = csr_matrix(euclidean_distances(X))
    skew_shuffle, skew_no_shuffle = \
        [Hubness(metric='precomputed', shuffle_equal=shuffle_equal, verbose=verbose)
         .fit(dist).score() for shuffle_equal in [True, False]]
    assert skew_no_shuffle == skew_shuffle


@pytest.mark.xfail(reason="Test rationale might be flawed. Need to think again.")
def test_atkinson():
    X, _ = make_classification(random_state=123)
    hub = Hubness(return_k_occurrence=True).fit(X)
    k_occ = hub.score().get("k_occurrence")

    atkinson_0999 = hub._calc_atkinson_index(k_occ, eps=.999)
    atkinson_1000 = hub._calc_atkinson_index(k_occ, eps=1)

    np.testing.assert_almost_equal(atkinson_0999, atkinson_1000, decimal=3)


@pytest.mark.parametrize("seed", [0, 626])
@pytest.mark.parametrize("n_samples", [20, 200])
@pytest.mark.parametrize("n_features", [2, 500])
@pytest.mark.parametrize("k", [1, 5, 10])
def test_hubness_return_values_are_self_consistent(n_samples, n_features, k, seed):
    """Test that the three returned values fit together"""
    np.random.seed(seed)
    vectors = 99. * (np.random.rand(n_samples, n_features) - 0.5)
    k = 10
    hub = Hubness(
        k=k,
        metric="euclidean",
        return_k_occurrence=True,
    )
    hub.fit(vectors)
    res = hub.score()
    skew = res.get("k_skewness")
    neigh = hub._k_neighbors(hub.kng_indexed_)
    occ = res.get("k_occurrence")
    # Neighbors are just checked for correct shape
    assert neigh.shape == (n_samples, k)
    # Count k-occurrence (different method than in module)
    neigh = neigh.ravel()
    occ_true = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        occ_true[i] = (neigh == i).sum()
    np.testing.assert_array_equal(occ, occ_true)
    # Calculate skewness (different method than in module)
    x0 = occ - occ.mean()
    s2 = (x0 ** 2).mean()
    m3 = (x0 ** 3).mean()
    skew_true = m3 / (s2 ** 1.5)
    np.testing.assert_equal(skew, skew_true)


@pytest.mark.parametrize("dist", [DIST])
@pytest.mark.parametrize("n_jobs", [-1, 1, 2, 4])
@pytest.mark.filterwarnings("ignore:divide by zero")
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_parallel_hubness_equal_serial_hubness_distance_based(dist, n_jobs):
    # Parallel
    hub = Hubness(k=4, metric="precomputed", return_k_occurrence=True, n_jobs=n_jobs)
    hub.fit(dist)
    res = hub.score()
    skew_p = res.get("k_skewness")
    occ_p = res.get("k_occurrence")

    # Sequential
    hub = Hubness(k=4, metric="precomputed", return_k_occurrence=True, n_jobs=1)
    hub.fit(dist)
    res = hub.score()
    skew_s = res.get("k_skewness")
    occ_s = res.get("k_occurrence")

    np.testing.assert_array_almost_equal(skew_p, skew_s, decimal=7)
    np.testing.assert_array_almost_equal(occ_p, occ_s, decimal=7)


@pytest.mark.parametrize("has_self_distances", [True, False])
def test_hubness_against_distance(has_self_distances):
    """Test hubness class against distance-based methods."""

    np.random.seed(123)
    X = np.random.rand(100, 50)
    D = euclidean_distances(X)
    verbose = 1

    hub = Hubness(
        k=10,
        metric="precomputed",
        return_k_occurrence=True,
    )
    hub.fit(D)
    res = hub.score()
    skew_d = res.get("k_skewness")
    occ_d = res.get("k_occurrence")

    hub = Hubness(
        k=10,
        metric="euclidean",
        return_k_occurrence=True,
        verbose=verbose,
    )
    hub.fit(X)
    res = hub.score(X if not has_self_distances else None)
    skew_v = res.get("k_skewness")
    occ_v = res.get("k_occurrence")

    np.testing.assert_allclose(skew_d, skew_v, atol=1e-7)
    np.testing.assert_array_equal(occ_d, occ_v)


@pytest.mark.parametrize("hubness_measure", ["k_skewness", "k_skewness_truncnorm", "robinhood", "gini", "atkinson"])
def test_hubness_independent_on_data_set_size(hubness_measure):
    """ New measures should pass, traditional skewness should fail. """
    thousands = 3
    n_objects = thousands * 1_000
    rng = np.random.RandomState(1247)
    X = rng.rand(n_objects, 128)
    N_SAMPLES_LIST = np.arange(1, thousands + 1) * 1_000
    value = np.empty(N_SAMPLES_LIST.size)
    for i, n_samples in enumerate(N_SAMPLES_LIST):
        ind = rng.permutation(n_objects)[:n_samples]
        X_sample = X[ind, :]
        hub = Hubness(return_value="all")
        hub.fit(X_sample)
        measures = hub.score()
        if hubness_measure == "k_skewness":
            value[i] = measures.get("k_skewness")
        elif hubness_measure == "k_skewness_truncnorm":
            value[i] = measures.get("k_skewness_truncnorm")
        elif hubness_measure == "robinhood":
            value[i] = measures.get("robinhood")
        elif hubness_measure == "gini":
            value[i] = measures.get("gini")
        elif hubness_measure == "atkinson":
            value[i] = measures.get("atkinson")
        assert value[i] == measures[hubness_measure]
        if i > 0:
            if hubness_measure == "k_skewness":
                with np.testing.assert_raises(AssertionError):
                    np.testing.assert_allclose(value[i], value[i-1], rtol=0.1)
            else:
                np.testing.assert_allclose(
                    value[i], value[i - 1], rtol=2e-1,
                    err_msg=(f"Hubness measure is too dependent on data set "
                             f"size with S({N_SAMPLES_LIST[i]}) = x "
                             f"and S({N_SAMPLES_LIST[i-1]}) = y."))
    if hubness_measure == "k_skewness":
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(value[-1], value[0], rtol=0.1)
    else:
        np.testing.assert_allclose(value[-1], value[0], rtol=2e-1)
