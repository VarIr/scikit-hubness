# SPDX-License-Identifier: BSD-3-Clause

from itertools import product
from pickle import PicklingError
import sys
import warnings

import numpy as np
from scipy.sparse import (bsr_matrix, coo_matrix, csc_matrix, csr_matrix,
                          dok_matrix, lil_matrix, issparse)

import pytest

from sklearn import metrics
from sklearn import datasets
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors.base import VALID_METRICS_SPARSE, VALID_METRICS
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_in
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raises_regex
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.validation import check_random_state

from sklearn.utils._joblib import joblib
from sklearn.utils._joblib import parallel_backend

from skhubness import neighbors

rng = np.random.RandomState(0)
# load and shuffle iris dataset
iris = datasets.load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# load and shuffle digits
digits = datasets.load_digits()
perm = rng.permutation(digits.target.size)
digits.data = digits.data[perm]
digits.target = digits.target[perm]

SPARSE_TYPES = (bsr_matrix, coo_matrix, csc_matrix, csr_matrix, dok_matrix,
                lil_matrix)
SPARSE_OR_DENSE = SPARSE_TYPES + (np.asarray,)

EXACT_ALGORITHMS = ('ball_tree',
                    'brute',
                    'kd_tree',
                    'auto',
                    )

# lsh uses FALCONN, which does not support Windows
if sys.platform == 'win32':
    APPROXIMATE_ALGORITHMS = ('hnsw',  # only on win32
                              )
else:
    APPROXIMATE_ALGORITHMS = ('lsh',
                              'hnsw',
                              )
HUBNESS_ALGORITHMS = ('mp',
                      'ls',
                      'dsl',
                      )
MP_PARAMS = tuple({'method': method} for method in ['normal', 'empiric'])
LS_PARAMS = tuple({'method': method} for method in ['standard', 'nicdm'])
DSL_PARAMS = tuple({'squared': val} for val in [True, False])
HUBNESS_ALGORITHMS_WITH_PARAMS = ((None, {}),
                                  *product(['mp'], MP_PARAMS),
                                  *product(['ls'], LS_PARAMS),
                                  *product(['dsl'], DSL_PARAMS),
                                  )
P = (1, 3, 4, np.inf, 2)  # Euclidean last, for tests against approx NN
JOBLIB_BACKENDS = list(joblib.parallel.BACKENDS.keys())

# Filter deprecation warnings.
neighbors.kneighbors_graph = ignore_warnings(neighbors.kneighbors_graph)
neighbors.radius_neighbors_graph = ignore_warnings(
    neighbors.radius_neighbors_graph)


def _weight_func(dist):
    """ Weight function to replace lambda d: d ** -2.
    The lambda function is not valid because:
    if d==0 then 0^-2 is not valid. """

    # Dist could be multidimensional, flatten it so all values
    # can be looped
    with np.errstate(divide='ignore'):
        retval = 1. / dist
    return retval ** 2


@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
def test_unsupervised_kneighbors(hubness_and_params,
                                 n_samples=20, n_features=5,
                                 n_query_pts=2, n_neighbors=5):
    # Test unsupervised neighbors methods
    hubness, hubness_params = hubness_and_params
    X = rng.rand(n_samples, n_features)
    test = rng.rand(n_query_pts, n_features)

    for p in P:
        results_nodist = []
        results = []

        for algorithm in EXACT_ALGORITHMS:
            neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors,
                                               algorithm=algorithm,
                                               algorithm_params={'n_candidates': n_neighbors},
                                               hubness=hubness, hubness_params=hubness_params,
                                               p=p)
            if hubness == 'dsl' and p != 2:
                with pytest.warns(UserWarning):
                    neigh.fit(X)
            else:
                neigh.fit(X)

            results_nodist.append(neigh.kneighbors(test,
                                                   return_distance=False))
            results.append(neigh.kneighbors(test, return_distance=True))

        for i in range(len(results) - 1):
            assert_array_almost_equal(results_nodist[i], results[i][1])
            assert_array_almost_equal(results[i][0], results[i + 1][0])
            assert_array_almost_equal(results[i][1], results[i + 1][1])

    # Test approximate NN against exact NN with Euclidean distances
    assert p == 2, f'Internal: last parameter p={p}, should have been 2'
    for algorithm in APPROXIMATE_ALGORITHMS:
        neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors,
                                           algorithm=algorithm,
                                           algorithm_params={'n_candidates': n_neighbors},
                                           hubness=hubness, hubness_params=hubness_params,
                                           p=p)
        neigh.fit(X)
        results_approx_nodist = neigh.kneighbors(test, return_distance=False)
        results_approx = neigh.kneighbors(test, return_distance=True)

        assert_array_equal(results_approx_nodist, results_approx[1])
        assert_array_almost_equal(results_approx[0], results[1][0])
        assert_array_almost_equal(results_approx[1], results[1][1])


@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.filterwarnings('ignore:Cannot perform hubness reduction with a single neighbor per query')
def test_unsupervised_inputs(hubness_and_params):
    # test the types of valid input into NearestNeighbors
    hubness, hubness_params = hubness_and_params
    X = rng.random_sample((10, 3))

    n_neighbors = 1
    hubness_params['k'] = n_neighbors
    nbrs_fid = neighbors.NearestNeighbors(n_neighbors=n_neighbors,
                                          algorithm_params={'n_candidates': n_neighbors},
                                          hubness=hubness, hubness_params=hubness_params,
                                          )
    nbrs_fid.fit(X)

    dist1, ind1 = nbrs_fid.kneighbors(X)

    nbrs = neighbors.NearestNeighbors(n_neighbors=n_neighbors,
                                      algorithm_params={'n_candidates': n_neighbors},
                                      hubness=hubness, hubness_params=hubness_params,
                                      )

    inputs = [nbrs_fid, neighbors.BallTree(X), neighbors.KDTree(X),
              neighbors.HNSW(n_candidates=1).fit(X),
              ]
    if sys.platform != 'win32':
        inputs += [neighbors.LSH(n_candidates=1).fit(X), ]

    for input_ in inputs:
        nbrs.fit(input_)
        dist2, ind2 = nbrs.kneighbors(X)

        assert_array_almost_equal(dist1, dist2)
        assert_array_almost_equal(ind1, ind2)


def test_n_neighbors_datatype():
    # Test to check whether n_neighbors is integer
    X = [[1, 1], [1, 1], [1, 1]]
    expected_msg = "n_neighbors does not take .*float.* " \
                   "value, enter integer value"
    msg = "Expected n_neighbors > 0. Got -3"

    neighbors_ = neighbors.NearestNeighbors(n_neighbors=3.,
                                            algorithm_params={'n_candidates': 2},
                                            )
    assert_raises_regex(TypeError, expected_msg, neighbors_.fit, X)
    assert_raises_regex(ValueError, msg,
                        neighbors_.kneighbors, X=X, n_neighbors=-3)
    assert_raises_regex(TypeError, expected_msg,
                        neighbors_.kneighbors, X=X, n_neighbors=3.)


def test_not_fitted_error_gets_raised():
    X = [[1]]
    neighbors_ = neighbors.NearestNeighbors()
    assert_raises(NotFittedError, neighbors_.kneighbors_graph, X)
    assert_raises(NotFittedError, neighbors_.radius_neighbors_graph, X)


def test_precomputed(random_state=42):
    """Tests unsupervised NearestNeighbors with a distance matrix."""
    # Note: smaller samples may result in spurious test success
    rng = np.random.RandomState(random_state)
    X = rng.random_sample((10, 4))
    Y = rng.random_sample((3, 4))
    DXX = metrics.pairwise_distances(X, metric='euclidean')
    DYX = metrics.pairwise_distances(Y, X, metric='euclidean')
    for method in ['kneighbors']:
        # TODO: also test radius_neighbors, but requires different assertion

        # As a feature matrix (n_samples by n_features)
        nbrs_X = neighbors.NearestNeighbors(n_neighbors=3,
                                            algorithm_params={'n_candidates': 2},
                                            )
        nbrs_X.fit(X)
        dist_X, ind_X = getattr(nbrs_X, method)(Y)

        # As a dense distance matrix (n_samples by n_samples)
        nbrs_D = neighbors.NearestNeighbors(n_neighbors=3,
                                            algorithm='brute',
                                            algorithm_params={'n_candidates': 2},
                                            metric='precomputed')
        nbrs_D.fit(DXX)
        dist_D, ind_D = getattr(nbrs_D, method)(DYX)
        assert_array_almost_equal(dist_X, dist_D)
        assert_array_almost_equal(ind_X, ind_D)

        # Check auto works too
        nbrs_D = neighbors.NearestNeighbors(n_neighbors=3,
                                            algorithm='auto',
                                            algorithm_params={'n_candidates': 2},
                                            metric='precomputed')
        nbrs_D.fit(DXX)
        dist_D, ind_D = getattr(nbrs_D, method)(DYX)
        assert_array_almost_equal(dist_X, dist_D)
        assert_array_almost_equal(ind_X, ind_D)

        # Check X=None in prediction
        dist_X, ind_X = getattr(nbrs_X, method)(None)
        dist_D, ind_D = getattr(nbrs_D, method)(None)
        assert_array_almost_equal(dist_X, dist_D)
        assert_array_almost_equal(ind_X, ind_D)

        # Must raise a ValueError if the matrix is not of correct shape
        assert_raises(ValueError, getattr(nbrs_D, method), X)

    target = np.arange(X.shape[0])
    for Est in (neighbors.KNeighborsClassifier,
                neighbors.RadiusNeighborsClassifier,
                neighbors.KNeighborsRegressor,
                neighbors.RadiusNeighborsRegressor):
        print(Est, end=' - ')
        est = Est(metric='euclidean',
                  algorithm_params={'n_candidates': 5},
                  )
        est.radius = est.n_neighbors = 1
        pred_X = est.fit(X, target).predict(Y)
        est.metric = 'precomputed'
        pred_D = est.fit(DXX, target).predict(DYX)
        assert_array_almost_equal(pred_X, pred_D)
        print('SUCCESS')


@pytest.mark.filterwarnings('ignore: The default value of cv')  # 0.22
def test_precomputed_cross_validation():
    # Ensure array is split correctly
    rng = np.random.RandomState(0)
    X = rng.rand(20, 2)
    D = pairwise_distances(X, metric='euclidean')
    y = rng.randint(3, size=20)
    for Est in (neighbors.KNeighborsClassifier,
                neighbors.RadiusNeighborsClassifier,
                neighbors.KNeighborsRegressor,
                neighbors.RadiusNeighborsRegressor):
        metric_score = cross_val_score(Est(algorithm_params={'n_candidates': 5}), X, y)
        precomp_score = cross_val_score(Est(metric='precomputed',
                                            algorithm_params={'n_candidates': 5},
                                            ),
                                        D, y)
        assert_array_equal(metric_score, precomp_score)


def test_unsupervised_radius_neighbors(n_samples=20, n_features=5,
                                       n_query_pts=2, radius=0.5,
                                       random_state=0):
    # Test unsupervised radius-based query
    rng = np.random.RandomState(random_state)

    from sklearn.metrics import euclidean_distances
    X = rng.rand(n_samples, n_features)
    D = euclidean_distances(X, squared=False)

    test = rng.rand(n_query_pts, n_features)
    test_dist = euclidean_distances(test, X, squared=False)
    for p in P:
        results = []

        for algorithm in EXACT_ALGORITHMS:
            neigh = neighbors.NearestNeighbors(radius=radius,
                                               algorithm=algorithm,
                                               algorithm_params={'n_candidates': 5},
                                               p=p)
            neigh.fit(X)

            ind1 = neigh.radius_neighbors(test, return_distance=False)

            # sort the results: this is not done automatically for
            # radius searches
            dist, ind = neigh.radius_neighbors(test, return_distance=True)
            for (d, i, i1) in zip(dist, ind, ind1):
                j = d.argsort()
                d[:] = d[j]
                i[:] = i[j]
                i1[:] = i1[j]
            results.append((dist, ind))

            assert_array_almost_equal(np.concatenate(list(ind)),
                                      np.concatenate(list(ind1)))

        for i in range(len(results) - 1):
            assert_array_almost_equal(np.concatenate(list(results[i][0])),
                                      np.concatenate(list(results[i + 1][0]))),
            assert_array_almost_equal(np.concatenate(list(results[i][1])),
                                      np.concatenate(list(results[i + 1][1])))

    # test ANN only in l2 space
    results_ann = []
    for algorithm in APPROXIMATE_ALGORITHMS:
        neigh = neighbors.NearestNeighbors(radius=radius,
                                           algorithm=algorithm,
                                           algorithm_params={'n_candidates': 5},
                                           p=p)
        neigh.fit(X)

        if algorithm in ['hnsw']:
            with pytest.raises(ValueError):
                ind1 = neigh.radius_neighbors(test, return_distance=False)
            continue
        else:
            ind1 = neigh.radius_neighbors(test, return_distance=False)

        # sort the results: this is not done automatically for
        # radius searches
        dist, ind = neigh.radius_neighbors(test, return_distance=True)
        for (d, i, i1) in zip(dist, ind, ind1):
            j = d.argsort()
            d[:] = d[j]
            i[:] = i[j]
            i1[:] = i1[j]
        results_ann.append((dist, ind))

        assert_array_almost_equal(np.concatenate(list(ind)),
                                  np.concatenate(list(ind1)))

    for i in range(len(results_ann)):
        assert_array_almost_equal(np.concatenate(list(results[-1][0])),
                                  np.concatenate(list(results_ann[i][0]))),
        assert_array_almost_equal(np.concatenate(list(results[-1][1])),
                                  np.concatenate(list(results_ann[i][1])))


@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
def test_kneighbors_classifier(hubness_and_params,
                               n_samples=40,
                               n_features=5,
                               n_test_pts=10,
                               n_neighbors=5,
                               random_state=0):
    # Test k-neighbors classification
    hubness, hubness_params = hubness_and_params
    hubness_params['k'] = 1
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < .5).astype(np.int)
    y_str = y.astype(str)

    weight_func = _weight_func

    for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
        for weights in ['uniform', 'distance', weight_func]:
            knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                                 weights=weights,
                                                 hubness=hubness, hubness_params=hubness_params,
                                                 algorithm_params={'n_candidates': 5},
                                                 algorithm=algorithm)
            knn.fit(X, y)
            epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
            y_pred = knn.predict(X[:n_test_pts] + epsilon)
            assert_array_equal(y_pred, y[:n_test_pts])
            # Test prediction with y_str
            knn.fit(X, y_str)
            y_pred = knn.predict(X[:n_test_pts] + epsilon)
            assert_array_equal(y_pred, y_str[:n_test_pts])


@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
def test_kneighbors_classifier_float_labels(hubness_and_params,
                                            n_samples=40, n_features=5,
                                            n_test_pts=10, n_neighbors=5,
                                            random_state=0):
    # Test k-neighbors classification
    hubness, hubness_params = hubness_and_params
    hubness_params['k'] = 1
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < .5).astype(np.int)

    knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                         hubness=hubness, hubness_params=hubness_params,
                                         algorithm_params={'n_candidates': 5})
    knn.fit(X, y.astype(np.float))
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    assert_array_equal(y_pred, y[:n_test_pts])


def test_kneighbors_classifier_predict_proba():
    # Test KNeighborsClassifier.predict_proba() method
    X = np.array([[0, 2, 0],
                  [0, 2, 1],
                  [2, 0, 0],
                  [2, 2, 0],
                  [0, 0, 2],
                  [0, 0, 1]])
    y = np.array([4, 4, 5, 5, 1, 1])
    cls = neighbors.KNeighborsClassifier(n_neighbors=3,
                                         algorithm_params={'n_candidates': 4},
                                         p=1)  # cityblock dist
    cls.fit(X, y)
    y_prob = cls.predict_proba(X)
    real_prob = np.array([[0, 2. / 3, 1. / 3],
                          [1. / 3, 2. / 3, 0],
                          [1. / 3, 0, 2. / 3],
                          [0, 1. / 3, 2. / 3],
                          [2. / 3, 1. / 3, 0],
                          [2. / 3, 1. / 3, 0]])
    assert_array_equal(real_prob, y_prob)
    # Check that it also works with non integer labels
    cls.fit(X, y.astype(str))
    y_prob = cls.predict_proba(X)
    assert_array_equal(real_prob, y_prob)
    # Check that it works with weights='distance'
    cls = neighbors.KNeighborsClassifier(
        n_neighbors=2,
        algorithm_params={'n_candidates': 3},
        p=1,
        weights='distance')
    cls.fit(X, y)
    y_prob = cls.predict_proba(np.array([[0, 2, 0],
                                         [2, 2, 2]]))
    real_prob = np.array([[0, 1, 0],
                          [0, 0.4, 0.6]])
    assert_array_almost_equal(real_prob, y_prob)


@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.filterwarnings('ignore:n_candidates > n_samples')
def test_radius_neighbors_classifier(hubness_and_params,
                                     n_samples=40,
                                     n_features=5,
                                     n_test_pts=10,
                                     radius=0.5,
                                     random_state=0):
    # Test radius-based classification
    hubness, hubness_params = hubness_and_params
    hubness_params['k'] = 1
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < .5).astype(np.int)
    y_str = y.astype(str)

    weight_func = _weight_func

    for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
        for weights in ['uniform', 'distance', weight_func]:
            neigh = neighbors.RadiusNeighborsClassifier(radius=radius,
                                                        weights=weights,
                                                        hubness=hubness, hubness_params=hubness_params,
                                                        algorithm=algorithm)
            neigh.fit(X, y)
            epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
            if algorithm in ['hnsw']:
                with pytest.raises(ValueError):
                    y_pred = neigh.predict(X[:n_test_pts] + epsilon)
                continue
            else:
                y_pred = neigh.predict(X[:n_test_pts] + epsilon)
            assert_array_equal(y_pred, y[:n_test_pts])
            neigh.fit(X, y_str)
            y_pred = neigh.predict(X[:n_test_pts] + epsilon)
            assert_array_equal(y_pred, y_str[:n_test_pts])


@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.filterwarnings('ignore:n_candidates > n_samples')
def test_radius_neighbors_classifier_when_no_neighbors(hubness_and_params):
    # Test radius-based classifier when no neighbors found.
    # In this case it should raise an informative exception
    hubness, hub_params = hubness_and_params
    hub_params['k'] = 1
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    y = np.array([1, 2])
    radius = 0.1

    z1 = np.array([[1.01, 1.01], [2.01, 2.01]])  # no outliers
    z2 = np.array([[1.01, 1.01], [1.4, 1.4]])    # one outlier

    weight_func = _weight_func

    for outlier_label in [0, -1, None]:
        for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
            for weights in ['uniform', 'distance', weight_func]:
                rnc = neighbors.RadiusNeighborsClassifier
                clf = rnc(radius=radius, weights=weights, algorithm=algorithm,
                          hubness=hubness, hubness_params=hub_params,
                          outlier_label=outlier_label)
                clf.fit(X, y)
                if algorithm in ['hnsw']:
                    with pytest.raises(ValueError):
                        prediction = clf.predict(z1)
                    continue
                prediction = clf.predict(z1)
                assert_array_equal(np.array([1, 2]), prediction)
                if outlier_label is None:
                    assert_raises(ValueError, clf.predict, z2)


@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.filterwarnings('ignore:n_candidates > n_samples')
def test_radius_neighbors_classifier_outlier_labeling(hubness_and_params):
    # Test radius-based classifier when no neighbors found and outliers
    # are labeled.
    hub, params = hubness_and_params
    params['k'] = 1
    X = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [0.99, 0.99],
                  [0.98, 0.98],
                  [2.01, 2.01]])
    y = np.array([1, 2, 1, 1, 2])
    radius = 0.1

    z1 = np.array([[1.01, 1.01],
                   [2.01, 2.01]])  # no outliers
    z2 = np.array([[1.4, 1.4],
                   [1.01, 1.01],
                   [2.01, 2.01]])    # one outlier
    correct_labels1 = np.array([1, 2])
    correct_labels2 = np.array([-1, 1, 2])

    weight_func = _weight_func

    for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
        for weights in ['uniform', 'distance', weight_func]:
            clf = neighbors.RadiusNeighborsClassifier(radius=radius,
                                                      weights=weights,
                                                      algorithm=algorithm,
                                                      hubness=hub, hubness_params=params,
                                                      outlier_label=-1)
            clf.fit(X, y)
            if algorithm in ['hnsw']:
                assert_raises(ValueError, clf.predict, z1)
                continue
            assert_array_equal(correct_labels1, clf.predict(z1))
            assert_array_equal(correct_labels2, clf.predict(z2))


@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.filterwarnings('ignore:n_candidates > n_samples')
def test_radius_neighbors_classifier_zero_distance(hubness_and_params):
    # Test radius-based classifier, when distance to a sample is zero.
    hub, h_params = hubness_and_params
    h_params['k'] = 1
    X = np.array([[1.0, 1.0],
                  [2.0, 2.0]])
    y = np.array([1, 2])
    radius = 0.1

    z1 = np.array([[1.01, 1.01],
                   [2.0, 2.0]])
    correct_labels1 = np.array([1, 2])

    weight_func = _weight_func

    for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
        for weights in ['uniform', 'distance', weight_func]:
            clf = neighbors.RadiusNeighborsClassifier(radius=radius,
                                                      weights=weights,
                                                      hubness=hub, hubness_params=h_params,
                                                      algorithm=algorithm)
            clf.fit(X, y)
            if algorithm in ['hnsw']:
                assert_raises(ValueError, clf.predict, z1)
            else:
                assert_array_equal(correct_labels1, clf.predict(z1))


def test_neighbors_regressors_zero_distance():
    # Test radius-based regressor, when distance to a sample is zero.

    X = np.array([[1.0, 1.0],
                  [1.0, 1.0],
                  [2.0, 2.0],
                  [2.5, 2.5]])
    y = np.array([1.0, 1.5, 2.0, 0.0])
    radius = 0.2
    z = np.array([[1.1, 1.1],
                  [2.0, 2.0]])

    rnn_correct_labels = np.array([1.25, 2.0])

    knn_correct_unif = np.array([1.25, 1.0])
    knn_correct_dist = np.array([1.25, 2.0])

    for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
        # we don't test for weights=_weight_func since user will be expected
        # to handle zero distances themselves in the function.
        for weights in ['uniform', 'distance']:
            rnn = neighbors.RadiusNeighborsRegressor(radius=radius,
                                                     weights=weights,
                                                     algorithm=algorithm)
            rnn.fit(X, y)
            if algorithm in ['hnsw']:
                assert_raises(ValueError, rnn.predict, z)
            else:
                assert_array_almost_equal(rnn_correct_labels, rnn.predict(z))

        for weights, corr_labels in zip(['uniform', 'distance'],
                                        [knn_correct_unif, knn_correct_dist]):
            knn = neighbors.KNeighborsRegressor(n_neighbors=2,
                                                weights=weights,
                                                algorithm=algorithm)
            knn.fit(X, y)
            assert_array_almost_equal(corr_labels, knn.predict(z))


def test_radius_neighbors_boundary_handling():
    """Test whether points lying on boundary are handled consistently

    Also ensures that even with only one query point, an object array
    is returned rather than a 2d array.
    """

    X = np.array([[1.5],
                  [3.0],
                  [3.01]])
    radius = 3.0

    for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
        nbrs = neighbors.NearestNeighbors(radius=radius,
                                          algorithm_params={'n_candidates': 2},
                                          algorithm=algorithm,
                                          ).fit(X)
        if algorithm in ['hnsw']:
            assert_raises(ValueError, nbrs.radius_neighbors, [[0.0]])
            continue
        results = nbrs.radius_neighbors([[0.0]], return_distance=False)
        assert_equal(results.shape, (1,))
        assert_equal(results.dtype, object)
        assert_array_equal(results[0], [0, 1])


def test_RadiusNeighborsClassifier_multioutput():
    # Test k-NN classifier on multioutput data
    rng = check_random_state(0)
    n_features = 2
    n_samples = 40
    n_output = 3

    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, 3, (n_samples, n_output))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    weights = [None, 'uniform', 'distance', _weight_func]

    for algorithm, weights in product(EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS, weights):
        # skip
        if algorithm in ['hnsw']:
            with pytest.raises(ValueError):
                neighbors.RadiusNeighborsClassifier(weights=weights, algorithm=algorithm)\
                    .fit(X_train, y_train)\
                    .predict(X_test)
            return

        # Stack single output prediction
        y_pred_so = []
        for o in range(n_output):
            rnn = neighbors.RadiusNeighborsClassifier(weights=weights,
                                                      algorithm=algorithm)
            rnn.fit(X_train, y_train[:, o])
            y_pred_so.append(rnn.predict(X_test))

        y_pred_so = np.vstack(y_pred_so).T
        assert_equal(y_pred_so.shape, y_test.shape)

        # Multioutput prediction
        rnn_mo = neighbors.RadiusNeighborsClassifier(weights=weights,
                                                     algorithm=algorithm)
        rnn_mo.fit(X_train, y_train)
        y_pred_mo = rnn_mo.predict(X_test)

        assert_equal(y_pred_mo.shape, y_test.shape)
        assert_array_almost_equal(y_pred_mo, y_pred_so)


def test_kneighbors_classifier_sparse(n_samples=40,
                                      n_features=5,
                                      n_test_pts=10,
                                      n_neighbors=5,
                                      random_state=0):
    # Test k-NN classifier on sparse matrices
    # Like the above, but with various types of sparse matrices
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    X *= X > .2
    y = ((X ** 2).sum(axis=1) < .5).astype(np.int)

    for sparsemat in SPARSE_TYPES:
        knn = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors,
                                             algorithm_params={'n_candidates': n_neighbors},
                                             algorithm='auto')
        knn.fit(sparsemat(X), y)
        epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)
        for sparsev in SPARSE_TYPES + (np.asarray,):
            X_eps = sparsev(X[:n_test_pts] + epsilon)
            y_pred = knn.predict(X_eps)
            assert_array_equal(y_pred, y[:n_test_pts])


@pytest.mark.parametrize('verbose', [0, 1, 2, 3])
def test_KNeighborsClassifier_multioutput(verbose):
    # Test k-NN classifier on multioutput data
    rng = check_random_state(0)
    n_features = 5
    n_samples = 50
    n_output = 3

    X = rng.rand(n_samples, n_features)
    y = rng.randint(0, 3, (n_samples, n_output))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    weights = [None, 'uniform', 'distance', _weight_func]

    for algorithm, weights in product(EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS, weights):
        # Stack single output prediction
        y_pred_so = []
        y_pred_proba_so = []
        for o in range(n_output):
            knn = neighbors.KNeighborsClassifier(weights=weights,
                                                 algorithm=algorithm,
                                                 verbose=verbose)
            knn.fit(X_train, y_train[:, o])
            y_pred_so.append(knn.predict(X_test))
            y_pred_proba_so.append(knn.predict_proba(X_test))

        y_pred_so = np.vstack(y_pred_so).T
        assert_equal(y_pred_so.shape, y_test.shape)
        assert_equal(len(y_pred_proba_so), n_output)

        # Multioutput prediction
        knn_mo = neighbors.KNeighborsClassifier(weights=weights,
                                                algorithm=algorithm)
        knn_mo.fit(X_train, y_train)
        y_pred_mo = knn_mo.predict(X_test)

        assert_equal(y_pred_mo.shape, y_test.shape)
        assert_array_almost_equal(y_pred_mo, y_pred_so)

        # Check proba
        y_pred_proba_mo = knn_mo.predict_proba(X_test)
        assert_equal(len(y_pred_proba_mo), n_output)

        for proba_mo, proba_so in zip(y_pred_proba_mo, y_pred_proba_so):
            assert_array_almost_equal(proba_mo, proba_so)


@pytest.mark.parametrize('verbose', [0, 3])
def test_kneighbors_regressor(verbose,
                              n_samples=40,
                              n_features=5,
                              n_test_pts=10,
                              n_neighbors=3,
                              random_state=0):
    # Test k-neighbors regression
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()

    y_target = y[:n_test_pts]

    weight_func = _weight_func

    for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
        for weights in ['uniform', 'distance', weight_func]:
            knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors,
                                                weights=weights,
                                                algorithm=algorithm,
                                                verbose=verbose,
                                                )
            knn.fit(X, y)
            epsilon = 1E-5 * (2 * rng.rand(1, n_features) - 1)
            y_pred = knn.predict(X[:n_test_pts] + epsilon)
            assert np.all(abs(y_pred - y_target) < 0.3)


@pytest.mark.parametrize('algorithm', EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS)
@pytest.mark.parametrize('weights', [None, 'uniform'])
def test_KNeighborsRegressor_multioutput_uniform_weight(algorithm, weights):
    # Test k-neighbors in multi-output regression with uniform weight
    rng = check_random_state(0)
    n_features = 5
    n_samples = 40
    n_output = 4

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_output)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    knn = neighbors.KNeighborsRegressor(weights=weights,
                                        algorithm=algorithm)
    knn.fit(X_train, y_train)

    neigh_idx = knn.kneighbors(X_test, return_distance=False)
    y_pred_idx = np.array([np.mean(y_train[idx], axis=0)
                           for idx in neigh_idx])

    y_pred = knn.predict(X_test)

    assert_equal(y_pred.shape, y_test.shape)
    assert_equal(y_pred_idx.shape, y_test.shape)
    assert_array_almost_equal(y_pred, y_pred_idx)


@pytest.mark.parametrize('algorithm', EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS)
@pytest.mark.parametrize('weights', ['uniform', 'distance', _weight_func])
def test_kneighbors_regressor_multioutput(algorithm, weights,
                                          n_samples=40,
                                          n_features=5,
                                          n_test_pts=10,
                                          n_neighbors=3,
                                          random_state=0):
    # Test k-neighbors in multi-output regression
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()
    y = np.vstack([y, y]).T

    y_target = y[:n_test_pts]

    knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors,
                                        weights=weights,
                                        algorithm=algorithm)
    knn.fit(X, y)
    epsilon = 1E-5 * (2 * rng.rand(1, n_features) - 1)
    y_pred = knn.predict(X[:n_test_pts] + epsilon)
    assert_equal(y_pred.shape, y_target.shape)

    assert np.all(np.abs(y_pred - y_target) < 0.3)


def test_radius_neighbors_regressor(n_samples=40,
                                    n_features=3,
                                    n_test_pts=10,
                                    radius=0.5,
                                    random_state=0):
    # Test radius-based neighbors regression
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()

    y_target = y[:n_test_pts]

    weight_func = _weight_func

    for algorithm in EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS:
        for weights in ['uniform', 'distance', weight_func]:
            neigh = neighbors.RadiusNeighborsRegressor(radius=radius,
                                                       weights=weights,
                                                       algorithm=algorithm)
            neigh.fit(X, y)
            epsilon = 1E-5 * (2 * rng.rand(1, n_features) - 1)
            if algorithm in ['hnsw']:
                assert_raises(ValueError, neigh.predict, X[:n_test_pts] + epsilon)
                continue
            y_pred = neigh.predict(X[:n_test_pts] + epsilon)
            assert np.all(abs(y_pred - y_target) < radius / 2)

    # test that nan is returned when no nearby observations
    for weights in ['uniform', 'distance']:
        neigh = neighbors.RadiusNeighborsRegressor(radius=radius,
                                                   weights=weights,
                                                   algorithm='auto')
        neigh.fit(X, y)
        X_test_nan = np.full((1, n_features), -1.)
        empty_warning_msg = ("One or more samples have no neighbors "
                             "within specified radius; predicting NaN.")
        pred = assert_warns_message(UserWarning,
                                    empty_warning_msg,
                                    neigh.predict,
                                    X_test_nan)
        assert np.all(np.isnan(pred))


@pytest.mark.parametrize('algorithm',
                         list(EXACT_ALGORITHMS)
                         + [pytest.param('lsh',
                                         marks=pytest.mark.skipif(sys.platform == 'win32',
                                                                  reason='falconn does not support Windows')), ]
                         + [pytest.param('hnsw',
                                         marks=pytest.mark.xfail(reason="hnsw does not support radius queries")), ])
@pytest.mark.parametrize('weights', [None, 'uniform'])
def test_RadiusNeighborsRegressor_multioutput_with_uniform_weight(algorithm, weights):
    # Test radius neighbors in multi-output regression (uniform weight)

    rng = check_random_state(0)
    n_features = 5
    n_samples = 40
    n_output = 4

    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples, n_output)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    rnn = neighbors. RadiusNeighborsRegressor(weights=weights,
                                              algorithm=algorithm)
    rnn.fit(X_train, y_train)

    neigh_idx = rnn.radius_neighbors(X_test, return_distance=False)
    y_pred_idx = np.array([np.mean(y_train[idx], axis=0)
                           for idx in neigh_idx])

    y_pred_idx = np.array(y_pred_idx)
    y_pred = rnn.predict(X_test)

    assert_equal(y_pred_idx.shape, y_test.shape)
    assert_equal(y_pred.shape, y_test.shape)
    assert_array_almost_equal(y_pred, y_pred_idx)


@pytest.mark.parametrize('algorithm',
                         list(EXACT_ALGORITHMS)
                         + [pytest.param('lsh',
                                         marks=pytest.mark.skipif(sys.platform == 'win32',
                                                                  reason='falconn does not support Windows')), ]
                         + [pytest.param('hnsw', marks=pytest.mark.xfail(
                             reason="hnsw does not support radius queries")), ])
@pytest.mark.parametrize('weights', ['uniform', 'distance', _weight_func])
def test_RadiusNeighborsRegressor_multioutput(algorithm, weights,
                                              n_samples=40,
                                              n_features=5,
                                              n_test_pts=10,
                                              n_neighbors=3,
                                              random_state=0):
    # Test k-neighbors in multi-output regression with various weight
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = np.sqrt((X ** 2).sum(1))
    y /= y.max()
    y = np.vstack([y, y]).T

    y_target = y[:n_test_pts]

    rnn = neighbors.RadiusNeighborsRegressor(n_neighbors=n_neighbors,
                                             weights=weights,
                                             algorithm=algorithm)
    rnn.fit(X, y)
    epsilon = 1E-5 * (2 * rng.rand(1, n_features) - 1)
    y_pred = rnn.predict(X[:n_test_pts] + epsilon)

    assert_equal(y_pred.shape, y_target.shape)
    assert np.all(np.abs(y_pred - y_target) < 0.3)


@pytest.mark.parametrize('sparsemat', SPARSE_TYPES)
def test_kneighbors_regressor_sparse(sparsemat,
                                     n_samples=40,
                                     n_features=5,
                                     n_neighbors=5,
                                     random_state=0):
    # Test radius-based regression on sparse matrices
    # Like the above, but with various types of sparse matrices
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < .25).astype(np.int)

    knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors,
                                        algorithm='auto')
    knn.fit(sparsemat(X), y)

    knn_pre = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors,
                                            metric='precomputed')
    knn_pre.fit(pairwise_distances(X, metric='euclidean'), y)

    for sparsev in SPARSE_OR_DENSE:
        X2 = sparsev(X)
        assert np.mean(knn.predict(X2).round() == y) > 0.95

        X2_pre = sparsev(pairwise_distances(X, metric='euclidean'))
        if issparse(sparsev(X2_pre)):
            assert_raises(ValueError, knn_pre.predict, X2_pre)
        else:
            assert np.mean(knn_pre.predict(X2_pre).round() == y) > 0.95


@pytest.mark.parametrize('algorithm', EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS)
@pytest.mark.parametrize('hubness_algorithm_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.filterwarnings('ignore:invalid value encountered')
@pytest.mark.filterwarnings('ignore:divide by zero encountered')
def test_neighbors_iris(algorithm, hubness_algorithm_and_params):
    # Sanity checks on the iris dataset
    # Puts three points of each label in the plane and performs a
    # nearest neighbor query on points near the decision boundary.

    hubness, hubness_params = hubness_algorithm_and_params

    clf = neighbors.KNeighborsClassifier(n_neighbors=1,
                                         algorithm=algorithm,
                                         hubness=hubness,
                                         hubness_params=hubness_params,
                                         )
    clf.fit(iris.data, iris.target)
    y_pred = clf.predict(iris.data)
    if hubness == 'dsl' or (algorithm == 'hnsw' and hubness in ['mp']):
        # Spurious small errors occur
        assert np.mean(y_pred == iris.target) > 0.95, f'Below 95% accuracy'
    else:
        assert_array_equal(y_pred, iris.target)

    clf.set_params(n_neighbors=9, algorithm=algorithm)
    clf.fit(iris.data, iris.target)
    assert np.mean(clf.predict(iris.data) == iris.target) > 0.95

    rgs = neighbors.KNeighborsRegressor(n_neighbors=5, algorithm=algorithm)
    rgs.fit(iris.data, iris.target)
    assert_greater(np.mean(rgs.predict(iris.data).round() == iris.target),
                   0.95)


def test_neighbors_digits():
    # Sanity check on the digits dataset
    # the 'brute' algorithm has been observed to fail if the input
    # dtype is uint8 due to overflow in distance calculations.

    X = digits.data.astype('uint8')
    Y = digits.target
    (n_samples, n_features) = X.shape
    train_test_boundary = int(n_samples * 0.8)
    train = np.arange(0, train_test_boundary)
    test = np.arange(train_test_boundary, n_samples)
    (X_train, Y_train, X_test, Y_test) = X[train], Y[train], X[test], Y[test]

    clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    score_uint8 = clf.fit(X_train, Y_train).score(X_test, Y_test)
    score_float = clf.fit(X_train.astype(float, copy=False), Y_train).score(
        X_test.astype(float, copy=False), Y_test)
    assert_equal(score_uint8, score_float)


@pytest.mark.parametrize('algorithm', ['auto'] + list(APPROXIMATE_ALGORITHMS))
@pytest.mark.parametrize('hubness_and_params', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.filterwarnings('ignore:No ground truth available for hubness reduced')
def test_kneighbors_graph(algorithm, hubness_and_params):
    hubness, hubness_params = hubness_and_params
    hubness_params['k'] = 1

    # Test kneighbors_graph to build the k-Nearest Neighbor graph.
    X = np.array([[0, 1],
                  [1.01, 1.],
                  [2, 0]])

    # n_neighbors = 1
    A = neighbors.kneighbors_graph(X, 1, mode='connectivity',
                                   algorithm=algorithm,
                                   hubness=hubness, hubness_params=hubness_params,
                                   include_self=True,
                                   )
    assert_array_equal(A.toarray(), np.eye(A.shape[0]))

    A = neighbors.kneighbors_graph(X, 1, mode='distance',
                                   algorithm=algorithm,
                                   hubness=hubness, hubness_params=hubness_params,
                                   )
    if hubness is not None:
        warnings.warn(f'No ground truth available for hubness reduced kNN graph.')
    else:
        assert_array_almost_equal(
            A.toarray(),
            [[0.00, 1.01, 0.],
             [1.01, 0., 0.],
             [0.00, 1.40716026, 0.]])

    # n_neighbors = 2
    A = neighbors.kneighbors_graph(X, 2, mode='connectivity',
                                   algorithm=algorithm,
                                   hubness=hubness, hubness_params=hubness_params,
                                   include_self=True)
    assert_array_equal(
        A.toarray(),
        [[1., 1., 0.],
         [1., 1., 0.],
         [0., 1., 1.]])

    A = neighbors.kneighbors_graph(X, 2, mode='distance',
                                   algorithm=algorithm,
                                   hubness=hubness, hubness_params=hubness_params,
                                   )
    if hubness is not None:
        warnings.warn(f'No ground truth available for hubness reduced kNN graph.')
    else:
        assert_array_almost_equal(
            A.toarray(),
            [[0., 1.01, 2.23606798],
             [1.01, 0., 1.40716026],
             [2.23606798, 1.40716026, 0.]])

    # n_neighbors = 3
    A = neighbors.kneighbors_graph(X, 3, mode='connectivity',
                                   algorithm=algorithm,
                                   hubness=hubness, hubness_params=hubness_params,
                                   include_self=True,
                                   )
    assert_array_almost_equal(
        A.toarray(),
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


@pytest.mark.parametrize('n_neighbors', [1, 2, 3])
@pytest.mark.parametrize('mode', ["connectivity", "distance"])
def test_kneighbors_graph_sparse(n_neighbors, mode, seed=36):
    # Test kneighbors_graph to build the k-Nearest Neighbor graph for sparse input.
    rng = np.random.RandomState(seed)
    X = rng.randn(10, 10)
    Xcsr = csr_matrix(X)

    assert_array_almost_equal(
        neighbors.kneighbors_graph(X,
                                   n_neighbors,
                                   mode=mode).toarray(),
        neighbors.kneighbors_graph(Xcsr,
                                   n_neighbors,
                                   mode=mode).toarray())


def test_radius_neighbors_graph():
    # Test radius_neighbors_graph to build the Nearest Neighbor graph.
    X = np.array([[0, 1], [1.01, 1.], [2, 0]])

    A = neighbors.radius_neighbors_graph(X, 1.5, mode='connectivity',
                                         include_self=True)
    assert_array_equal(
        A.toarray(),
        [[1., 1., 0.],
         [1., 1., 1.],
         [0., 1., 1.]])

    A = neighbors.radius_neighbors_graph(X, 1.5, mode='distance')
    assert_array_almost_equal(
        A.toarray(),
        [[0., 1.01, 0.],
         [1.01, 0., 1.40716026],
         [0., 1.40716026, 0.]])


@pytest.mark.parametrize('n_neighbors', [1, 2, 3])
@pytest.mark.parametrize('mode', ["connectivity", "distance"])
def test_radius_neighbors_graph_sparse(n_neighbors, mode, seed=36):
    # Test radius_neighbors_graph to build the Nearest Neighbor graph
    # for sparse input.
    rng = np.random.RandomState(seed)
    X = rng.randn(10, 10)
    Xcsr = csr_matrix(X)

    assert_array_almost_equal(
        neighbors.radius_neighbors_graph(X,
                                         n_neighbors,
                                         mode=mode).toarray(),
        neighbors.radius_neighbors_graph(Xcsr,
                                         n_neighbors,
                                         mode=mode).toarray())


def test_neighbors_badargs():
    # Test bad argument values: these should all raise ValueErrors
    assert_raises(ValueError,
                  neighbors.NearestNeighbors,
                  algorithm='blah')

    X = rng.random_sample((10, 2))
    Xsparse = csr_matrix(X)
    X3 = rng.random_sample((10, 3))
    y = np.ones(10)

    for cls in (neighbors.KNeighborsClassifier,
                neighbors.RadiusNeighborsClassifier,
                neighbors.KNeighborsRegressor,
                neighbors.RadiusNeighborsRegressor):
        assert_raises(ValueError,
                      cls,
                      weights='blah')
        assert_raises(ValueError,
                      cls, p=-1)
        assert_raises(ValueError,
                      cls, algorithm='blah')

        nbrs = cls(algorithm='ball_tree', metric='haversine')
        assert_raises(ValueError,
                      nbrs.predict,
                      X)
        assert_raises(ValueError,
                      ignore_warnings(nbrs.fit),
                      Xsparse, y)

        nbrs = cls(metric='haversine', algorithm='brute')
        nbrs.fit(X3, y)
        assert_raise_message(ValueError,
                             "Haversine distance only valid in 2 dimensions",
                             nbrs.predict,
                             X3)

        nbrs = cls()
        assert_raises(ValueError,
                      nbrs.fit,
                      np.ones((0, 2)), np.ones(0))
        assert_raises(ValueError,
                      nbrs.fit,
                      X[:, :, None], y)
        nbrs.fit(X, y)
        assert_raises(ValueError,
                      nbrs.predict,
                      [[]])
        if (isinstance(cls(), neighbors.KNeighborsClassifier) or
                isinstance(cls(), neighbors.KNeighborsRegressor)):
            nbrs = cls(n_neighbors=-1)
            assert_raises(ValueError, nbrs.fit, X, y)

    nbrs = neighbors.NearestNeighbors(algorithm_params={'n_candidates': 9}).fit(X)

    assert_raises(ValueError, nbrs.kneighbors_graph, X, mode='blah')
    assert_raises(ValueError, nbrs.radius_neighbors_graph, X, mode='blah')


def test_neighbors_metrics(n_samples=20, n_features=3,
                           n_query_pts=2, n_neighbors=5):
    # Test computing the neighbors for various metrics
    # create a symmetric matrix
    V = rng.rand(n_features, n_features)
    VI = np.dot(V, V.T)

    metrics = [('euclidean', {}),
               ('manhattan', {}),
               ('minkowski', dict(p=1)),
               ('minkowski', dict(p=2)),
               ('minkowski', dict(p=3)),
               ('minkowski', dict(p=np.inf)),
               ('chebyshev', {}),
               ('seuclidean', dict(V=rng.rand(n_features))),
               ('wminkowski', dict(p=3, w=rng.rand(n_features))),
               ('mahalanobis', dict(VI=VI)),
               ('haversine', {})]
    algorithms = ['brute', 'ball_tree', 'kd_tree']
    X = rng.rand(n_samples, n_features)

    test = rng.rand(n_query_pts, n_features)

    for metric, metric_params in metrics:
        results = {}
        p = metric_params.pop('p', 2)
        for algorithm in algorithms:
            # KD tree doesn't support all metrics
            if (algorithm == 'kd_tree' and
                    metric not in neighbors.KDTree.valid_metrics):
                assert_raises(ValueError,
                              neighbors.NearestNeighbors,
                              algorithm=algorithm,
                              metric=metric, metric_params=metric_params)
                continue
            neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors,
                                               algorithm=algorithm,
                                               algorithm_params={'n_candidates': n_neighbors},
                                               metric=metric, p=p,
                                               metric_params=metric_params)

            # Haversine distance only accepts 2D data
            feature_sl = (slice(None, 2)
                          if metric == 'haversine' else slice(None))

            neigh.fit(X[:, feature_sl])
            results[algorithm] = neigh.kneighbors(test[:, feature_sl],
                                                  return_distance=True)

        assert_array_almost_equal(results['brute'][0], results['ball_tree'][0])
        assert_array_almost_equal(results['brute'][1], results['ball_tree'][1])
        if 'kd_tree' in results:
            assert_array_almost_equal(results['brute'][0],
                                      results['kd_tree'][0])
            assert_array_almost_equal(results['brute'][1],
                                      results['kd_tree'][1])


def test_callable_metric():

    def custom_metric(x1, x2):
        return np.sqrt(np.sum(x1 ** 2 + x2 ** 2))

    X = np.random.RandomState(42).rand(20, 2)
    nbrs1 = neighbors.NearestNeighbors(3, algorithm='auto',
                                       algorithm_params={'n_candidates': 19},
                                       metric=custom_metric)
    nbrs2 = neighbors.NearestNeighbors(3, algorithm='brute',
                                       algorithm_params={'n_candidates': 19},
                                       metric=custom_metric)

    nbrs1.fit(X)
    nbrs2.fit(X)

    dist1, ind1 = nbrs1.kneighbors(X)
    dist2, ind2 = nbrs2.kneighbors(X)

    assert_array_almost_equal(dist1, dist2)


def test_valid_brute_metric_for_auto_algorithm():
    X = rng.rand(12, 12)
    Xcsr = csr_matrix(X)

    # check that there is a metric that is valid for brute
    # but not ball_tree (so we actually test something)
    assert_in("cosine", VALID_METRICS['brute'])
    assert "cosine" not in VALID_METRICS['ball_tree']

    # Metric which don't required any additional parameter
    require_params = ['mahalanobis', 'wminkowski', 'seuclidean']
    for metric in VALID_METRICS['brute']:
        if metric != 'precomputed' and metric not in require_params:
            nn = neighbors.NearestNeighbors(n_neighbors=3,
                                            algorithm='auto',
                                            algorithm_params={'n_candidates': 3},
                                            metric=metric)
            if metric != 'haversine':
                nn.fit(X)
                nn.kneighbors(X)
            else:
                nn.fit(X[:, :2])
                nn.kneighbors(X[:, :2])
        elif metric == 'precomputed':
            X_precomputed = rng.random_sample((10, 4))
            Y_precomputed = rng.random_sample((3, 4))
            DXX = metrics.pairwise_distances(X_precomputed, metric='euclidean')
            DYX = metrics.pairwise_distances(Y_precomputed, X_precomputed,
                                             metric='euclidean')
            nb_p = neighbors.NearestNeighbors(n_neighbors=3,
                                              algorithm_params={'n_candidates': 3},
                                              )
            nb_p.fit(DXX)
            nb_p.kneighbors(DYX)

    for metric in VALID_METRICS_SPARSE['brute']:
        if metric != 'precomputed' and metric not in require_params:
            nn = neighbors.NearestNeighbors(n_neighbors=3, algorithm='auto',
                                            algorithm_params={'n_candidates': 3},
                                            metric=metric).fit(Xcsr)
            nn.kneighbors(Xcsr)

    # Metric with parameter
    VI = np.dot(X, X.T)
    list_metrics = [('seuclidean', dict(V=rng.rand(12))),
                    ('wminkowski', dict(w=rng.rand(12))),
                    ('mahalanobis', dict(VI=VI))]
    for metric, params in list_metrics:
        nn = neighbors.NearestNeighbors(n_neighbors=3, algorithm='auto',
                                        algorithm_params={'n_candidates': 3},
                                        metric=metric,
                                        metric_params=params).fit(X)
        nn.kneighbors(X)


def test_metric_params_interface():
    assert_warns(SyntaxWarning, neighbors.KNeighborsClassifier,
                 metric_params={'p': 3})


@pytest.mark.parametrize('algorithm', ['kd_tree', 'ball_tree'] + list(APPROXIMATE_ALGORITHMS))
@pytest.mark.parametrize('cls', [neighbors.KNeighborsClassifier, neighbors.KNeighborsRegressor])
def test_predict_sparse_ball_kd_tree(algorithm, cls):
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)
    y = rng.randint(0, 2, 5)
    nbrs = cls(1, algorithm=algorithm)
    nbrs.fit(X, y)
    assert_raises((ValueError, TypeError, ), nbrs.predict, csr_matrix(X))


def test_non_euclidean_kneighbors():
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)

    # Find a reasonable radius.
    dist_array = pairwise_distances(X).flatten()
    np.sort(dist_array)
    radius = dist_array[15]

    # Test kneighbors_graph
    for metric in ['manhattan', 'chebyshev']:
        nbrs_graph = neighbors.kneighbors_graph(
            X, 3, metric=metric, mode='connectivity',
            include_self=True).toarray()
        nbrs1 = neighbors.NearestNeighbors(3, metric=metric).fit(X)
        assert_array_equal(nbrs_graph, nbrs1.kneighbors_graph(X).toarray())

    # Test radiusneighbors_graph
    for metric in ['manhattan', 'chebyshev']:
        nbrs_graph = neighbors.radius_neighbors_graph(
            X, radius, metric=metric, mode='connectivity',
            include_self=True).toarray()
        nbrs1 = neighbors.NearestNeighbors(metric=metric, radius=radius).fit(X)
        assert_array_equal(nbrs_graph, nbrs1.radius_neighbors_graph(X).A)

    # Raise error when wrong parameters are supplied,
    X_nbrs = neighbors.NearestNeighbors(3, metric='manhattan')
    X_nbrs.fit(X)
    assert_raises(ValueError, neighbors.kneighbors_graph, X_nbrs, 3,
                  metric='euclidean')
    X_nbrs = neighbors.NearestNeighbors(radius=radius, metric='manhattan')
    X_nbrs.fit(X)
    assert_raises(ValueError, neighbors.radius_neighbors_graph, X_nbrs,
                  radius, metric='euclidean')


def check_object_arrays(nparray, list_check):
    for ind, ele in enumerate(nparray):
        assert_array_equal(ele, list_check[ind])


@pytest.mark.parametrize('algorithm', EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS)
def test_k_and_radius_neighbors_train_is_not_query(algorithm):
    # Test kneighbors et.al when query is not training data

    nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)

    X = [[0], [1]]
    nn.fit(X)
    test_data = [[2], [1]]

    # Test neighbors.
    dist, ind = nn.kneighbors(test_data)
    assert_array_equal(dist, [[1], [0]])
    assert_array_equal(ind, [[1], [1]])
    if algorithm in ['hnsw']:
        assert_raises(ValueError, nn.radius_neighbors, [[2], [1]], radius=1.5)
    else:
        dist, ind = nn.radius_neighbors([[2], [1]], radius=1.5)
        # sklearn does not guarantee sorted radius neighbors, but LSH sorts automatically,
        # so we make sure, that all results here are sorted
        dist_true = [[1], [0, 1]]
        ind_true = [[1], [1, 0]]
        for i, (distance, index) in enumerate(zip(dist, ind)):
            sort = np.argsort(distance)
            check_object_arrays(distance[sort], dist_true[i])
            check_object_arrays(index[sort], ind_true[i])

    # Test the graph variants.
    assert_array_equal(
        nn.kneighbors_graph(test_data).A, [[0., 1.], [0., 1.]])
    assert_array_equal(
        nn.kneighbors_graph([[2], [1]], mode='distance').A,
        np.array([[0., 1.], [0., 0.]]))
    if algorithm in ['hnsw']:
        assert_raises(ValueError, nn.radius_neighbors_graph, [[2], [1]], radius=1.5)
    else:
        rng = nn.radius_neighbors_graph([[2], [1]], radius=1.5)
        assert_array_equal(rng.A, [[0, 1], [1, 1]])


@pytest.mark.parametrize('algorithm', EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS)
def test_k_and_radius_neighbors_X_None(algorithm):
    # Test kneighbors et.al when query is None

    nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)

    X = [[0], [1]]
    nn.fit(X)

    dist, ind = nn.kneighbors()
    assert_array_equal(dist, [[1], [1]])
    assert_array_equal(ind, [[1], [0]])
    if algorithm in ['hnsw']:
        assert_raises(ValueError, nn.radius_neighbors, None, radius=1.5)
    else:
        dist, ind = nn.radius_neighbors(None, radius=1.5)
        check_object_arrays(dist, [[1], [1]])
        check_object_arrays(ind, [[1], [0]])

    # Test the graph variants.
    graphs = []
    graphs += [nn.kneighbors_graph(None), ]
    if algorithm in ['hnsw']:
        assert_raises(ValueError, nn.radius_neighbors_graph, None, radius=1.5)
    else:
        graphs += [nn.radius_neighbors_graph(None, radius=1.5), ]
    for graph in graphs:
        assert_array_equal(graph.A, [[0, 1], [1, 0]])
        assert_array_equal(graph.data, [1, 1])
        assert_array_equal(graph.indices, [1, 0])

    X = [[0, 1], [0, 1], [1, 1]]
    nn = neighbors.NearestNeighbors(n_neighbors=2, algorithm=algorithm)
    nn.fit(X)
    assert_array_equal(
        nn.kneighbors_graph().A,
        np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0]]))


@pytest.mark.parametrize('algorithm', EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS)
def test_k_and_radius_neighbors_duplicates(algorithm):
    # Test behavior of kneighbors when duplicates are present in query

    nn = neighbors.NearestNeighbors(n_neighbors=1, algorithm=algorithm)
    nn.fit([[0], [1]])

    # Do not do anything special to duplicates.
    kng = nn.kneighbors_graph([[0], [1]], mode='distance')
    assert_array_equal(
        kng.A,
        np.array([[0., 0.], [0., 0.]]))
    assert_array_equal(kng.data, [0., 0.])
    assert_array_equal(kng.indices, [0, 1])

    if algorithm in ['hnsw']:
        assert_raises(ValueError, nn.radius_neighbors, [[0], [1]], radius=1.5)
    else:
        dist, ind = [np.stack(x) for x in nn.radius_neighbors([[0], [1]], radius=1.5)]
        sort = np.argsort(dist)
        dist = np.take_along_axis(dist, sort, axis=1)
        ind = np.take_along_axis(ind, sort, axis=1)
        check_object_arrays(dist, [[0, 1], [0, 1]])
        check_object_arrays(ind, [[0, 1], [1, 0]])

        rng = nn.radius_neighbors_graph([[0], [1]], radius=1.5)
        assert_array_equal(rng.A, np.ones((2, 2)))

        rng = nn.radius_neighbors_graph([[0], [1]], radius=1.5,
                                        mode='distance')
        if algorithm in ['lsh']:
            assert_array_equal(rng.A, [[0, 1], [1, 0]])
            assert_array_equal(rng.indices, [0, 1, 1, 0])
            assert_array_equal(rng.data, [0, 1, 0, 1])
        else:
            assert_array_equal(rng.A, [[0, 1], [1, 0]])
            assert_array_equal(rng.indices, [0, 1, 0, 1])
            assert_array_equal(rng.data, [0, 1, 1, 0])

    # Mask the first duplicates when n_duplicates > n_neighbors.
    X = np.ones((3, 1))
    nn = neighbors.NearestNeighbors(n_neighbors=1)
    nn.fit(X)
    dist, ind = nn.kneighbors()
    assert_array_equal(dist, np.zeros((3, 1)))
    assert_array_equal(ind, [[1], [0], [1]])

    # Test that zeros are explicitly marked in kneighbors_graph.
    kng = nn.kneighbors_graph(mode='distance')
    assert_array_equal(
        kng.A, np.zeros((3, 3)))
    assert_array_equal(kng.data, np.zeros(3))
    assert_array_equal(kng.indices, [1., 0., 1.])
    assert_array_equal(
        nn.kneighbors_graph().A,
        np.array([[0., 1., 0.],
                  [1., 0., 0.],
                  [0., 1., 0.]]))


def test_include_self_neighbors_graph():
    # Test include_self parameter in neighbors_graph
    X = [[2, 3], [4, 5]]
    kng = neighbors.kneighbors_graph(X, 1, include_self=True).A
    kng_not_self = neighbors.kneighbors_graph(X, 1, include_self=False).A
    assert_array_equal(kng, [[1., 0.], [0., 1.]])
    assert_array_equal(kng_not_self, [[0., 1.], [1., 0.]])

    rng = neighbors.radius_neighbors_graph(X, 5.0, include_self=True).A
    rng_not_self = neighbors.radius_neighbors_graph(
        X, 5.0, include_self=False).A
    assert_array_equal(rng, [[1., 1.], [1., 1.]])
    assert_array_equal(rng_not_self, [[0., 1.], [1., 0.]])


@pytest.mark.parametrize('algorithm', EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS)
def test_same_knn_parallel(algorithm):
    X, y = datasets.make_classification(n_samples=30, n_features=5,
                                        n_redundant=0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = neighbors.KNeighborsClassifier(n_neighbors=3,
                                         algorithm=algorithm)
    clf.fit(X_train, y_train)
    y = clf.predict(X_test)
    dist, ind = clf.kneighbors(X_test)
    graph = clf.kneighbors_graph(X_test, mode='distance').toarray()

    clf.set_params(n_jobs=3)
    clf.fit(X_train, y_train)
    y_parallel = clf.predict(X_test)
    dist_parallel, ind_parallel = clf.kneighbors(X_test)
    graph_parallel = \
        clf.kneighbors_graph(X_test, mode='distance').toarray()

    assert_array_equal(y, y_parallel)
    assert_array_almost_equal(dist, dist_parallel)
    assert_array_equal(ind, ind_parallel)
    assert_array_almost_equal(graph, graph_parallel)


@pytest.mark.parametrize('algorithm', list(EXACT_ALGORITHMS)
                         + [pytest.param('lsh',
                                         marks=pytest.mark.skipif(sys.platform == 'win32',
                                                                  reason='falconn does not support Windows')), ]
                         + [pytest.param('hnsw', marks=pytest.mark.xfail(
                                         reason="hnsw does not support radius queries")),
                            ])
def test_same_radius_neighbors_parallel(algorithm):
    X, y = datasets.make_classification(n_samples=30, n_features=5,
                                        n_redundant=0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = neighbors.RadiusNeighborsClassifier(radius=10,
                                              algorithm=algorithm)
    clf.fit(X_train, y_train)
    y = clf.predict(X_test)
    dist, ind = clf.radius_neighbors(X_test)
    graph = clf.radius_neighbors_graph(X_test, mode='distance').toarray()

    clf.set_params(n_jobs=3)
    clf.fit(X_train, y_train)
    y_parallel = clf.predict(X_test)
    dist_parallel, ind_parallel = clf.radius_neighbors(X_test)
    graph_parallel = \
        clf.radius_neighbors_graph(X_test, mode='distance').toarray()

    assert_array_equal(y, y_parallel)
    for i in range(len(dist)):
        assert_array_almost_equal(dist[i], dist_parallel[i])
        assert_array_equal(ind[i], ind_parallel[i])
    assert_array_almost_equal(graph, graph_parallel)


@pytest.mark.parametrize('backend', JOBLIB_BACKENDS)
@pytest.mark.parametrize('algorithm', EXACT_ALGORITHMS + APPROXIMATE_ALGORITHMS)
def test_knn_forcing_backend(backend, algorithm):
    # Non-regression test which ensure the knn methods are properly working
    # even when forcing the global joblib backend.
    with parallel_backend(backend):
        X, y = datasets.make_classification(n_samples=30, n_features=5,
                                            n_redundant=0, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        clf = neighbors.KNeighborsClassifier(n_neighbors=3,
                                             algorithm=algorithm,
                                             n_jobs=3)
        clf.fit(X_train, y_train)
        if algorithm in ['lsh'] and backend in ['multiprocessing', 'loky']:
            # can't pickle _falconn.LSHConstructionParameters objects
            assert_raises((TypeError, PicklingError, ), clf.predict, X_test)
        else:
            clf.predict(X_test)
            clf.kneighbors(X_test)
            clf.kneighbors_graph(X_test, mode='distance').toarray()


def test_dtype_convert():
    classifier = neighbors.KNeighborsClassifier(n_neighbors=1)
    CLASSES = 15
    X = np.eye(CLASSES)
    y = [ch for ch in 'ABCDEFGHIJKLMNOPQRSTU'[:CLASSES]]

    result = classifier.fit(X, y).predict(X)
    assert_array_equal(result, y)


def test_sparse_metric_callable():
    def sparse_metric(x, y):  # Metric accepting sparse matrix input (only)
        assert issparse(x) and issparse(y)
        return x.dot(y.T).A.item()

    X = csr_matrix([  # Population matrix
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 0, 1, 0, 0]
    ])

    Y = csr_matrix([  # Query matrix
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1]
    ])

    nn = neighbors.NearestNeighbors(algorithm='brute', n_neighbors=2,
                                    metric=sparse_metric).fit(X)
    N = nn.kneighbors(Y, return_distance=False)

    # GS indices of nearest neighbours in `X` for `sparse_metric`
    gold_standard_nn = np.array([
        [2, 1],
        [2, 1]
    ])

    assert_array_equal(N, gold_standard_nn)


# ignore conversion to boolean in pairwise_distances
@ignore_warnings(category=DataConversionWarning)
def test_pairwise_boolean_distance():
    # Non-regression test for #4523
    # 'brute': uses scipy.spatial.distance through pairwise_distances
    # 'ball_tree': uses sklearn.neighbors.dist_metrics
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(6, 5))
    NN = neighbors.NearestNeighbors

    nn1 = NN(metric="jaccard", algorithm='brute').fit(X)
    nn2 = NN(metric="jaccard", algorithm='ball_tree').fit(X)
    assert_array_equal(nn1.kneighbors(X)[0], nn2.kneighbors(X)[0])
