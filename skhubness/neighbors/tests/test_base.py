# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from skhubness.neighbors import NearestNeighbors
from skhubness.neighbors.base import NeighborsBase, ANN_ALG, EXACT_ALG, VALID_METRICS, VALID_METRICS_SPARSE
from skhubness.reduction import hubness_algorithms, hubness_algorithms_long


@pytest.mark.parametrize('algo', hubness_algorithms + hubness_algorithms_long)
def test_check_hubness_accepts_valid_values(algo):
    NeighborsBase(hubness=algo)._check_hubness_algorithm()


@pytest.mark.parametrize('algo', ['auto', *EXACT_ALG, *ANN_ALG])
def test_check_algorithm_accepts_valid_values(algo):
    NeighborsBase(algorithm=algo)._check_algorithm_metric()


@pytest.mark.parametrize('algo', list(EXACT_ALG) + list(ANN_ALG))
def test_valid_metrics_for_algorithm(algo):
    VALID_METRICS[algo]
    VALID_METRICS_SPARSE[algo]


@pytest.mark.parametrize('hr', hubness_algorithms_long + hubness_algorithms)
def test_sparse_and_hubness_reduction_disables_hr_and_warns(hr):
    X = csr_matrix([[0, 0],
                    [0, 1],
                    [0, 3]])
    nn_true = [1, 0, 1]
    nn = NearestNeighbors(n_neighbors=1, hubness=hr, algorithm_params={'n_candidates': 1})
    msg = 'cannot use hubness reduction with sparse data: disabling hubness reduction.'
    with pytest.warns(UserWarning, match=msg):
        nn.fit(X)
    nn_pred = nn.kneighbors(n_neighbors=1, return_distance=False).ravel()
    np.testing.assert_array_equal(nn_true, nn_pred)
