# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
from scipy.sparse import csr_matrix
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
