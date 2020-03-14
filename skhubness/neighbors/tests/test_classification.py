# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.datasets import make_multilabel_classification
from skhubness.neighbors import KNeighborsClassifier


@pytest.mark.parametrize('n_jobs', [None, 1, 2])
@pytest.mark.parametrize('n_neighbors', [1, 5])
def test_sparse_multilabel_targets(n_neighbors, n_jobs):
    X, y_dense = make_multilabel_classification(random_state=123)
    thresh = 80

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                               n_jobs=n_jobs, )
    assert not issparse(y_dense)
    knn.fit(X[:thresh], y_dense[:thresh])
    y_pred = knn.predict(X[thresh:])

    y_sparse = csr_matrix(y_dense)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                               n_jobs=n_jobs,)
    assert issparse(y_sparse)
    knn.fit(X[:thresh], y_sparse[:thresh])
    y_pred_sparse = knn.predict(X[thresh:, :])

    # Test array equality
    np.testing.assert_array_equal(y_pred, y_pred_sparse.toarray())
