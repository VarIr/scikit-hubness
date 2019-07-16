# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np

__all__ = ['load_dexter']


def load_dexter() -> (np.ndarray, np.ndarray):
    """Load the example data set (dexter).

    Returns
    -------
    X, y : ndarray, ndarray
        Vector data, and class labels
    """
    n = 300
    dim = 20000

    # Read class labels
    dexter_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dexter')
    dexter_labels = os.path.join(dexter_path, 'dexter_train.labels')
    dexter_vectors = os.path.join(dexter_path, 'dexter_train.data')
    y = np.loadtxt(dexter_labels)

    # Read data
    X = np.zeros((n, dim))
    with open(dexter_vectors, mode='r') as fid:
        data = fid.readlines()
    row = 0
    for line in data:
        line = line.strip().split()  # line now contains pairs of dim:val
        for word in line:
            col, val = word.split(':')
            X[row][int(col) - 1] = int(val)
        row += 1

    return X, y
