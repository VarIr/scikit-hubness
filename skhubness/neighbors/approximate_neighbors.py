# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from multiprocessing import cpu_count
from typing import Union, Tuple
import warnings
import numpy as np


class ApproximateNearestNeighbor(ABC):
    """ Abstract base class for approximate nearest neighbor search methods.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "euclidean", "manhattan", "hamming", "dot"
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.
    """
    def __init__(self, n_candidates: int = 5, metric: str = 'sqeuclidean',
                 n_jobs: int = 1, verbose: int = 0, *args, **kwargs):
        self.n_candidates = n_candidates
        self.metric = metric
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs
        self.verbose = verbose

    @abstractmethod
    def fit(self, X, y=None):
        """ Setup ANN index from training data.

        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored
        """
        pass  # pragma: no cover

    @abstractmethod
    def kneighbors(self, X=None, n_candidates=None, return_distance=True) -> Union[Tuple[np.array, np.array], np.array]:
        """ Retrieve k nearest neighbors.

        Parameters
        ----------
        X: np.array or None, optional, default = None
            Query objects. If None, search among the indexed objects.
        n_candidates: int or None, optional, default = None
            Number of neighbors to retrieve.
            If None, use the value passed during construction.
        return_distance: bool, default = True
            If return_distance, will return distances and indices to neighbors.
            Else, only return the indices.
        """
        pass  # pragma: no cover


class UnavailableANN(ApproximateNearestNeighbor):
    """ Placeholder for ANN methods that are not available on specific platforms. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(f'The chosen approximate nearest neighbor method is not supported on your platform.')

    def fit(self, X, y=None):
        pass

    def kneighbors(self, X=None, n_candidates=None, return_distance=True):
        pass
