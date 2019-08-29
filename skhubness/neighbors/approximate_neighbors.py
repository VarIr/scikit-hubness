# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from multiprocessing import cpu_count
import warnings


class ApproximateNearestNeighbor(ABC):
    """ Abstract base class for approximate nearest neighbor search methods. """

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
        pass  # pragma: no cover

    @abstractmethod
    def kneighbors(self, X=None, n_candidates=None, return_distance=True):
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
