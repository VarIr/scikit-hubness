from abc import abstractmethod


class ApproximateNearestNeighbor:
    """ Abstract base class for approximate nearest neighbor search methods. """

    def __init__(self, n_candidates: int = 5, metric: str = 'sqeuclidean',
                 n_jobs: int = 1, verbose: int = 0, *args, **kwargs):
        self.n_candidates = n_candidates
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def kneighbors(self, X, n_candidates, return_distance):
        pass
