# SPDX-License-Identifier: BSD-3-Clause

"""Unsupervised nearest neighbors learner"""

from .base import NeighborsBase
from .base import KNeighborsMixin
from .base import RadiusNeighborsMixin
from .base import UnsupervisedMixin


class NearestNeighbors(NeighborsBase, KNeighborsMixin,
                       RadiusNeighborsMixin, UnsupervisedMixin):
    """Unsupervised learner for implementing neighbor searches.

    Read more in the `scikit-learn User Guide
    <https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-neighbors>`_.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    radius : float, optional (default = 1.0)
        Range of parameter space to use by default for :meth:`radius_neighbors`
        queries.

    algorithm : {'auto', 'hnsw', 'lsh', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:

        - 'hnsw' will use :class:`HNSW`
        - 'lsh' will use :class:`LSH`
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    algorithm_params : dict, optional
        Override default parameters of the NN algorithm.
        For example, with algorithm='lsh' and algorithm_params={n_candidates: 100}
        one hundred approximate neighbors are retrieved with LSH.
        If parameter hubness is set, the candidate neighbors are further reordered
        with hubness reduction.
        Finally, n_neighbors objects are used from the (optionally reordered) candidates.

    # TODO add all supported hubness reduction methods
    hubness : {'mutual_proximity', 'local_scaling', 'dis_sim_local', None}, optional
        Hubness reduction algorithm
        - 'mutual_proximity' or 'mp' will use :class:`MutualProximity'
        - 'local_scaling' or 'ls' will use :class:`LocalScaling`
        - 'dis_sim_local' or 'dsl' will use :class:`DisSimLocal`
        If None, no hubness reduction will be performed (=vanilla kNN).

    hubness_params: dict, optional
        Override default parameters of the selected hubness reduction algorithm.
        For example, with hubness='mp' and hubness_params={'method': 'normal'}
        a mutual proximity variant is used, which models distance distributions
        with independent Gaussians.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or callable, default 'minkowski'
        metric to use for distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.

    p : integer, optional (default = 2)
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        See `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_ for more details.

    Examples
    --------
      >>> import numpy as np
      >>> from skhubness.neighbors import NearestNeighbors
      >>> samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]

      >>> neigh = NearestNeighbors(2, 0.4)
      >>> neigh.fit(samples)  #doctest: +ELLIPSIS
      NearestNeighbors(...)

      >>> neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
      ... #doctest: +ELLIPSIS
      array([[2, 0]]...)

      >>> nbrs = neigh.radius_neighbors([[0, 0, 1.3]], 0.4, return_distance=False)
      >>> np.asarray(nbrs[0][0])
      array(2)

    See also
    --------
    KNeighborsClassifier
    RadiusNeighborsClassifier
    KNeighborsRegressor
    RadiusNeighborsRegressor
    BallTree

    Notes
    -----
    See `Nearest Neighbors <https://scikit-learn.org/stable/modules/neighbors.html#neighbors>`_
    in the scikit-learn online documentation for a discussion
    of the choice of ``algorithm`` and ``leaf_size``.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
    """

    def __init__(self, n_neighbors=5, radius=1.0,
                 algorithm: str = 'auto', algorithm_params: dict = None,
                 hubness: str = None, hubness_params: dict = None,
                 leaf_size=30, metric='minkowski',
                 p=2, metric_params=None, n_jobs=None, **kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            hubness=hubness,
            hubness_params=hubness_params,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params, n_jobs=n_jobs, **kwargs)
