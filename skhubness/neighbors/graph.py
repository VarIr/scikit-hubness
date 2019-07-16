# SPDX-License-Identifier: BSD-3-Clause

"""Nearest Neighbors graph functions"""

# Author: Jake Vanderplas <vanderplas@astro.washington.edu>
#         Roman Feldbauer <roman.feldbauer@univie.ac.at>
#
# License of sklearn code: BSD 3 clause (C) INRIA, University of Amsterdam

from .base import KNeighborsMixin, RadiusNeighborsMixin
from .unsupervised import NearestNeighbors


def _check_params(X, metric, p, metric_params):
    """Check the validity of the input parameters"""
    params = zip(['metric', 'p', 'metric_params'],
                 [metric, p, metric_params])
    est_params = X.get_params()
    for param_name, func_param in params:
        if func_param != est_params[param_name]:
            raise ValueError(
                "Got %s for %s, while the estimator has %s for "
                "the same parameter." % (
                    func_param, param_name, est_params[param_name]))


def _query_include_self(X, include_self):
    """Return the query based on include_self param"""
    if include_self:
        query = X._fit_X
    else:
        query = None

    return query


def kneighbors_graph(X, n_neighbors, mode='connectivity',
                     algorithm: str = 'auto', algorithm_params: dict = None,
                     hubness: str = None, hubness_params: dict = None,
                     metric='minkowski', p=2, metric_params=None,
                     include_self=False, n_jobs=None):
    """Computes the (weighted) graph of k-Neighbors for points in X

    Read more in the `scikit-learn User Guide
    <https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-neighbors>`_.

    Parameters
    ----------
    X : array-like or BallTree, shape = [n_samples, n_features]
        Sample data, in the form of a numpy array or a precomputed
        :class:`BallTree`.

    n_neighbors : int
        Number of neighbors for each sample.

    mode : {'connectivity', 'distance'}, optional
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

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

    metric : string, default 'minkowski'
        The distance metric used to calculate the k-Neighbors for each sample
        point. The DistanceMetric class gives a list of available metrics.
        The default distance is 'euclidean' ('minkowski' metric with the p
        param equal to 2.)

    p : int, default 2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional
        additional keyword arguments for the metric function.

    include_self : bool, default=False.
        Whether or not to mark each sample as the first nearest neighbor to
        itself. If `None`, then True is used for mode='connectivity' and False
        for mode='distance' as this will preserve backwards compatibility.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        See `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_ for more details.

    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.

    Examples
    --------
    >>> X = [[0], [3], [1]]
    >>> from skhubness.neighbors import kneighbors_graph
    >>> A = kneighbors_graph(X, 2, mode='connectivity', include_self=True)
    >>> A.toarray()
    array([[1., 0., 1.],
           [0., 1., 1.],
           [1., 0., 1.]])

    See also
    --------
    radius_neighbors_graph
    """
    if not isinstance(X, KNeighborsMixin):
        knn = NearestNeighbors(n_neighbors,
                               algorithm=algorithm, algorithm_params=algorithm_params,
                               hubness=hubness, hubness_params=hubness_params,
                               metric=metric, p=p,
                               metric_params=metric_params,
                               n_jobs=n_jobs,
                               include_self=include_self,
                               )
        X = knn.fit(X)
    else:
        _check_params(X, metric, p, metric_params)

    query = _query_include_self(X, include_self)
    return X.kneighbors_graph(X=query, n_neighbors=n_neighbors, mode=mode)


def radius_neighbors_graph(X, radius, mode='connectivity',
                           algorithm: str = 'auto', algorithm_params: dict = None,
                           hubness: str = None, hubness_params: dict = None,
                           metric='minkowski', p=2, metric_params=None,
                           include_self=False, n_jobs=None):
    """Computes the (weighted) graph of Neighbors for points in X

    Neighborhoods are restricted the points at a distance lower than
    radius.

    Read more in the `scikit-learn User Guide
    <https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-neighbors>`_.

    Parameters
    ----------
    X : array-like or BallTree, shape = [n_samples, n_features]
        Sample data, in the form of a numpy array or a precomputed
        :class:`BallTree`.

    radius : float
        Radius of neighborhoods.

    mode : {'connectivity', 'distance'}, optional
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

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

    metric : string, default 'minkowski'
        The distance metric used to calculate the neighbors within a
        given radius for each sample point. The DistanceMetric class
        gives a list of available metrics. The default distance is
        'euclidean' ('minkowski' metric with the param equal to 2.)

    p : int, default 2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional
        additional keyword arguments for the metric function.

    include_self : bool, default=False
        Whether or not to mark each sample as the first nearest neighbor to
        itself. If `None`, then True is used for mode='connectivity' and False
        for mode='distance' as this will preserve backwards compatibility.

    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    Returns
    -------
    A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j.

    Examples
    --------
    >>> X = [[0], [3], [1]]
    >>> from skhubness.neighbors import radius_neighbors_graph
    >>> A = radius_neighbors_graph(X, 1.5, mode='connectivity',
    ...                            include_self=True)
    >>> A.toarray()
    array([[1., 0., 1.],
           [0., 1., 0.],
           [1., 0., 1.]])

    See also
    --------
    kneighbors_graph
    """
    if not isinstance(X, RadiusNeighborsMixin):
        X = NearestNeighbors(radius=radius,
                             algorithm=algorithm, algorithm_params=algorithm_params,
                             hubness=hubness, hubness_params=hubness_params,
                             metric=metric, p=p,
                             metric_params=metric_params, n_jobs=n_jobs).fit(X)
    else:
        _check_params(X, metric, p, metric_params)

    query = _query_include_self(X, include_self)
    return X.radius_neighbors_graph(query, radius, mode)
