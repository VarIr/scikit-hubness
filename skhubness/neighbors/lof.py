# SPDX-License-Identifier: BSD-3-Clause

# Authors: Nicolas Goix <nicolas.goix@telecom-paristech.fr>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

from __future__ import annotations

import warnings
import numpy as np
from sklearn.base import OutlierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

from .base import NeighborsBase
from .base import KNeighborsMixin
from .base import UnsupervisedMixin


__all__ = ["LocalOutlierFactor"]


class LocalOutlierFactor(NeighborsBase, KNeighborsMixin, UnsupervisedMixin,
                         OutlierMixin):
    """Unsupervised Outlier Detection using Local Outlier Factor (LOF)

    The anomaly score of each sample is called Local Outlier Factor.
    It measures the local deviation of density of a given sample with
    respect to its neighbors.
    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of
    its neighbors, one can identify samples that have a substantially lower
    density than their neighbors. These are considered outliers.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.

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

    hubness : {'mutual_proximity', 'local_scaling', 'dis_sim_local', None}, optional
        Hubness reduction algorithm

        - 'mutual_proximity' or 'mp' will use :class:`MutualProximity`
        - 'local_scaling' or 'ls' will use :class:`LocalScaling`
        - 'dis_sim_local' or 'dsl' will use :class:`DisSimLocal`

        If None, no hubness reduction will be performed (=vanilla kNN).

    hubness_params: dict, optional
        Override default parameters of the selected hubness reduction algorithm.
        For example, with hubness='mp' and hubness_params={'method': 'normal'}
        a mutual proximity variant is used, which models distance distributions
        with independent Gaussians.

    leaf_size: int, optional (default=30)
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.

    metric: string or callable, default 'minkowski'
        metric used for the distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.

        If 'precomputed', the training input X is expected to be a distance
        matrix.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics:
        https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    p: integer, optional (default=2)
        Parameter for the Minkowski metric from
        :func:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params: dict, optional (default=None)
        Additional keyword arguments for the metric function.

    contamination: 'auto' or float, optional (default='auto')
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the scores of the samples.

        - if 'auto', the threshold is determined as in the
          original paper,
        - if a float, the contamination should be in the range [0, 0.5].

        .. versionchanged:: 0.22
           The default value of ``contamination`` changed from 0.1
           to ``'auto'``.

    novelty: boolean, default False
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        that you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set.

    n_jobs: int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        See `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs/>`_ for more details.
        Affects only :meth:`kneighbors` and :meth:`kneighbors_graph` methods.


    Attributes
    ----------
    negative_outlier_factor_: numpy array, shape (n_samples,)
        The opposite LOF of the training samples. The higher, the more normal.
        Inliers tend to have a LOF score close to 1 (``negative_outlier_factor_``
        close to -1), while outliers tend to have a larger LOF score.

        The local outlier factor (LOF) of a sample captures its
        supposed 'degree of abnormality'.
        It is the average of the ratio of the local reachability density of
        a sample and those of its k-nearest neighbors.

    n_neighbors_: integer
        The actual number of neighbors used for :meth:`kneighbors` queries.

    offset_: float
        Offset used to obtain binary labels from the raw scores.
        Observations having a negative_outlier_factor smaller than `offset_`
        are detected as abnormal.
        The offset is set to -1.5 (inliers score around -1), except when a
        contamination parameter different than "auto" is provided. In that
        case, the offset is defined in such a way we obtain the expected
        number of outliers in training.

    References
    ----------
    .. [1] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000, May).
           LOF: identifying density-based local outliers. In ACM sigmod record.
    """
    def __init__(self, n_neighbors=20,
                 algorithm: str = 'auto', algorithm_params: dict = None,
                 hubness: str = None, hubness_params: dict = None,
                 leaf_size=30, metric='minkowski', p=2, metric_params=None,
                 contamination="auto", novelty=False, n_jobs=None):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            hubness=hubness,
            hubness_params=hubness_params,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params, n_jobs=n_jobs)
        self.contamination = contamination
        self.novelty = novelty

    @property
    def fit_predict(self):
        """"Fits the model to the training set X and returns the labels.

        Label is 1 for an inlier and -1 for an outlier according to the LOF
        score and the contamination parameter.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples.

        y: Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        is_inlier: array, shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """

        # As fit_predict would be different from fit.predict, fit_predict is
        # only available for outlier detection (novelty=False)

        if self.novelty:
            msg = ('fit_predict is not available when novelty=True. Use '
                   'novelty=False if you want to predict on the training set.')
            raise AttributeError(msg)

        return self._fit_predict

    def _fit_predict(self, X, y=None):
        """"Fits the model to the training set X and returns the labels.

        Label is 1 for an inlier and -1 for an outlier according to the LOF
        score and the contamination parameter.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples.

        Returns
        -------
        is_inlier: array, shape (n_samples,)
            Returns -1 for anomalies/outliers and 1 for inliers.
        """

        # As fit_predict would be different from fit.predict, fit_predict is
        # only available for outlier detection (novelty=False)

        return self.fit(X)._predict()

    def fit(self, X, y=None) -> LocalOutlierFactor:
        """Fit the model using X as training data.

        Parameters
        ----------
        X: {array-like, sparse matrix, BallTree, KDTree}
            Training data. If array or matrix, shape [n_samples, n_features],
            or [n_samples, n_samples] if metric='precomputed'.

        y: Ignored
            not used, present for API consistency by convention.

        Returns
        -------
        self: object
        """
        if self.contamination != 'auto':
            if not(0. < self.contamination <= .5):
                raise ValueError("contamination must be in (0, 0.5], "
                                 "got: %f" % self.contamination)

        super().fit(X)

        n_samples = self._fit_X.shape[0]
        if self.n_neighbors > n_samples:
            warnings.warn("n_neighbors (%s) is greater than the "
                          "total number of samples (%s). n_neighbors "
                          "will be set to (n_samples - 1) for estimation."
                          % (self.n_neighbors, n_samples))
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        self._distances_fit_X_, _neighbors_indices_fit_X_ = (
            self.kneighbors(None, n_neighbors=self.n_neighbors_))

        self._lrd = self._local_reachability_density(
            self._distances_fit_X_, _neighbors_indices_fit_X_)

        # Compute lof score over training samples to define offset_:
        lrd_ratios_array = (self._lrd[_neighbors_indices_fit_X_] /
                            self._lrd[:, np.newaxis])

        self.negative_outlier_factor_ = -np.mean(lrd_ratios_array, axis=1)

        if self.contamination == "auto":
            # inliers score around -1 (the higher, the less abnormal).
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(self.negative_outlier_factor_,
                                         100. * self.contamination)

        return self

    @property
    def predict(self):
        """Predict the labels (1 inlier, -1 outlier) of X according to LOF.

        This method allows to generalize prediction to *new observations* (not
        in the training set). Only available for novelty detection (when
        novelty is set to True).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples.

        Returns
        -------
        is_inlier: array, shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        if not self.novelty:
            msg = ('predict is not available when novelty=False, use '
                   'fit_predict if you want to predict on training data. Use '
                   'novelty=True if you want to use LOF for novelty detection '
                   'and predict on new unseen data.')
            raise AttributeError(msg)

        return self._predict

    def _predict(self, X=None):
        """Predict the labels (1 inlier, -1 outlier) of X according to LOF.

        If X is None, returns the same as fit_predict(X_train).

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features), default=None
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. to the training samples. If None, makes prediction on the
            training data without considering them as their own neighbors.

        Returns
        -------
        is_inlier: array, shape (n_samples,)
            Returns -1 for anomalies/outliers and +1 for inliers.
        """
        check_is_fitted(self, ["offset_", "negative_outlier_factor_",
                               "n_neighbors_", "_distances_fit_X_"])

        if X is not None:
            X = check_array(X, accept_sparse='csr')
            is_inlier = np.ones(X.shape[0], dtype=int)
            is_inlier[self.decision_function(X) < 0] = -1
        else:
            is_inlier = np.ones(self._fit_X.shape[0], dtype=int)
            is_inlier[self.negative_outlier_factor_ < self.offset_] = -1

        return is_inlier

    @property
    def decision_function(self):
        """Shifted opposite of the Local Outlier Factor of X.

        Bigger is better, i.e. large values correspond to inliers.

        The shift offset allows a zero threshold for being an outlier.
        Only available for novelty detection (when novelty is set to True).
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        Returns
        -------
        shifted_opposite_lof_scores: array, shape (n_samples,)
            The shifted opposite of the Local Outlier Factor of each input
            samples. The lower, the more abnormal. Negative scores represent
            outliers, positive scores represent inliers.
        """
        if not self.novelty:
            msg = ('decision_function is not available when novelty=False. '
                   'Use novelty=True if you want to use LOF for novelty '
                   'detection and compute decision_function for new unseen '
                   'data. Note that the opposite LOF of the training samples '
                   'is always available by considering the '
                   'negative_outlier_factor_ attribute.')
            raise AttributeError(msg)

        return self._decision_function

    def _decision_function(self, X):
        """Shifted opposite of the Local Outlier Factor of X.

        Bigger is better, i.e. large values correspond to inliers.

        The shift offset allows a zero threshold for being an outlier.
        Only available for novelty detection (when novelty is set to True).
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        Returns
        -------
        shifted_opposite_lof_scores: array, shape (n_samples,)
            The shifted opposite of the Local Outlier Factor of each input
            samples. The lower, the more abnormal. Negative scores represent
            outliers, positive scores represent inliers.
        """

        return self._score_samples(X) - self.offset_

    @property
    def score_samples(self):
        """Opposite of the Local Outlier Factor of X.

        It is the opposite as bigger is better, i.e. large values correspond
        to inliers.

        Only available for novelty detection (when novelty is set to True).
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.
        The score_samples on training data is available by considering the
        the ``negative_outlier_factor_`` attribute.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        Returns
        -------
        opposite_lof_scores: array, shape (n_samples,)
            The opposite of the Local Outlier Factor of each input samples.
            The lower, the more abnormal.
        """
        if not self.novelty:
            msg = ('score_samples is not available when novelty=False. The '
                   'scores of the training samples are always available '
                   'through the negative_outlier_factor_ attribute. Use '
                   'novelty=True if you want to use LOF for novelty detection '
                   'and compute score_samples for new unseen data.')
            raise AttributeError(msg)

        return self._score_samples

    def _score_samples(self, X):
        """Opposite of the Local Outlier Factor of X.

        It is the opposite as bigger is better, i.e. large values correspond
        to inliers.

        Only available for novelty detection (when novelty is set to True).
        The argument X is supposed to contain *new data*: if X contains a
        point from training, it considers the later in its own neighborhood.
        Also, the samples in X are not considered in the neighborhood of any
        point.
        The score_samples on training data is available by considering the
        the ``negative_outlier_factor_`` attribute.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The query sample or samples to compute the Local Outlier Factor
            w.r.t. the training samples.

        Returns
        -------
        opposite_lof_scores: array, shape (n_samples,)
            The opposite of the Local Outlier Factor of each input samples.
            The lower, the more abnormal.
        """
        check_is_fitted(self, ["offset_", "negative_outlier_factor_",
                               "_distances_fit_X_"])
        X = check_array(X, accept_sparse='csr')

        distances_X, neighbors_indices_X = (
            self.kneighbors(X, n_neighbors=self.n_neighbors_))
        X_lrd = self._local_reachability_density(distances_X,
                                                 neighbors_indices_X)

        lrd_ratios_array = (self._lrd[neighbors_indices_X] /
                            X_lrd[:, np.newaxis])

        # as bigger is better:
        return -np.mean(lrd_ratios_array, axis=1)

    def _local_reachability_density(self, distances_X, neighbors_indices):
        """The local reachability density (LRD)

        The LRD of a sample is the inverse of the average reachability
        distance of its k-nearest neighbors.

        Parameters
        ----------
        distances_X: array, shape (n_query, self.n_neighbors)
            Distances to the neighbors (in the training samples `self._fit_X`)
            of each query point to compute the LRD.

        neighbors_indices: array, shape (n_query, self.n_neighbors)
            Neighbors indices (of each query point) among training samples
            self._fit_X.

        Returns
        -------
        local_reachability_density: array, shape (n_samples,)
            The local reachability density of each sample.
        """
        dist_k = self._distances_fit_X_[neighbors_indices,
                                        self.n_neighbors_ - 1]
        reach_dist_array = np.maximum(distances_X, dist_k)

        # 1e-10 to avoid `nan' when nb of duplicates > n_neighbors_:
        divisor = (np.mean(reach_dist_array, axis=1) + 1e-10)
        return 1. / divisor
