========================================================
Nearest neighbors
========================================================

The :mod:`skhubness.neighbors` subpackage provides several neighbors-based learning methods.
It is designed as a drop-in replacement for scikit-learn's ``neighbors``.
The package provides all functionality from ``sklearn.neighbors``,
and adds support for transparent hubness reduction, where applicable, including

- classification (e.g. :class:`KNeighborsClassifier <skhubness.neighbors.KNeighborsClassifier>`),
- regression (e.g. :class:`RadiusNeighborsRegressor <skhubness.neighbors.RadiusNeighborsRegressor>`),
- unsupervised learning (e.g. :class:`NearestNeighbors <skhubness.neighbors.NearestNeighbors>`),
- outlier detection (:class:`LocalOutlierFactor <skhubness.neighbors.LocalOutlierFactor>`), and
- kNN graphs (:meth:`kneighbors_graph <skhubness.neighbors.kneighbors_graph>`).

In addition, scikit-hubness provides approximate nearest neighbor (ANN) search,
in order to support large data sets with millions of data objects and more.
A list of currently provided ANN methods is available
:ref:`here <Approximate nearest neighbor search methods>`.

Hubness reduction and ANN search can be used independently or in conjunction,
the latter yielding `approximate hubness reduction`.
User of scikit-learn will find that only minor modification of their code
is required to enable one or both of the above.
We describe how to do so :ref:`here <The scikit-hubness package>`.

For general information and details about nearest neighbors,
we refer to the excellent scikit-learn
`User Guide on Nearest Neighbors <https://scikit-learn.org/stable/modules/neighbors.html>`__.
