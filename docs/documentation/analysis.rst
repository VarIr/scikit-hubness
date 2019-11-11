==================
Hubness analysis
==================

You can use the ``skhubness.analysis`` subpackage
in order to assess whether your data is prone to hubness.
Currently, the ``Hubness`` class acts as a one-stop-shop for hubness estimation.

``Hubness`` provides several hubness measures,
that are all computed from a kNNG.
More specifically, hubness is measured from `k-occurrence`,
that is, how often does an object occur in the k-nearest neighbor lists of other objects
(reverse nearest neighbors).

Traditionally, hubness has been measured by the skewness of the k-occurrence histogram,
where skewness to the right indicates higher hubness (due to objects that appear very
often as nearest neighbors).

Recently, additional measures borrowed from inequality research have been proposed,
such as calculating the Robin Hood index or Gini index from k-occurrences,
which may have more desirable features w.r.t to large datasets and interpretability.

The ``Hubness`` class provides a variety of these measures.
It is based on ``sklearn.BaseEstimator``, and thus follows scikit-learn principles.
When a new instance is created, sensible default parameters are used,
unless specific choices are made.
Typically, the user may want to choose a parameter ``k`` to define the size
of nearest neighbor lists, or ``metric``, in case the default Euclidean distances
do not fit the data well.
Parameter ``return_value`` defines which hubness measures to use.
``skhubness.analysis.VALID_HUBNESS_MEASURES`` provides a list of available measures.
If ``return_values=='all'``, all available measures are computed.

The ``algorithm`` parameter defines how to compute the kNN graph.
This is especially relevant for large datasets, as it provides more efficient index
structures and approximate nearest neighbor algorithms.
For example, ``algorithm='hnsw'`` uses a hierarchical navigable small-world graph
to compute the hubness measures in log-linear time (instead of quadratic).

``Hubness`` uses ``fit`` and ``score`` methods to estimate hubness.
In this fictional example, we estimate hubness in terms of the Robin Hood index in some large dataset:

.. code-block:: python

    >>> X = (some large dataset)
    >>> hub = Hubness(k=10,
    >>>               return_value='robinhood',
    >>>               algorithm='hnsw')
    >>> hub.fit(X)  # Creates the HNSW index
    >>> hub.score()
    0.56

A Robin Hood index of 0.56 indicates,
that 56% of all slots in nearest neighbor lists would need to be redistributed,
in order to obtain equal k-occurrence for all objects.
We'd consider this rather high hubness.

In order to evaluate, whether hubness reduction might be beneficial
for downstream tasks (learning etc.),
we can perform the same estimation with hubness reduction enabled.
We use the same code as above, but add the ``hubness`` parameter:

.. code-block:: python
    :emphasize-lines: 5,8

    >>> X = (some large dataset)
    >>> hub = Hubness(k=10,
    >>>               return_value='robinhood',
    >>>               algorithm='hnsw',
    >>>               hubness='local_scaling')
    >>> hub.fit(X)
    >>> hub.score()
    0.35

Here, the hubness reduction method `local scaling` resulted in a markedly lower
Robin Hood index.

Note, that we used the complete data set ``X`` in the examples above.
We can also split the data into some ``X_train`` and ``X_test``:

.. code-block:: python3

    >>> hub.fit(X_train)
    >>> hub.score(X_test)
    0.36

This is useful, when you want to tune hyperparameters towards
low hubness, and prevent data leakage.
