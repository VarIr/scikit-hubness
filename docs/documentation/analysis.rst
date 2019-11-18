==================
Hubness analysis
==================

You can use the :mod:`skhubness.analysis` subpackage
to assess whether your data is prone to hubness.
Currently, the :class:`Hubness <skhubness.analysis.Hubness>` class
acts as a one-stop-shop for hubness estimation.
It provides several hubness measures,
that are all computed from a nearest neighbor graph (kNNG).
More specifically, hubness is measured from `k-occurrence`,
that is, how often does an object occur in the k-nearest neighbor lists of other objects
(reverse nearest neighbors).
Traditionally, hubness has been measured by the skewness of the k-occurrence histogram,
where higher skewness to the right indicates higher hubness (due to objects that appear very
often as nearest neighbors).
Recently, additional indices borrowed from inequality research have been proposed for measuring hubness,
such as calculating the Robin Hood index or Gini index from k-occurrences,
which may have more desirable features w.r.t to large datasets and interpretability.

The :class:`Hubness <skhubness.analysis.Hubness>` class provides a variety of these measures.
It is based on scikit-learn's ``BaseEstimator``, and thus follows scikit-learn principles.
When a new instance is created, sensible default parameters are used,
unless specific choices are made.
Typically, the user may want to choose a parameter ``k`` to define the size
of nearest neighbor lists, or ``metric``, in case the default Euclidean distances
do not fit the data well.
Parameter ``return_value`` defines which hubness measures to use.
:const:`VALID_HUBNESS_MEASURES <skhubness.analysis.VALID_HUBNESS_MEASURES>`
provides a list of available measures.
If ``return_values=='all'``, all available measures are computed.
The ``algorithm`` parameter defines how to compute the kNN graph.
This is especially relevant for large datasets, as it provides more efficient index
structures and approximate nearest neighbor algorithms.
For example, ``algorithm='hnsw'`` uses a hierarchical navigable small-world graph
to compute the hubness measures in log-linear time (instead of quadratic).

:class:`Hubness <skhubness.analysis.Hubness>` uses :meth:`fit <skhubness.analysis.Hubness.fit>`
and :meth:`score <skhubness.analysis.Hubness.score>` methods to estimate hubness.
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


Hubness measures
----------------

The degree of hubness in a dataset typically measured from its k-occurrence histogram :math:`O^k`.
For an individual data object **x**, its k-occurrence :math:`O^k(x)` is defined as the number of times
**x** resides among the *k*-nearest neighbors of all other objects in the data set.
In the notion of network analysis, :math:`O^k(x)` is the indegree of **x** in a directed kNN graph.
It is also known as reverse neighbor count.

The following measures are provided in :class:`Hubness <skhubness.analysis.Hubness>`
by passing the corresponding argument values:

- 'k_skewness': Skewness, the third central moment of the k-occurrence distribution,
  as introduced by `Radovanović et al. 2010 <http://www.jmlr.org/papers/v11/radovanovic10a.html>`_
- 'k_skewness_truncnorm': skewness of truncated normal distribution estimated from k-occurrence distribution.
- 'atkinson': the `Atkinson index <https://en.wikipedia.org/wiki/Atkinson_index>`_ of inequality,
  which can be tuned in order to be more sensitive towards antihub or hubs.
- 'gini': the `Gini coefficient <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of inequality,
  defined as the half of the relative mean absolute difference
- 'robinhood': the `Robin Hood or Hoover index <https://en.wikipedia.org/wiki/Hoover_index>`_,
  which gives the amount that needs to be redistributed in order to obtain equality
  (e.g. proportion of total income, so that there is equal income for all;
  or the number of nearest neighbor slot, so that all objects are among the k-nearest neighbors
  of others exactly k times).
- 'antihubs': returns the indices of antihubs in data set **X** (which are never
  among the nearest neighbors of other objects.
- 'antihub_occurrence': proportion of antihubs in the data set (percentage of total objects,
  which are antihubs).
- 'hubs':  indices of hub objects **x** in data set **X**
  (with :math:`O^k(x) > \text{hub_size} * k`, where :math:`\text{hub_size} = 2` by default).
- 'hub_occurrence': proportion of nearest neighbor slots occupied by hubs
- 'groupie_ratio': proportion of objects with the largest hub in their neighborhood
- 'k_neighbors': indices to k-nearest neighbors for each object
- 'k_occurrence': reverse neighbor count for each object
- 'all': return a dictionary containing all of the above
