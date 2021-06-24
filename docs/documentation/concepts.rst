=============
Core Concepts
=============

There are three main parts of ``scikit-hubness``. Before we describe the corresponding subpackages,
we briefly introduce the `hubness` phenomenon itself.


The hubness phenomenon
----------------------

`Hubness` is a phenomenon of intrinsically high-dimensional data,
detrimental to data mining and learning tasks.
It refers to the tendency of `hub` and `antihub` emergence in k-nearest neighbor graphs (kNNGs):
Hubs are objects that appear unwontedly often among the k-nearest neighbor lists of other objects,
while antihubs hardly or never appear in these lists.
Thus, hubs propagate their metainformation (such as class labels) widely within a kNNG.
Conversely, information carried by antihubs is effectively lost.
As a result, hubness leads to semantically distorted spaces,
that negatively impact a large variety of tasks.

Music information retrieval is a show-case example for hubness:
It has been shown, that recommendation lists based on music similarity scores
tend to completely ignore certain songs (`antihubs`).
On the other hand, different songs are recommended over and over again (`hubs`),
sometimes even when they do not fit.
Both effects are problematic: Users are provided with unsuitable (hub) recommendations,
while artists that (unknowingly) producing antihub songs, may remain fameless unjustifiably.


The scikit-hubness package
--------------------------

``scikit-hubness`` reflects our effort to make hubness analysis and
hubness reduction readily available and easy-to-use for both machine learning
researchers and practitioners.

The package builds upon ``scikit-learn``.
When feasible, their design decisions, code style, development practise etc. are
adopted, so that new users can work their way into ``scikit-hubness`` rapidly.
Workflows, therefore, comprise the well-known ``fit``, ``predict``, and ``score`` methods.

Two subpackages offer complementary functionality to ``scikit-learn``:

- :mod:`skhubness.analysis` allows to estimate hubness in data
- :mod:`skhubness.reduction` provides hubness reduction algorithms

The :mod:`skhubness.neighbors` subpackage, on the other hand, acts as a drop-in
replacement for ``sklearn.neighbors``. It provides all of its functionality,
and adds two major components:

- transparent hubness reduction
- approximate nearest neighbor (ANN) search

and combinations of both. From the coding point-of-view,
this is achieved by adding a handful new parameters to most classes
(:class:`KNeighborsClassifier <skhubness.neighbors.KNeighborsClassifier>`,
:class:`RadiusNeighborRegressor <skhubness.neighbors.RadiusNeighborsRegressor>`,
:class:`NearestNeighbors <skhubness.neighbors.NearestNeighbors>`,
etc).

- ``hubness`` defines the hubness reduction algorithm used to compute the nearest neighbor graph (kNNG).
  Supported algorithms and corresponding parameter values are presented :ref:`here <Hubness reduction methods>`,
  and are available as a Python list in :const:`<skhubness.reduction.hubness_algorithms>`.
- ``algorithm`` defines the kNNG construction algorithm similarly to the
  way ``sklearn`` does it. That is, all of ``sklearn``'s algorithms are available,
  but in addition, several approximate nearest neighbor algorithms are provided as well.
  :ref:`See below <Approximate nearest neighbor search methods>` for a list of
  currently supported algorithms and their corresponding parameter values.

By providing the two arguments above, you select algorithms
for hubness reduction and nearest neighbor search, respectively.
Most of these algorithms can be further tuned by individual hyperparameters.
These are not explicitly made accessible in high-level classes  like ``KNeighborsClassifier``,
in order to avoid very long lists of arguments,
because they differ from algorithm to algorithm.
Instead, two dictionaries

- ``hubness_params`` and
- ``algorithm_params``

are available in all high-level classes to set the nested arguments
for ANN and hubness reduction methods.


The following example shows how to perform approximate hubness estimation
(1) without, and (2) with hubness reduction by local scaling
in an artificial data set.

In part 1, we estimate hubness in the original data.

.. code-block:: python

    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1_000_000,
                               n_features=500,
                               n_informative=400,
                               random_state=123)

    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=456)

    from skhubness.analysis import Hubness
    hub = Hubness(k=10,
                       metric='euclidean',
                       algorithm='hnsw',
                       algorithm_params={'n_candidates': 100,
                                         'metric': 'euclidean',
                                         'post_processing': 2,
                                         },
                       return_value='robinhood',
                       n_jobs=8,
                       )
    hub.fit(X_train)
    robin_hood = hub.score(X_test)
    print(robin_hood)
    0.7873205555555555  # before hubness reduction

There is high hubness in this dataset. In part 2, we estimate hubness after reduction by local scaling.

.. code-block:: python
    :emphasize-lines: 3,4,16

    hub = Hubness(k=10,
                  metric='euclidean',
                  hubness='local_scaling',
                  hubness_params={'k': 7},
                  algorithm='hnsw',
                  algorithm_params={'n_candidates': 100,
                                    'metric': 'euclidean',
                                    'post_processing': 2,
                                   },
                  return_value='robinhood',
                  verbose=2
                  )
    hub.fit(X_train)
    robin_hood = hub.score(X_test)
    print(robin_hood)
    0.6614583333333331  # after hubness reduction


Approximate nearest neighbor search methods
-------------------------------------------

Set the parameter ``algorithm`` to one of the following in order to enable ANN in
most of the classes from :mod:`skhubness.neighbors` or :class:`Hubness <skhubness.analysis.Hubness>`:

- 'hnsw' uses `hierarchical navigable small-world graphs` (provided by the ``nmslib`` library)
  in the wrapper class :class:`LegacyHNSW <skhubness.neighbors.LegacyHNSW>`.
- 'lsh' uses `locality sensitive hashing` (provided by the  ``puffinn`` library)
  in the wrapper class :class:`PuffinnLSH <skhubness.neighbors.PuffinnLSH>`.
- 'falconn_lsh' uses `locality sensitive hashing` (provided by the ``falconn`` library)
  in the wrapper class :class:`FalconnLSH <skhubness.neighbors.FalconnLSH>`.
- 'nng' uses ANNG or ONNG (provided by the ``NGT`` library)
  in the wrapper class :class:`NNG <skhubness.neighbors.NNG>`.
- 'rptree' uses random projections trees (provided by the ``annoy`` library)
  in the wrapper class :class:`LegacyRandomProjectionTree <skhubness.neighbors.LegacyRandomProjectionTree>`.

Configure parameters of the chosen algorithm with ``algorithm_params``.
This dictionary is passed to the corresponding wrapper class.
Take a look at their documentation in order to see, which parameters are available
for each individual class.


Hubness reduction methods
-------------------------

Set the parameter ``hubness`` to one of the following identifiers
in order to use the corresponding hubness reduction algorithm:

- 'mp' or 'mutual_proximity' use `mutual proximity` (Gaussian or empiric distribution)
  as implemented in :class:`MutualProximity <skhubness.reduction.MutualProximity>`.
- 'ls' or 'local_scaling' use `local scaling` or `NICDM`
  as implemented in :class:`LocalScaling <skhubness.reduction.LocalScaling>`.
- 'dsl' or 'dis_sim_local' use `DisSim Local`
  as implemented in :class:`DisSimLocal <skhubness.reduction.DisSimLocal>`.

Variants and additional parameters are set with the ``hubness_params`` dictionary.
Have a look at the individual hubness reduction classes for available parameters.


Approximate hubness reduction
-----------------------------

*Exact* hubness reduction scales at least quadratically with the number of samples.
To reduce computational complexity, *approximate* hubness reduction can be applied,
as described in the paper "Fast approximate hubness reduction for large high-dimensional data"
(ICBK2018, `on IEEE Xplore <https://ieeexplore.ieee.org/document/8588814>`_,
also available as `technical report <http://www.ofai.at/cgi-bin/tr-online?number+2018-02>`_).

The general idea behind approximate hubness reduction works as follows:

#. retrieve ``n_candidates``-nearest neighbors using an ANN method
#. refine and reorder the candidate list by hubness reduction
#. return ``n_neighbors`` nearest neighbors from the reordered candidate list

The procedure is implemented in scikit-hubness by simply passing both
``algorithm`` and ``hubness`` parameters to the relevant classes.

Also consider passing ``algorithm_params={'n_candidates': n_candidates}``.
Make sure to set the ``n_candidates`` high enough, for high sensitivity
(towards "good" nearest neighbors). Too large values may, however, lead
to long query times. As a rule of thumb for this trade-off, you can
start by retrieving ten times as many candidates as you need nearest neighbors.
