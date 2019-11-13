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
while artists that (unknowingly) producing antihub songs, are subject to financial losses.


The scikit-hubness package
--------------------------

The ``scikit-hubness`` package builds upon ``scikit-learn``.
When feasible, their design decisions, code style, development practise etc. are
adopted, so that new users can work their way into ``scikit-hubness`` rapidly.
Workflows, therefore, comprise the well-known ``fit``, ``predict``, and ``score`` methods.

Two subpackages offer orthogonal functionality to ``scikit-learn``:

- ``skhubness.analysis`` allows to estimate hubness in data
- ``skhubness.reduction`` provides hubness reduction algorithms

The ``skhubness.neighbors`` subpackages, on the other hand, acts as a drop-in
replacement for ``sklearn.neighbors``. It provides all of its functionality,
and adds two major components:

- transparent hubness reduction
- approximate nearest neighbor (ANN) search

and combinations of both. From the coding point-of-view,
this is achieved by adding a handful new parameters to most classes
(``KNeighborsClassifier``, ``RadiusNeighborRegressor``, ``NearestNeighbors``, etc).

- ``hubness`` defines the hubness reduction algorithm used to compute the kNNG.
  Supported values are available in ``skhubness.reduction.hubness_algorihtms``.
- ``algorithm`` defines the kNNG construction algorithm similarly to the
  way ``sklearn`` does it. That is, all of ``sklearn``'s algorithms are available,
  but in addition, several approximate nearest neighbor algorithms are provided as well.
  See below, for a list of currently supported values.

Both of the above select algorithms, most of which can be further tuned by
individual parameters.
These are not explicitly made accessible in high-level classes  like ``KNeighborsClassifier``,
in order to avoid very long lists of parameters,
because they differ from algorithm to algorithm.
Instead, two dictionaries

- ``hubness_params`` and
- ``algorithm_params``

are available in all high-level classes to set the nested parameters
for ANN and hubness reduction.


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
most of the classes from ``skhubness.neighbors`` or ``skhubness.Hubness``:

- 'hnsw' uses `hierarchical navigable small-world graphs` (provided by the ``nmslib`` library)
  in the wrapper class :class:`HNSW`.
- 'lsh' uses `locality sensitive hashing` (provided by the  ``puffinn`` library)
  in the wrapper class :class:`PuffinnLSH`.
- 'falconn_lsh' uses `locality sensitive hashing` (provided by the ``falconn`` library)
  in the wrapper class :class:`FalconnLSH`.
- 'nng' uses ANNG or ONNG (provided by the ``NGT`` library)
  in the wrapper class :class:`NNG`.
- 'rptree' uses the ``annoy`` library provided in the wrapper class :class:`Annoy`.

Configure parameters of the chosen algorithm with ``algorithm_params``.
This dictionary is passed to the corresponding wrapper class.
Take a look at their documentation in order to see, which parameters are available
for each individual class.

ANN can be combined with providing a ``hubness`` parameter in order to obtain
approximate hubness reduction.


Hubness reduction methods
-------------------------

Set the parameter ``hubness`` to one of the following identifiers
in order to use the corresponding hubness reduction algorithm:

- 'mp' or 'mutual_proximity' use `mutual proximity` (Gaussian or empiric distribution)
- 'ls' or 'local_scaling' use `local scaling` or `NICDM`
- 'dsl' or 'dis_sim_local' use `DisSim Local`

Variants are set with the `hubness_params` dictionary.
