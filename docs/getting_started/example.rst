===================
Quick start example
===================

Users of ``scikit-hubness`` typically want to

1. analyse, whether their data show hubness
2. reduce hubness
3. perform learning (classification, regression, ...)
4. or simply perform fast approximate nearest neighbor search regardless of hubness

The following example shows all these steps for an example dataset
from the text domain (dexter).
Please make sure you have installed ``scikit-hubness``
(`installation instructions <installation.html>`_).

First, we load the dataset and inspect its size.

.. code-block:: python

    from skhubness.data import load_dexter
    X, y = load_dexter()
    print(f'X.shape = {X.shape}, y.shape={y.shape}')

Dexter is embedded in a high-dimensional space,
and could, thus, be prone to hubness.
Therefore, we assess the actual degree of hubness.

.. code-block:: python

    from skhubness import Hubness
    hub = Hubness(k=10, metric='cosine')
    hub.fit(X)
    k_skew = hub.score()
    print(f'Skewness = {k_skew:.3f}')


As a rule-of-thumb, skewness > 1.2 indicates significant hubness.
Additional hubness indices are available, for example:

.. code-block:: python

    hub = Hubness(k=10, return_value="all", metric='cosine')
    scores = hub.fit(X).score()
    print(f'Robin hood index:   {scores.get("robinhood"):.3f}')
    print(f'Antihub occurrence: {scores.get("antihub_occurrence"):.3f}')
    print(f'Hub occurrence:     {scores.get("hub_occurrence"):.3f}')


There is considerable hubness in dexter.
Let's see, whether hubness reduction can improve
kNN classification performance.

.. code-block:: python

    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer

    from skhubness.neighbors import NMSlibTransformer
    from skhubness.reduction import MutualProximity


    knn = KNeighborsTransformer(n_neighbors=50, metric="cosine")
    # Alternatively, create an approximate KNeighborsTransformer, e.g.,
    # knn = NMSlibTransformer(n_neighbors=50, metric="cosine")
    kneighbors_graph = knn.fit_transform(X, y)

    # vanilla kNN without hubness reduction
    clf = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
    acc_standard = cross_val_score(clf, kneighbors_graph, y, cv=5)

    # kNN with hubness reduction (mutual proximity) reuses the
    # precomputed graph and works in sklearn workflows:
    mp = MutualProximity(method="normal")
    mp_graph = mp.fit_transform(kneighbors_graph)
    acc_mp = cross_val_score(clf, mp_graph, y, cv=5)

    print(f'Accuracy (vanilla kNN): {acc_standard.mean():.3f}')
    print(f'Accuracy (kNN with hubness reduction): {acc_mp.mean():.3f}')


Accuracy was considerably improved by mutual proximity (MP).
But did MP actually reduce hubness?

.. code-block:: python

    mp_scores = hub.fit(mp_graph).score()
    print(f'k-skewness after MP: {mp_scores.get("k_skewness"):.3f} '
          f'(reduction of {scores.get("k_skewness") - mp_scores.get("k_skewness"):.3f})')
    print(f'Robinhood after MP:  {mp_scores.get("robinhood"):.3f} '
          f'(reduction of {scores.get("robinhood") - mp_scores.get("robinhood"):.3f})')

Yes!

The neighbor graphs can be reused for various purposes, like classification, hubness estimation,
hubness reduction, etc. This avoids expensive re-calculation for each individual step.
