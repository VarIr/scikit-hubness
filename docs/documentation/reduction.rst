=================
Hubness reduction
=================

The :mod:`skhubness.reduction` subpackage provides several hubness reduction methods.
Currently, the supported methods are

- Mutual proximity (independent Gaussian distance distribution),
  provided by :class:`MutualProximity <skhubness.reduction.MutualProximity>` with ``method='normal'`` (default),
- Mutual proximity (empiric distance distribution),
  provided by :class:`MutualProximity <skhubness.reduction.MutualProximity>` with ``method='empiric'``,
- Local scaling,
  provided by :class:`LocalScaling <skhubness.reduction.LocalScaling>` with ``method='standard'`` (default),
- Non-iterative contextual dissimilarity measure,
  provided by :class:`LocalScaling <skhubness.reduction.LocalScaling>` with ``method='nicdm'``,
- DisSim Local,
  provided by :class:`DisSimLocal <skhubness.reduction.DisSimLocal>`,

which represent the most successful hubness reduction methods as identified in
our paper "A comprehensive empirical comparison of hubness reduction in high-dimensional spaces",
KAIS (2019), `DOI <https://doi.org/10.1007/s10115-018-1205-y>`__.
This survey paper also comes with an overview of how the individual methods work.

There are two ways to use perform hubness reduction in scikit-hubness:

- Implicitly, using the classes in :mod:`skhubness.neighbors`
  (see :ref:`User Guide: Nearest neighbors <Nearest neighbors>`),
- Explicitly, using the classes in :mod:`skhubness.reduction`.

The former is the common approach, if you simply want to improve your learning task
by hubness reduction. Most examples here also do so.
The latter may, however, be more useful for researchers, who would like to
investigate the hubness phenomenon itself.

All hubness reducers inherit from a common base class
:class:`HubnessReduction <skhubness.reduction.base.HubnessReduction>`.
This abstract class defines two important methods:
:meth:`fit <skhubness.reduction.base.HubnessReduction.fit>` and
:meth:`transform <skhubness.reduction.base.HubnessReduction.transform>`,
thus allowing to transform previously unseen data after the initial fit.
Most hubness reduction methods do not operate on vector data,
but manipulate pre-computed distances, in order to obtain `secondary distances`.
Therefore, ``fit`` and ``transform`` take neighbor graphs as input, instead of vectors.
Have a look at their signatures:

.. code-block:: Python3

    @abstractmethod
    def fit(self, neigh_dist, neigh_ind, X, assume_sorted, *args, **kwargs):
        pass  # pragma: no cover

    @abstractmethod
    def transform(self, neigh_dist, neigh_ind, X, assume_sorted, return_distance=True):
        pass  # pragma: no cover

The arguments ``neigh_dist`` and ``neigh_ind`` are two arrays representing the nearest neighbor graph
with shape ``(n_indexed, n_neighbors)`` during fit, and
shape ``(n_query, n_neighbors)`` during transform.
The i-th row in each array corresponds to the i-th object in the data set.
The j-th column in ``neigh_ind`` contains the index of one of the k-nearest neighbors among the indexed objects,
while the j-th column in ``neigh_dist`` contains the corresponding distance.
Note, that this is the same format as obtained by scikit-learn's ``kneighbors(return_distances=True)``
method.

This way, the user has full flexibility on how to calculate primary distances (Euclidean, cosine, KL divergence, etc).
:class:`DisSimLocal <skhubness.reduction.DisSimLocal>` (DSL) is the exception to this rule,
because it is formulated specifically for Euclidean distances.
DSL, therefore, also requires the training vectors in ``fit(..., X=X_train)``,
and the test set vectors in ``transform(..., X=X_test)``.
Argument ``X`` is ignored in the other hubness reduction methods.

When the neighbor graph is already sorted (lowest to highest distance),
``assume_sorted=True`` should be set, so that hubness reduction methods
will not sort the arrays again, thus saving computational time.

Hubness reduction methods transform the primary distance graph,
and return secondary distances.
Note that for efficiency reasons, the returned arrays are not sorted.
Please make sure to sort the arrays, if downstream tasks assume sorted arrays.
