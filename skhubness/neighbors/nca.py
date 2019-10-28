# coding: utf-8
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.neighbors import NeighborhoodComponentsAnalysis

__all__ = ['NeighborhoodComponentsAnalysis']

nca_docs = NeighborhoodComponentsAnalysis.__doc__
old_str = 'Read more in the :ref:`User Guide <nca>`.'
new_str = ('Read more in the `scikit-learn User Guide '
           '<https://scikit-learn.org/stable/modules/neighbors.html#nca>`_.')
nca_docs_new = nca_docs.replace(old_str, new_str, 1)
NeighborhoodComponentsAnalysis.__doc__ = nca_docs_new
