# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.neighbors import NearestCentroid

__all__ = ['NearestCentroid']

nc_docs = NearestCentroid.__doc__
old_str = 'Read more in the :ref:`User Guide <nearest_centroid_classifier>`.'
new_str = ('Read more in the `scikit-learn User Guide '
           '<https://scikit-learn.org/stable/modules/neighbors.html#nearest-centroid-classifier>`_.')
nc_docs_new = nc_docs.replace(old_str, new_str, 1)
NearestCentroid.__doc__ = nc_docs_new
