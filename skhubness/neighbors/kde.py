# SPDX-License-Identifier: BSD-3-Clause

from sklearn.neighbors.kde import KernelDensity

__all__ = ['KernelDensity']

kde_docs = KernelDensity.__doc__
old_str = 'Read more in the :ref:`User Guide <kernel_density>`.'
new_str = ('Read more in the `scikit-learn User Guide '
           '<https://scikit-learn.org/stable/modules/density.html#kernel-density>`_.')
kde_docs_new = kde_docs.replace(old_str, new_str, 1)
KernelDensity.__doc__ = kde_docs_new
