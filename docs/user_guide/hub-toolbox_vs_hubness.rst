`hubness`: successor to the Hub-Toolbox
=======================================

The `hubness` package builds upon previous software: the Hub-Toolbox.
The original `Hub-Toolbox <https://github.com/OFAI/hub-toolbox-matlab>`_
was written for Matlab, and released in parallel
with the release of the first hubness reduction methods in
`JMLR <http://www.jmlr.org/papers/v13/schnitzer12a.html>`_.
In essence, it comprises methods to reduce hubness in distance matrices.

The `Hub-Toolbox for Python3 <https://github.com/OFAI/hub-toolbox-python3>`_
is a port from Matlab to Python,
which over the years got several extensions and additional functionality,
such as more hubness reduction methods (Localized Centering, DisSimLocal, mp-dissim, etc.),
approximate hubness reduction, and more.
The software was developed by hubness researchers for hubness research.

The new `hubness` package is rewritten from scratch with a different goal in mind:
Providing easy-to-use neighborhood-based data mining methods (classification, regression, etc.)
with transparent hubness reduction.
Building upon scikit-learn's `neighbors` package, we provide a drop-in replacement
called `hubness.neighbors`, which offers all the functionality of `sklearn.neighbors`,
but adds additional functionality (approximate nearest neighbor search, hubness reduction).

This way, we think that machine learning researchers and practitioners
(many of which will be fluent in scikit-learn)
can quickly and effectively employ `hubness` in their existing workflows,
and improve learning in their high-dimensional data.
