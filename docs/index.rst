.. scikit-hubness documentation master file, created by
   sphinx-quickstart on Mon Jul  8 13:54:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`scikit-hubness`: high-dimensional data mining
==============================================

`scikit-hubness` is a Python package for analysis and reduction of hubness
in high-dimensional data. It provides a drop-in replacement for `sklearn.neighbors`
with additional transparent support for approximate nearest neighbor search and hubness reduction.

User Guide
----------

The user guide explains how to install `scikit-hubness`, how to analyze your
data sets for hubness, and how to use the package to lift this
*curse of dimensionality*.

You will also find examples how to use `skhubness.neighbors` for approximate nearest neighbor search.


.. toctree::
  :maxdepth: 2

  user_guide/hub-toolbox_vs_scikit-hubness
  user_guide/installation
  user_guide/tutorial


`API Documentation <documentation.html>`_
------------------------------------------

The API documentation provides detailed information of the implemented methods.
This information includes method's descriptions, parameters, references, examples, etc.
Find all the information about specific modules and functions of `scikit-hubness` in this section.

.. toctree::
  :maxdepth: 2

  documentation

* :ref:`search`
