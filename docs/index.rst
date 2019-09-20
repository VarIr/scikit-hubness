.. scikit-hubness documentation master file, created by
   sphinx-quickstart on Mon Jul  8 13:54:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`scikit-hubness`: high-dimensional data mining
==============================================

`scikit-hubness` is a Python package for analysis of hubness
in high-dimensional data. It provides hubness reduction and
approximate nearest neighbor search via a drop-in replacement for
`sklearn.neighbors <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors>`_.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   Installation <getting_started/installation>
   Quick start example <getting_started/example>
   History <getting_started/history>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation

   User Guide <documentation/user_guide>
   scikit-hubness API <documentation/documentation>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:
   :caption: Development

   Contributing <https://github.com/VarIr/scikit-hubness/blob/master/CONTRIBUTING.md>
   Github Repository <https://github.com/VarIr/scikit-hubness>
   What's new (Changelog) <changelog.md>


`Getting started <getting_started>`_
------------------------------------

The user guide explains how to install `scikit-hubness`, how to analyze your
data sets for hubness, and how to use the package to lift this
*curse of dimensionality*.

You will also find examples how to use `skhubness.neighbors` for approximate nearest neighbor search.

`API Documentation <documentation/documentation.html>`_
--------------------------------------------------------

The API documentation provides detailed information of the implemented methods.
This information includes method's descriptions, parameters, references, examples, etc.
Find all the information about specific modules and functions of `scikit-hubness` in this section.

* :ref:`search`
