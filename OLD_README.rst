.. image:: https://img.shields.io/pypi/v/hubness.svg
    :alt: PyPI

.. image:: https://readthedocs.org/projects/hubness/badge/?version=latest
    :target: https://hubness.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.com/VarIr/hubness.svg?branch=master
    :target: https://travis-ci.com/VarIr/hubness

.. image:: https://codecov.io/gh/VarIr/hubness/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/VarIr/hubness

.. image:: https://img.shields.io/github/license/VarIr/hubness.svg
    :alt: GitHub
    :target: https://github.com/VarIr/hubness/blob/master/LICENSE.txt



HUBNESS
===========

The `hubness` package comprises tools for the analysis and
reduction of hubness in high-dimensional data.
Hubness is an aspect of the "curse of dimensionality" and was
shown to be detrimental to many machine learning and data mining tasks.
(Throughout the docs we will refer to the phenomenon as 'hubness' (lower case)
and to the package as 'HUBNESS'.)

The HUBNESS package allows you to

- analyze, whether your data sets show hubness
- reduce hubness via a variety of different techniques 
  (including mutual proximity, local scaling, DisSimLocal and others)
  and obtain secondary distances for downstream analysis inside or 
  outside the package
- perform evaluation tasks with both internal and external measures
  (e.g. Goodman-Kruskal index and k-NN classification)

We try to follow the API conventions and code style of scikit-learn.

Installation
------------

Make sure you have a working Python3 environment (at least 3.7) with
numpy, scipy and scikit-learn packages. Approximate hubness reduction
additionally requires nmslib and/or falconn. Some modules require pandas or joblib.

Use pip3 to install the latest stable version of HUBNESS:

.. code-block:: bash

  pip3 install hubness

For more details and alternatives, please see the `Installation instructions
<http://hubness.readthedocs.io/en/latest/user/installation.html>`_.

Documentation
-------------

Documentation is available online: 
http://hubness.readthedocs.io/en/latest/index.html

Example
-------
.. TODO adapt to actual package structure when done

To run a full hubness analysis on the example data set (DEXTER)
using some of the provided hubness reduction methods, 
simply run the following in a Python shell:

.. code-block:: python

    >>> from hubness.HubnessAnalysis import HubnessAnalysis
    >>> ana = HubnessAnalysis()
    >>> ana.analyze_hubness()

See how you can conduct the individual analysis steps:

.. code-block:: python

    import hubness

    # load the DEXTER example data set
    X, y, D = hubness.utils.load_dexter()

    # calculate intrinsic dimension estimate
    d_mle = hubness.intrinsic_dimension.intrinsic_dimension(vectors)

    # calculate hubness (here, skewness of 5-occurence)
    S_k, _, _ = hubness.analysis.skewness(D=D, k=5, metric='distance')

    # perform k-NN classification LOO-CV for two different values of k
    acc, _, _ = hubness.analysis.score(
        D=D, target=labels, k=[1,5], metric='distance')

    # calculate Goodman-Kruskal index
    gamma = hubness.analysis.goodman_kruskal_index(
        D=D, classes=labels, metric='distance')

    # Reduce hubness with Mutual Proximity (Empiric distance distribution)
    D_mp = hubness.reduction.mutual_proximity_empiric(
        D=D, metric='distance')

    # Reduce hubness with Local Scaling variant NICDM
    D_nicdm = hubness.reduction.nicdm(D=D, k=10, metric='distance')

    # Check whether indices improve after hubness reduction
    S_k_mp, _, _ = hubness.analysis.skewness(D=D_mp, k=5, metric='distance')
    acc_mp, _, _ = hubness.analysis.score(
        D=D_mp, target=labels, k=[1,5], metric='distance')
    gamma_mp = hubness.analysis.goodman_kruskal_index(
        D=D_mp, classes=labels, metric='distance')

    # Repeat the last steps for all secondary distances you calculated
    ...

Check the `Tutorial
<http://hubness.readthedocs.io/en/latest/user/tutorial.html>`_
for in-depth explanations of the same. 


Development
-----------

The HUBNESS package is a work in progress. Get in touch with us if you have
comments, would like to see an additional feature implemented, would like
to contribute code or have any other kind of issue. Please don't hesitate
to file an `issue <https://github.com/VarIr/hubness/issues>`_
here on GitHub. 

.. code-block:: text

    (c) 2018-2019, Roman Feldbauer
    Austrian Research Institute for Artificial Intelligence (OFAI) and
    University of Vienna, Division of Computational Systems Biology (CUBE)
    Contact: <roman.feldbauer@univie.ac.at>

Citation
--------
.. TODO update when ICBK2018 is published

If you use the HUBNESS package in your scientific publication, please cite:

.. code-block:: text

    @Inbook{Feldbauer2018,
        author="Feldbauer, Roman
        and Leodolter, Maximilian
        and Plant, Claudia
        and Flexer, Arthur",
        title="Fast approximate hubness reduction for large high-dimensional data",
        bookTitle="IEEE International Conference on Big Knowledge 2018",
        year="2018",
        publisher="IEEE Computer Society",
        }

The technical report `Fast approximate hubness reduction for large high-dimensional data`
is available at
`<http://www.ofai.at/cgi-bin/tr-online?number+2018-02>`_.

Additional reading

`Local and Global Scaling Reduce Hubs in Space`, Journal of Machine Learning Research 2012,
`Link <http://www.jmlr.org/papers/v13/schnitzer12a.html>`_.

`A comprehensive empirical comparison of hubness reduction in high-dimensional spaces`,
Knowledge and Information Systems 2018, `DOI <https://doi.org/10.1007/s10115-018-1205-y>`_.

License
-------
The HUBNESS package is licensed under the terms of the GNU GPLv3.

Acknowledgements
----------------
PyVmMonitor is being used to support the development of this free open source 
software package. For more information go to http://www.pyvmmonitor.com
