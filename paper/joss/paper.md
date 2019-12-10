---
title: 'scikit-hubness: Hubness Reduction and Approximate Neighbor Search'

tags:
  - Python
  - scikit-learn
  - hubness
  - curse of dimensionality
  - nearest neighbors

authors:
  - name: Roman Feldbauer
    orcid: 0000-0003-2216-4295
    affiliation: 1
  - name: Thomas Rattei
    orcid: 0000-0002-0592-7791
    affiliation: 1
  - name: Arthur Flexer
    orcid: 0000-0002-1691-737X
    affiliation: 2

affiliations:
 - name: Division of Computational Systems Biology, Department of Microbiology and Ecosystem Science,
         University of Vienna, Althanstra&szlig;e 14, 1090 Vienna, Austria
   index: 1
 - name: Austrian Research Institute for Artificial Intelligence (OFAI),
         Freyung 6/6/7, 1010 Vienna, Austria
   index: 2

date: 09 December 2019

bibliography: paper.bib

---

# Summary

``scikit-hubness`` is a Python package for efficient
nearest neighbor search in high-dimensional spaces.
Hubness is an aspect of the *curse of dimensionality*
in nearest neighbor graphs.
Specifically, it describes the increasing occurrence of *hubs*
and *antihubs* with growing data dimensionality:
Hubs are objects, that appear unexpectedly often among the nearest neighbors
of others objects, while antihubs are never retrieved as neighbors.
As a consequence, hubs may propagate their information (for example, class labels)
too widely within the neighbor graph, while information from antihubs is depleted.
These semantically distorted graphs can reduce learning performance
in various tasks, such as
classification [@Radovanovic2010],
clustering [@Schnitzer2015],
or visualization [@Flexer2015a].
Hubness is known to affect a variety of applied learning systems [@Angiulli2018],
causing  &mdash; for instance  &mdash; overrepresentation of certain songs in music recommendations [@Flexer2018],
or improper transport mode detection [@Feldbauer2018].

Multiple hubness reduction algorithms have been developed to mitigate these
effects [@Schnitzer2012; @Flexer2013; @Hara2016].
We compared these algorithms exhaustively in a recent survey [@Feldbauer2019],
and developed approximate hubness reduction methods with linear time
and memory complexity [@Feldbauer2018]. 

Currently, there is a lack of fully-featured, up-to-date, user-friendly
software dealing with hubness.
Available packages miss critical features and have not been updated in years [@Hubminer],
or are not particularly user-friendly [@Hubtoolbox].
In this paper we describe ``scikit-hubness``, which
provides powerful, readily available, and easy-to-use hubness-related methods:

- hubness analysis ("Is my data affected by hubness?"):
Assess hubness with several measures, including
*k*-occurrence skewness [@Radovanovic2010],
and Robin-Hood index [@Feldbauer2018].

- hubness reduction ("How can we improve neighbor retrieval in
high dimensions?"): Mutual proximity, local scaling,
and DisSim<sup>Local</sup> are currently supported,
as they performed best in our survey.
Exact methods as well as their approximations are available.

- approximate neighbor search ("Does it work for large data sets?"):
Several methods are currently available, including
locality-sensitive hashing [@Aumueller2019]
and hierarchical navigable small-world graphs [@Malkov16].

``scikit-hubness`` builds upon the SciPy stack [@Virtanen2019]
and is integrated into the ``scikit-learn`` environment [@Pedregosa2011],
enabling rapid adoption by Python-based machine learning
researchers and practitioners.
Convenient interfaces to hubness-reduced neighbors-based learning
are available in the ``skhubness.neighbors`` subpackage.
It acts as a drop-in replacement for ``sklearn.neighbors``,
featuring all its functionality, and adding support for hubness reduction,
where applicable. This includes, for example,
the supervised ``KNeighborsClassifier`` and ``RadiusNeighborsRegressor``,
``NearestNeighbors`` for unsupervised learning,
and the general ``kneighbors_graph``.

``scikit-hubness`` is developed using several quality assessment tools and principles,
such as PEP8 compliance, unit tests with high code coverage, continuous integration
on all major platforms
(Linux and MacOS [1],
Windows [2]),
and additional checks by LGTM [3].
The source code is available at GitHub [4]
under the BSD 3-clause license.
The online documentation is available at Read the Docs [5].
Install from the Python package index
with ``$ pip install scikit-hubness``.

[1] https://travis-ci.com/VarIr/scikit-hubness/

[2] https://ci.appveyor.com/project/VarIr/scikit-hubness

[3] https://lgtm.com/projects/g/VarIr/scikit-hubness/

[4] https://github.com/VarIr/scikit-hubness 

[5] https://scikit-hubness.readthedocs.io/

# Outlook

Future plans include adaption to significant changes of ``sklearn.neighbors``
introduced in version 0.22 in December 2019:
The ``KNeighborsTransformer`` and ``RadiusNeighborsTransformer``
transform data into sparse neighbor graphs,
which can subsequently be used as input to other estimators.
Hubness reduction and approximate search can then be implemented as ``Transformers``.
This provides the means to turn ``skhubness.neighbors`` from a drop-in replacement
of ``sklearn.neighbors`` into a scikit-learn plugin,
which will (1) accelerate development, 
(2) simplify addition of new hubness reduction and approximate search methods, and
(3) facilitate more flexible usage.

[//]: #  (https://github.com/scikit-learn/scikit-learn/pull/10482)
[//]: #  (and, thus, between the main development
phase of scikit-hubness and submission of this manuscript)
# Acknowledgements

We thank Silvan David Peter for testing the software.<br/>
This research is supported by the Austrian Science Fund (FWF): P27703 and P31988

# References