[![PyPI](https://img.shields.io/pypi/v/scikit-hubness.svg)](
https://pypi.org/project/scikit-hubness)
[![Docs](https://readthedocs.org/projects/scikit-hubness/badge/?version=latest)](
https://scikit-hubness.readthedocs.io/en/latest/?badge=latest)
[![Actions](https://github.com/VarIr/scikit-hubness/actions/workflows/scikit-hubness_ci.yml/badge.svg?branch=main)](
https://github.com/VarIr/scikit-hubness/actions/workflows/scikit-hubness_ci.yml)
[![Coverage](https://codecov.io/gh/VarIr/scikit-hubness/branch/master/graph/badge.svg?branch=master)](
https://codecov.io/gh/VarIr/scikit-hubness)
[![Quality](https://img.shields.io/lgtm/grade/python/g/VarIr/scikit-hubness.svg?logo=lgtm&logoWidth=18)](
https://lgtm.com/projects/g/VarIr/scikit-hubness/context:python)
[![License](https://img.shields.io/github/license/VarIr/scikit-hubness.svg)](
https://github.com/VarIr/scikit-hubness/blob/master/LICENSE.txt)
[![DOI](https://zenodo.org/badge/193863864.svg)](
https://zenodo.org/badge/latestdoi/193863864)
[![arXiv](https://img.shields.io/badge/cs.LG-arXiv%3A1912.00706-B31B1B)](
https://arxiv.org/abs/1912.00706)
[![status](https://joss.theoj.org/papers/b9b56c7c109ff2a8a0c7c216cb3f8c39/status.svg)](
https://joss.theoj.org/papers/b9b56c7c109ff2a8a0c7c216cb3f8c39)

# scikit-hubness

`scikit-hubness` provides tools for the analysis and
reduction of hubness in high-dimensional data.
Hubness is an aspect of the _curse of dimensionality_
and is detrimental to many machine learning and data mining tasks.

The `skhubness.analysis` and `skhubness.reduction` packages allow to

- analyze, whether your data sets show hubness
- reduce hubness via a variety of different techniques
- perform downstream analysis (performance assessment) with `scikit-learn`
  due to compatible data structures

The `skhubness.neighbors` package provides approximate nearest neighbor (ANN)
search. This is compatible with scikit-learn classes and functions relying
on neighbors graphs due to compliance with [KNeighborsTransformer](
https://scikit-learn.org/stable/modules/neighbors.html#neighbors-transformer) APIs
and data structures. Using ANN can speed up many scikit-learn classification,
clustering, embedding and other methods, including:
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)
- [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)
- and many more.

`scikit-hubness` thus provides
- _approximate nearest neighbor_ search
- hubness reduction
- and combinations,

which allows for fast hubness-reduced neighbor search in large datasets
(tested with >1M objects).


## Installation

Make sure you have a working Python3 environment (at least 3.8).

Use pip to install the latest stable version of `scikit-hubness` from PyPI:

```bash
pip install scikit-hubness
```

NOTE: v0.30 is currently under development and not yet available on PyPI.
Install from sources to obtain the bleeding edge version.

Dependencies are installed automatically, if necessary.
`scikit-hubness` is based on the SciPy-stack, including `numpy`, `scipy` and `scikit-learn`.
Approximate nearest neighbor search and approximate hubness reduction
additionally require at least one of the following packages:
* [`nmslib`](https://github.com/nmslib/nmslib)
    for hierachical navigable small-world graphs in `skhubness.neighbors.NMSlibTransformer`
* [`ngtpy`](https://github.com/yahoojapan/NGT/)
    for nearest neighbor graphs (ANNG, ONNG) in `skhubness.neighbors.NGTTransformer`
* [`puffinn`](https://github.com/puffinn/puffinn)
    for locality-sensitive hashing in `skhubness.neighbors.PuffinnTransformer`
* [`annoy`](https://github.com/spotify/annoy)
    for random projection forests in `skhubness.neighobrs.AnnoyTransformer`
* Additional ANN libraries might be added in future releases. Please reach out to us in a Github Issue,
  if you think a specific library is missing (pull requests welcome).

For more details and alternatives, please see the [Installation instructions](
http://scikit-hubness.readthedocs.io/en/latest/user_guide/installation.html).

## Documentation

Additional documentation is available online: 
http://scikit-hubness.readthedocs.io/en/latest/index.html


## What's new

See the [changelog](docs/changelog.md) to find what's new in the latest package version.

 
## Quickstart

Users of `scikit-hubness` may want to 

1. analyse, whether their data show hubness
2. reduce hubness
3. perform learning (classification, regression, ...)

The following example shows all these steps for an example dataset
from the text domain (dexter). (Please make sure you have installed `scikit-hubness`).

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsTransformer

from skhubness import Hubness
from skhubness.data import load_dexter
from skhubness.neighbors import NMSlibTransformer
from skhubness.reduction import MutualProximity


# load the example dataset 'dexter' that is embedded in a
# high-dimensional space, and could, thus, be prone to hubness.
X, y = load_dexter()
print(f'X.shape = {X.shape}, y.shape = {y.shape}')

# assess the actual degree of hubness in dexter
hub = Hubness(k=10, metric='cosine')
hub.fit(X)
k_skew = hub.score()
print(f'Skewness = {k_skew:.3f}')

# additional hubness indices are available, for example:
hub = Hubness(k=10, return_value="all", metric='cosine')
scores = hub.fit(X).score()
print(f'Robin hood index:   {scores.get("robinhood"):.3f}')
print(f'Antihub occurrence: {scores.get("antihub_occurrence"):.3f}')
print(f'Hub occurrence:     {scores.get("hub_occurrence"):.3f}')

# There is considerable hubness in dexter. Let's see, whether 
# hubness reduction can improve kNN classification performance.
# We first create a kNN graph:
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

# Accuracy was considerably improved by mutual proximity.
# Did it actually reduce hubness?
mp_scores = hub.fit(mp_graph).score()
print(f'k-skewness after MP: {mp_scores.get("k_skewness"):.3f} '
      f'(reduction of {scores.get("k_skewness") - mp_scores.get("k_skewness"):.3f})')
print(f'Robinhood after MP:  {mp_scores.get("robinhood"):.3f} '
      f'(reduction of {scores.get("robinhood") - mp_scores.get("robinhood"):.3f})')
```

Check the [User Guide](http://scikit-hubness.readthedocs.io/en/latest/user_guide.html)
for additional example usage. 


## Development

The developers of `scikit-hubness` welcome all kinds of contributions!
Get in touch with us if you have comments,
would like to see an additional feature implemented,
would like to contribute code or have any other kind of issue.
Don't hesitate to file an [issue](https://github.com/VarIr/scikit-hubness/issues)
here on GitHub.

For more information about contributing, please have a look at the
[contributors guidelines](CONTRIBUTING.rst).

    (c) 2018-2022, Roman Feldbauer
    -2018: Austrian Research Institute for Artificial Intelligence (OFAI) and
    -2021: University of Vienna, Division of Computational Systems Biology (CUBE)
    2021-: Independent researcher
    Contact: <sci@feldbauer.org>

## Citation

If you use `scikit-hubness` in your scientific publication, please cite:

    @Article{Feldbauer2020,
      author  = {Roman Feldbauer and Thomas Rattei and Arthur Flexer},
      title   = {scikit-hubness: Hubness Reduction and Approximate Neighbor Search},
      journal = {Journal of Open Source Software},
      year    = {2020},
      volume  = {5},
      number  = {45},
      pages   = {1957},
      issn    = {2475-9066},
      doi     = {10.21105/joss.01957},
    }

To specifically acknowledge *approximate hubness reduction*, please cite:

    @INPROCEEDINGS{8588814,
    author={R. {Feldbauer} and M. {Leodolter} and C. {Plant} and A. {Flexer}},
    booktitle={2018 IEEE International Conference on Big Knowledge (ICBK)},
    title={Fast Approximate Hubness Reduction for Large High-Dimensional Data},
    year={2018},
    volume={},
    number={},
    pages={358-367},
    keywords={computational complexity;data analysis;data mining;mobile computing;public domain software;software packages;mobile device;open source software package;high-dimensional data mining;fast approximate hubness reduction;massive mobility data;linear complexity;quadratic algorithmic complexity;dimensionality curse;Complexity theory;Indexes;Estimation;Data mining;Approximation algorithms;Time measurement;curse of dimensionality;high-dimensional data mining;hubness;linear complexity;interpretability;smartphones;transport mode detection},
    doi={10.1109/ICBK.2018.00055},
    ISSN={},
    month={Nov},}

The technical report `Fast approximate hubness reduction for large high-dimensional data`
is available at [OFAI](http://www.ofai.at/cgi-bin/tr-online?number+2018-02).

### Additional reading

`Local and Global Scaling Reduce Hubs in Space`, Journal of Machine Learning Research 2012,
[Link](http://www.jmlr.org/papers/v13/schnitzer12a.html).

`A comprehensive empirical comparison of hubness reduction in high-dimensional spaces`,
Knowledge and Information Systems 2018, [DOI](https://doi.org/10.1007/s10115-018-1205-y).

License
-------
`scikit-hubness` is licensed under the terms of the BSD-3-Clause [license](LICENSE.txt).

------------------------------------------------------------------------------
Note:
Individual files contain the following tag instead of the full license text.

        SPDX-License-Identifier: BSD-3-Clause

This enables machine processing of license information based on the SPDX
License Identifiers that are here available: https://spdx.org/licenses/

Acknowledgements
----------------
Parts of `scikit-hubness` adapt code from `scikit-learn`.
We thank all the authors and contributors of this project
for the tremendous work they have done.

PyVmMonitor is being used to support the development of this free open source 
software package. For more information go to http://www.pyvmmonitor.com
