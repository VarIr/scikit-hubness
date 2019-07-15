[![PyPI](https://img.shields.io/pypi/v/scikit-hubness.svg)](
https://pypi.org/project/scikit-hubness)
[![Docs](https://readthedocs.org/projects/scikit-hubness/badge/?version=latest)](
https://scikit-hubness.readthedocs.io/en/latest/?badge=latest)
[![TravisCI](https://travis-ci.com/VarIr/scikit-hubness.svg?branch=master)](
https://travis-ci.com/VarIr/scikit-hubness)
[![Coverage](https://codecov.io/gh/VarIr/scikit-hubness/branch/master/graph/badge.svg?branch=master)](
https://codecov.io/gh/VarIr/scikit-hubness)
[![AppVeyorCI](https://ci.appveyor.com/api/projects/status/85bs46irwcwwbvyt/branch/master?svg=true)](
https://ci.appveyor.com/project/VarIr/hubness/branch/master)
[![Quality](https://img.shields.io/lgtm/grade/python/g/VarIr/scikit-hubness.svg?logo=lgtm&logoWidth=18)](
https://lgtm.com/projects/g/VarIr/scikit-hubness/context:python)
[![License](https://img.shields.io/github/license/VarIr/scikit-hubness.svg)](
https://github.com/VarIr/scikit-hubness/blob/master/LICENSE.txt)

# scikit-hubness

(NOTE: THIS IS CURRENTLY UNDER HEAVY DEVELOPMENT.
A reasonably stable version will be available soon,
and will then be uploaded to PyPI).

`scikit-hubness` comprises tools for the analysis and
reduction of hubness in high-dimensional data.
Hubness is an aspect of the _curse of dimensionality_
and is detrimental to many machine learning and data mining tasks.

The `skhubness.analysis` and `skhubness.reduction` packages allow to

- analyze, whether your data sets show hubness
- reduce hubness via a variety of different techniques 
- perform downstream analysis (performance assessment) with `scikit-learn`
  due to compatible data structures

The `skhubness.neighbors` package acts as a drop-in replacement for `sklearn.neighbors`.
In addition to the functionality inherited from `scikit-learn`,
it also features
- _approximate nearest neighbor_ search
- hubness reduction
- and combinations,

which allows for fast hubness-reduced neighbor search in large datasets
(tested with >1M objects).

We follow the API conventions and code style of `scikit-learn`.

## Installation


Make sure you have a working Python3 environment (at least 3.7).

Use pip to install the latest stable version of `scikit-hubness` from PyPI:

```bash
pip install scikit-hubness
```

Dependencies are installed automatically, if necessary.
`scikit-hubness` requires `numpy`, `scipy` and `scikit-learn`.
Approximate nearest neighbor search and approximate hubness reduction
additionally requires `nmslib` and/or `falconn`.
Some modules require `tqdm` or `joblib`. All these packages are available
from open repositories, such as [PyPI](https://pypi.org).

For more details and alternatives, please see the [Installation instructions](
http://scikit-hubness.readthedocs.io/en/latest/user_guide/installation.html).

## Documentation

Documentation is available online: 
http://scikit-hubness.readthedocs.io/en/latest/index.html

## Quickstart

Users of `scikit-hubness` may want to 

1. analyse, whether their data show hubness
2. reduce hubness
3. perform learning (classification, regression, ...)

The following example shows all these steps for an example dataset
from the text domain (dexter). (Please make sure you have installed `hubness`).

```python
# load the example dataset 'dexter'
from skhubness.data import load_dexter
X, y = load_dexter()

# dexter is embedded in a high-dimensional space,
# and could, thus, be prone to hubness
print(f'X.shape = {X.shape}, y.shape={y.shape}')

# assess the actual degree of hubness in dexter
from skhubness import Hubness
hub = Hubness(k=10, metric='cosine')
hub.fit(X)
k_skew = hub.score()
print(f'Skewness = {k_skew:.3f}')

# additional hubness indices are available, for example:
print(f'Robin hood index: {hub.robinhood_index:.3f}')
print(f'Antihub occurrence: {hub.antihub_occurrence:.3f}')
print(f'Hub occurrence: {hub.hub_occurrence:.3f}')

# There is considerable hubness in dexter.
# Let's see, whether hubness reduction can improve
# kNN classification performance 
from sklearn.model_selection import cross_val_score
from skhubness.neighbors import KNeighborsClassifier

# vanilla kNN
knn_standard = KNeighborsClassifier(n_neighbors=5,
                                    metric='cosine')
acc_standard = cross_val_score(knn_standard, X, y, cv=5)

# kNN with hubness reduction (mutual proximity)
knn_mp = KNeighborsClassifier(n_neighbors=5,
                              metric='cosine',
                              hubness='mutual_proximity')
acc_mp = cross_val_score(knn_mp, X, y, cv=5)

print(f'Accuracy (vanilla kNN): {acc_standard.mean():.3f}')
print(f'Accuracy (kNN with hubness reduction): {acc_mp.mean():.3f}')

# Accuracy was considerably improved by mutual proximity.
# Did it actually reduce hubness?
hub_mp = Hubness(k=10, metric='cosine',
                 hubness='mutual_proximity')
hub_mp.fit(X)
k_skew_mp = hub_mp.score()
print(f'Skewness: {k_skew:.3f} '
      f'(reduction of {k_skew - k_skew_mp:.3f})')
print(f'Robin hood: {hub_mp.robinhood_index:.3f} '
      f'(reduction of {hub.robinhood_index - hub_mp.robinhood_index:.3f})')

# The neighbor graph can also be created directly,
# with or without hubness reduction
from skhubness.neighbors import kneighbors_graph
neighbor_graph = kneighbors_graph(X, n_neighbors=5, hubness='mutual_proximity')
```

Check the [Tutorial](http://scikit-hubness.readthedocs.io/en/latest/user_guide/tutorial.html)
for additional example usage. 


## Development

The developers of `scikit-hubness` welcome all kinds of contributions!
Get in touch with us if you have comments,
would like to see an additional feature implemented,
would like to contribute code or have any other kind of issue.
Please don't hesitate to file an [issue](https://github.com/VarIr/scikit-hubness/issues)
here on GitHub. 

    (c) 2018-2019, Roman Feldbauer
    Austrian Research Institute for Artificial Intelligence (OFAI) and
    University of Vienna, Division of Computational Systems Biology (CUBE)
    Contact: <roman.feldbauer@univie.ac.at>

## Citation

A software publication paper is currently in preparation. Until then,
if you use `scikit-hubness` in your scientific publication, please cite:

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
The `skhubness.neighbors` package was modified from `sklearn.neighbors`,
distributed under the same [license](external/SCIKIT_LEARN_LICENSE.txt).
Users can, therefore, safely use `scikit-hubness` in the same way they
use `scikit-learn`.

Acknowledgements
----------------
Several parts of `scikit-hubness` adapt code from `scikit-learn`.
We thank all the authors and contributors of this project
for the tremendous work they have done.

PyVmMonitor is being used to support the development of this free open source 
software package. For more information go to http://www.pyvmmonitor.com
