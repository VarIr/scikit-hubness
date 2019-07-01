[![PyPI](https://img.shields.io/pypi/v/hubness.svg)](
https://pypi.org/project/hubness)
[![Docs](https://readthedocs.org/projects/hubness/badge/?version=latest)](
https://hubness.readthedocs.io/en/latest/?badge=latest)
[![Build](https://travis-ci.com/VarIr/hubness.svg?branch=master)](
https://travis-ci.com/VarIr/hubness)
[![Coverage](https://codecov.io/gh/VarIr/hubness/branch/master/graph/badge.svg?branch=master)](
https://codecov.io/gh/VarIr/hubness)
[![Quality](https://img.shields.io/lgtm/grade/python/g/VarIr/hubness.svg?logo=lgtm&logoWidth=18)](
https://lgtm.com/projects/g/VarIr/hubness/context:python)
![License](https://img.shields.io/github/license/VarIr/hubness.svg)

# Hubness

(NOTE: THIS IS CURRENTLY UNDER HEAVY DEVELOPMENT. The API is not stable yet,
things might be broken here and there, docs are missing, etc.
A reasonably stable version is hopefully available soon,
and will then be uploaded to PyPI).

The `hubness` package comprises tools for the analysis and
reduction of hubness in high-dimensional data.
Hubness is an aspect of the _curse of dimensionality_
and is detrimental to many machine learning and data mining tasks.

The `hubness.analysis` and `hubness.reduction` package allows you to

- analyze, whether your data sets show hubness
- reduce hubness via a variety of different techniques 
- perform evaluation tasks with both internal and external measures

The `hubness.neighbors` package acts as a drop-in replacement for `sklearn.neighbors`.
In addition to the functionality inherited from `scikit-learn`,
it also features
- _approximate nearest neighbor_ search
- hubness reduction
- and combinations,

which allows for fast hubness reduced neighbor search in large datasets
(tested with >1M objects).

We try to follow the API conventions and code style of scikit-learn.

## Installation


Make sure you have a working Python3 environment (at least 3.7) with
numpy, scipy and scikit-learn packages. Approximate hubness reduction
additionally requires nmslib and/or falconn. Some modules require pandas or joblib.

Use pip3 to install the latest stable version of HUBNESS:

```bash
pip3 install hubness
```

For more details and alternatives, please see the [Installation instructions](
http://hubness.readthedocs.io/en/latest/user/installation.html).

## Documentation

Documentation is available online: 
http://hubness.readthedocs.io/en/latest/index.html

## Example

TODO adapt to actual package structure when done

To run a full hubness analysis on the example data set (DEXTER)
using some of the provided hubness reduction methods, 
simply run the following in a Python shell:

    >>> from hubness.HubnessAnalysis import HubnessAnalysis
    >>> ana = HubnessAnalysis()
    >>> ana.analyze_hubness()

See how you can conduct the individual analysis steps:

```python
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
```
    

Check the [Tutorial](http://hubness.readthedocs.io/en/latest/user/tutorial.html)
for in-depth explanations of the same. 


## Development

The HUBNESS package is a work in progress. Get in touch with us if you have
comments, would like to see an additional feature implemented, would like
to contribute code or have any other kind of issue. Please don't hesitate
to file an [issue](https://github.com/VarIr/hubness/issues)
here on GitHub. 

    (c) 2018-2019, Roman Feldbauer
    Austrian Research Institute for Artificial Intelligence (OFAI) and
    University of Vienna, Division of Computational Systems Biology (CUBE)
    Contact: <roman.feldbauer@univie.ac.at>

## Citation

A software publication paper is currently in preparation. Until then,
if you use the HUBNESS package in your scientific publication, please cite:

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

Additional reading

`Local and Global Scaling Reduce Hubs in Space`, Journal of Machine Learning Research 2012,
[Link](http://www.jmlr.org/papers/v13/schnitzer12a.html).

`A comprehensive empirical comparison of hubness reduction in high-dimensional spaces`,
Knowledge and Information Systems 2018, [DOI](https://doi.org/10.1007/s10115-018-1205-y).

License
-------
The HUBNESS package is licensed under the terms of the GNU GPLv3.

Acknowledgements
----------------
PyVmMonitor is being used to support the development of this free open source 
software package. For more information go to http://www.pyvmmonitor.com
