# Changelog

## [Next release]
...


## [0.30.0] - 2022-04-xx

### Major changes
- Compatibility with up-to-date scikit-learn versions
- Building upon the KNeighborsTransformer API as outlined in
  [our paper's Outlook section](https://joss.theoj.org/papers/10.21105/joss.01957).
- `skhubness.neighbors` rewritten from scratch. Previously, this was a drop-in
  replacement for `sklearn.neighbors` heavily relying on a specific scikit-learn
  version. This was hard to maintain. Now, this package only contains lightweight
  wrappers for approximate nearest neighbor search tools (`nmslib`, `ngt`, etc.)
  that return `KNeighborsTransformer`-compatible k-neighbors graphs. These can be
  reused in numerous scikit-learn classes and functions.
- `skhubness.analysis` uses `KNeighborsTransformer`-compatible k-neighbors graphs.
- `skhubness.reduction` uses `KNeighborsTransformer`-compatible k-neighbors graphs.

### Added or enhanced
- Python 3.9 and Python 3.10 support
- Additional metrics available for ANN search with `nmslib` #87
- Additional metrics available for ANN search with `ngtpy` #95
- Lower memory footprint for sparse targets in multilabel classification 
  (previously converted to dense arrays) #61

### Removed
- `falconn` removed (not maintained for 5+ years; index structures cannot be serialized; no Windows support) #94
- Radius neighbor search. Few ANN packages provide radius search. No currently supported
  ANN package supports this on Windows. Radius search is not of particular interest to
  hubness research. Thus, we decided to drop radius search for the time being to speed up
  development. Later releases might re-introduce radius search. Users interested in this
  are asked to file an Issue at Github

### Fixes
- Hubness estimation could fail when ANN does not return enough neighbors #59
- Heuristic to choose memory for Puffinn LSH.

### Maintenance
- Switch to modern Python packaging with `pyproject.toml` and `setup.cfg`
- Switch to Github Actions, dropping Travis CI and AppVeyor
- Renamed 0.22 to 0.30. Previous versions reflected the compatibility with specific
  scikit-learn versions by matching version numbers. The bump to 0.30 indicates that
  this tight coupling is gone and future scikit-hubness releases should be compatible
  with multiple scikit-learn versions.


## [0.21.2] - 2020-01-14

This is a maintenance release due to the publication in the
Journal of Open Source Software. 


## [0.21.1] - 2019-12-10

This is a bugfix release due to the recent update of scikit-learn to v0.22.
 
### Fixes
- Require scikit-learn v0.21.3. 
  
  Until the necessary adaptions for v0.22 are completed,
  scikit-hubness will require scikit-learn v0.21.3.


## [0.21.0] - 2019-11-25

This is the first major release of scikit-hubness.

### Added
- Enable ONNG provided by NGT (optimized ANNG). Pass ``optimize=True`` to ``LegacyNNG``.
- User Guide: Description of all subpackages and common usage scenarios.
- Examples: Various usage examples 
- Several tests
- Classes inheriting from ``SupervisedIntegerMixin`` can be fit with an 
  ``ApproximateNearestNeighbor`` or ``NearestNeighbors`` instance,
  thus reuse precomputed indices.

### Changes
- Use argument ``algorithm='nng'`` for ANNG/ONNG provided by NGT instead of ``'onng'``.
  Also set ``optimize=True`` in order to use ONNG.

### Fixes
- DisSimLocal would previously fail when invoked as ``hubness='dis_sim_local'``.
- Hubness reduction would previously ignore ``verbose`` arguments under certain circumstances.
- ``HNSW`` would previously ignore ``n_jobs`` on index creation.
- Fix installation instructions for puffinn.

## [0.21.0a9] - 2019-10-30
### Added
- General structure for docs
- Enable NGT OpenMP support on MacOS (in addition to Linux)
- Enable Puffinn LSH also on MacOS

### Fixes
- Correct mutual proximity (empiric) calculation
- Better handling of optional packages (ANN libraries)

### Maintenance
- streamlined CI builds
- several minor code improvements

### New contributors
- Silvan David Peter


## [0.21.0a8] - 2019-09-12
### Added
- Approximate nearest neighbor search
    * LSH by an additional provider, [`puffinn`](https://github.com/puffinn/puffinn) (Linux only, atm)
    * ANNG provided by [`ngtpy`](https://github.com/yahoojapan/NGT/) (Linux, MacOS)
    * Random projection forests provided by [`annoy`](https://github.com/spotify/annoy) (Linux, MacOS, Windows)

### Fixes
- Several minor issues
- Several documentations issues


## [0.21.0a7] - 2019-07-17

The first alpha release of `scikit-hubness` to appear in this changelog.
It already contains the following features:

- Hubness estimation (exact or approximate)
- Hubness reduction (exact or approximate)
  * Mutual proximity
  * Local scaling
  * DisSim Local
- Approximate nearest neighbor search
  * HNSW provided by [nmslib](https://github.com/nmslib/nmslib)
  * LSH provided by [falconn](https://github.com/FALCONN-LIB/FALCONN)

[Next release]: https://github.com/VarIr/scikit-hubness/compare/v0.30.0...HEAD
[0.30.0]:   https://github.com/VarIr/scikit-hubness/releases/tag/v0.30.0
[0.21.2]:   https://github.com/VarIr/scikit-hubness/releases/tag/v0.21.2
[0.21.1]:   https://github.com/VarIr/scikit-hubness/releases/tag/v0.21.1
[0.21.0]:   https://github.com/VarIr/scikit-hubness/releases/tag/v0.21.0
[0.21.0a9]: https://github.com/VarIr/scikit-hubness/releases/tag/v0.21.0-alpha.9
[0.21.0a8]: https://github.com/VarIr/scikit-hubness/releases/tag/v0.21.0-alpha.8
[0.21.0a7]: https://github.com/VarIr/scikit-hubness/releases/tag/v0.21.0-alpha.7

[//]: # "Sections: Added, Fixed, Changed, Removed"
