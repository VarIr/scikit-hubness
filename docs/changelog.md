# Changelog

## [Next release]
...

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

[Next release]: https://github.com/VarIr/scikit-hubness/compare/release-0.21.0a8...HEAD
[0.21.0a8]: https://github.com/VarIr/scikit-hubness/compare/v0.21.0-alpha.7...release-0.21.0a8 
[0.21.0a7]: https://github.com/VarIr/scikit-hubness/releases/tag/v0.21.0-alpha.7

