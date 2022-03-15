# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Original work: https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html
# Author: Tom Dupre la Tour (original work)
#         Roman Feldbauer (adaptions for scikit-hubness)
# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations
import logging
from typing import Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

try:
    import nmslib
except ImportError:
    nmslib = None  # pragma: no cover

from .approximate_neighbors import ApproximateNearestNeighbor
from ..utils.check import check_n_candidates
from ..utils.io import create_tempfile_preferably_in_dir

__all__ = [
    "LegacyHNSW",
    "NMSlibTransformer",
]


class NMSlibTransformer(BaseEstimator, TransformerMixin):
    """Approximate nearest neighbors retrieval with NMSLIB (non-metric space library).

    Compatible with sklearn's KNeighborsTransformer.
    NMSLIB is an approximate nearest neighbor library,
    that builds (hierarchically navigable) small-world graphs from
    a large number of different dissimilarity measure.

    Parameters
    ----------
    n_neighbors: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = "euclidean"
        Distance metric, allowed are "cosine", "euclidean", and many others.
        See ``NMSlibTransformer.valid_metrics`` for the complete list.
    p : float, optional
        Set p to define Lp space when using ``metric=="lp"``.
    alpha : float, optional
        Set alpha when using sqfd metrics
    method : str, default = "hnsw",
        (Approximate) nearest neighbor method to use. Allowed are "hnsw", "sw-graph",
        "vp-tree", "napp", "simple_invindx", "brute_force". Methods have individual
        parameters that may be tuned.
    efConstruction : float, optional
        Increasing the value of efConstruction improves the quality of a constructed graph
        and leads to higher accuracy of search, at the cost of longer indexing times.
        Relevant for methods "sw-graph" and "hnsw".
    ef, efSearch : float, optional
        Increasing the value of ef ("hnsw") or efSearch ("sw-graph") improves
        recall at the expense of longer retrieval time.
        The reasonable range of values for these parameters is 100-2000.
        Relevant for method "hnsw" and "sw-graph", respectively.
    M, NN : float, optional
        The recall values are also affected by parameters NN (for "sw-graph") and M ("hnsw").
        Increasing the values of these parameters (to a certain degree) leads to better
        recall and shorter retrieval times (at the expense of longer indexing time).
        For low and moderate recall values (e.g., 60-80%) increasing these parameters
        may lead to longer retrieval times.
        The reasonable range of values for these parameters is 5-100.
        Relevant for method "hnsw" and "sw-graph", respectively.
    delaunay_type : int, one of 0, 1, 2, 3, optional
        There is a trade-off between retrieval performance and indexing time related
        to the choice of the pruning heuristic (controlled by the parameter delaunay_type).
        Specifically, by default delaunay_type is equal to 2. This default is generally quite good.
        Relevant for method "hnsw".
    post_processing: int, optional
        Defines the amount (and type) of postprocessing applied to the constructed graph.
        Value 0 means no postprocessing. Additional options are 1 and 2 (2 means more postprocessing).
        More post processing means longer index creation, and higher retrieval accuracy.
        Relevant for method "hnsw".
    skip_optimized_index : int, default = 0
        There is a pesky design decision in NMSLIB that an index does not necessarily contain the data points,
        which are loaded separately. HNSW chooses to include data points into the index in several important cases,
        which include the dense spaces for the Euclidean and the cosine distance.
        These optimized indices are created automatically whenever possible.
        However, this behavior can be overriden by setting the parameter skip_optimized_index to 1.
        Relevant for method "hnsw".
    desiredRecall, bucketSize, tuneK, tuneR, tuneQty, minExp, maxExp
        Parameters relevant for method "vp-tree".
        See https://github.com/nmslib/nmslib/blob/master/manual/methods.md for details.
    numPivot, numPivotIndex, numPivotSearch, hashTrickDim
        Parameters relevant for method "napp".
        See https://github.com/nmslib/nmslib/blob/master/manual/methods.md for details.
    n_jobs: int, default = 1
        Number of parallel jobs
    mmap_dir: str, default = 'auto'
        Memory-map the index to the given directory. This is required to make the the class pickleable.
        If None, keep everything in main memory (NON pickleable index),
        if mmap_dir is a string, it is interpreted as a directory to store the index into,
        if "auto", create a temp dir for the index, preferably in /dev/shm on Linux.
    verbose: int, default = 0
        Verbosity level. If verbose >= 2, show progress bar on indexing.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures
    """
    # https://github.com/nmslib/nmslib/tree/master/manual
    # Out-commented metrics are supported by NMSlib, but not yet here.
    # If you need them, file an issue with scikit-hubness at GitHub.
    valid_metrics = [
        # "bit_hamming",
        # "bit_jaccard",
        # "jaccard_sparse",
        "l1",
        "l1_sparse",
        "euclidean",
        "l2",
        "l2_sparse",
        "linf",
        "linf_sparse",
        "lp",
        "lp_sparse",
        "angulardist",
        "angulardist_sparse",
        "angulardist_sparse_fast",
        "jsmetrslow",
        "jsmetrfast",
        "jsmetrfastapprox",
        # "leven",
        # "sqfdminusfunc",
        # "sqfdheuristicfunc",
        # "sqfdgaussianfunc",
        # "sdivslow",
        "jsdivfast",
        "jsdivfastapprox",
        "cosine",
        "cosinesimil",
        "cosinesimil_sparse",
        "cosinesimil_sparse_fast",
        # "normleven",
        "kldivfast",
        "kldivfastrq",
        "kldivgenslow",
        "kldivgenfast",
        "kldivgenfastrq",
        # "itakurasaitoslow",
        # "itakurasaitofast",
        # "itakurasaitofastrq",
        # "renyidiv_slow",
        # "renyidiv_fast",
        "negdotprod_sparse_fast",
    ]

    def __init__(self, n_neighbors=5, metric="euclidean",
                 p: float = None,
                 alpha: float = None,
                 method: str = "hnsw",
                 efConstruction: float = None,
                 ef: float = None,
                 efSearch: float = None,
                 M: float = None,
                 NN: float = None,
                 delaunay_type: int = None,
                 post_processing: int = None,
                 skip_optimized_index: int = None,
                 desiredRecall=None, bucketSize=None, tuneK=None, tuneR=None, tuneQty=None, minExp=None, maxExp=None,
                 numPivot=None, numPivotIndex=None, numPivotSearch=None, hashTrickDim=None,
                 n_jobs: int = 1,
                 mmap_dir: str = "auto",
                 verbose: int = 0,
                 ):

        if nmslib is None:  # pragma: no cover
            raise ImportError(
                "Please install the nmslib package before using NMSlibTransformer.\n"
                "pip install nmslib\n"
                "For best performance, install from sources:\n"
                "pip install --no-binary :all: nmslib",
            ) from None

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.alpha = alpha
        self.method = method

        # HNSW and sw-graph parameters
        self.efConstruction = efConstruction
        self.ef = ef
        self.efSearch = efSearch
        self.M = M
        self.NN = NN
        self.delaunay_type = delaunay_type
        self.post_processing = post_processing
        self.skip_optimized_index = skip_optimized_index

        # vp-tree parameters
        self.desiredRecall = desiredRecall
        self.bucketSize = bucketSize
        self.tuneK = tuneK
        self.tuneR = tuneR
        self.tuneQty = tuneQty
        self.minExp = minExp
        self.maxExp = maxExp

        # napp parameters
        self.numPivot = numPivot
        self.numPivotIndex = numPivotIndex
        self.numPivotSearch = numPivotSearch
        self.hashTrickDim = hashTrickDim

        self.n_jobs = n_jobs
        self.mmap_dir = mmap_dir
        self.verbose = verbose

    def _construct_index_params_dict(self):
        if self.method in ["hnsw", "sw-graph"]:
            index_params = {
                "efConstruction": self.efConstruction,
                "delaunay_type": self.delaunay_type,
                "post": self.post_processing,
                "skip_optimized_index": self.skip_optimized_index,
                "indexThreadQty": self.n_jobs,

            }
            if self.method == "hnsw":
                index_params["ef"] = self.ef
                index_params["M"] = self.M
            else:
                index_params["efSearch"] = self.efSearch
                index_params["NN"] = self.NN
        elif self.method == "vp-tree":
            index_params = {
                "desiredRecall": self.desiredRecall,
                "bucketSize": self.bucketSize,
                "tuneK": self.tuneK,
                "tuneR": self.tuneR,
                "tuneQty": self.tuneQty,
                "minExp": self.minExp,
                "maxExp": self.maxExp,
            }
        elif self.method == "napp":
            index_params = {
                "numPivot": self.numPivot,
                "numPivotIndex": self.numPivotIndex,
                "numPivotSearch": self.numPivotSearch,
                "hashTrickDim": self.hashTrickDim,
            }
        elif self.method in ["simple_invindx", "brute_force"]:
            index_params = {}
        else:
            raise ValueError(f'Unknown method: {self.method}. Use one of: '
                             f'"hnsw", "sw-graph", "vp-tree", "napp", "simple_invindx", "brute_force"')
        # We only pass parameters that are explicitly provided by the user
        index_params = {k: v for k, v in index_params.items() if v is not None}
        return index_params

    def _possibly_mmap_index(self):
        # TODO create a MemMapMixin and move this code there
        if isinstance(self.mmap_dir, str):
            directory = "/dev/shm" if self.mmap_dir == "auto" else self.mmap_dir
            self.neighbor_index_ = create_tempfile_preferably_in_dir(
                prefix="skhubness_",
                suffix=".nmslib",
                directory=directory,
            )
            if self.mmap_dir == "auto":
                logging.warning(
                    f"The index will be stored in {self.neighbor_index_}. "
                    f"It will NOT be deleted automatically, when this instance is destructed.",
                )
        else:  # e.g. None
            self.mmap_dir = None

    def fit(self, X, y=None) -> NMSlibTransformer:
        """ Build the NMSLIB index and insert data from X.

        Parameters
        ----------
        X: array-like
            Data to be indexed
        y: ignored

        Returns
        -------
        self: NMSlibTransformer
            An instance of NMSlibTransformer with a built index
        """
        X: Union[np.ndarray, csr_matrix] = check_array(X, accept_sparse=True)  # noqa
        n_samples, n_features = X.shape
        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features

        space = {
            **{x: x for x in NMSlibTransformer.valid_metrics},
            "euclidean": "l2",
            "cosine": "cosinesimil",
        }.get(self.metric, None)
        if space is None:
            raise ValueError(f"Invalid metric: {self.metric}")
        self.space_ = space

        # Different nearest neighbor methods in NMSLIB have different parameters to tune,
        # and are passed as a dict to nmslib.init()
        self.index_params_ = self._construct_index_params_dict()

        # Save an index to disk or keep in memory, depending on self.mmap
        self._possibly_mmap_index()

        data_type = nmslib.DataType.DENSE_VECTOR
        dist_type = nmslib.DistType.FLOAT
        if "_sparse" in self.space_:
            data_type = nmslib.DataType.SPARSE_VECTOR

        self.data_type_ = data_type
        self.dist_type_ = dist_type

        # Some metrics require additional parameters
        space_params = {}
        if self.metric in ["lp", "lp_sparse"]:
            space_params["p"] = self.p
        self.space_params_ = space_params

        index = nmslib.init(
            method=self.method,
            space=space,
            space_params=self.space_params_,
            data_type=self.data_type_,
            dtype=self.dist_type_,
        )

        index.addDataPointBatch(X)
        index.createIndex(
            index_params=self.index_params_,
            print_progress=(self.verbose >= 2),
        )

        if self.mmap_dir is None:
            self.neighbor_index_ = index
        else:
            index.saveIndex(self.neighbor_index_, save_data=True)

        return self

    def transform(self, X) -> csr_matrix:
        """ Create k-neighbors graph for the query objects in X.

        Parameters
        ----------
        X : array-like
            Query objects

        Returns
        -------
        kneighbors_graph : csr_matrix
            The retrieved approximate nearest neighbors in the index for each query.
        """
        check_is_fitted(self, "neighbor_index_")
        X: Union[np.ndarray, csr_matrix] = check_array(X, accept_sparse=True)  # noqa

        n_samples_transform, n_features_transform = X.shape
        if n_features_transform != self.n_features_in_:
            raise ValueError(f"Shape of X ({n_features_transform} features) does not match "
                             f"shape of fitted data ({self.n_features_in_} features.")

        # Load memory-mapped nmslib.Index, unless it's already in main memory
        if isinstance(self.neighbor_index_, str):
            neighbor_index = nmslib.init(
                space=self.space_,
                space_params=self.space_params_,
                method=self.method,
                data_type=self.data_type_,
                dtype=self.dist_type_,
            )
            neighbor_index.loadIndex(self.neighbor_index_, load_data=True)
        else:
            neighbor_index = self.neighbor_index_

        # Do we ever need one additional neighbor (e.g., for self distances?)
        n_neighbors = self.n_neighbors

        results = neighbor_index.knnQueryBatch(
            X,
            k=n_neighbors,
            num_threads=self.n_jobs,
        )
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)
        distances = distances.astype(X.dtype)

        indptr = np.arange(
            start=0,
            stop=n_samples_transform * n_neighbors + 1,
            step=n_neighbors,
        )
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_in_),
        )

        return kneighbors_graph


class LegacyHNSW(ApproximateNearestNeighbor):
    """Wrapper for using nmslib

    Hierarchical navigable small-world graphs are data structures,
    that allow for approximate nearest neighbor search.
    Here, an implementation from nmslib is used.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are "angular", "euclidean", "manhattan", "hamming", "dot"
    method: str, default = 'hnsw',
        ANN method to use. Currently, only 'hnsw' is supported.
    post_processing: int, default = 2
        More post processing means longer index creation,
        and higher retrieval accuracy.
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose >= 2, show progress bar on indexing.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures
    """
    valid_metrics = ["euclidean", "l2", "minkowski", "squared_euclidean", "sqeuclidean",
                     "cosine", "cosinesimil"]

    def __init__(self, n_candidates: int = 5, metric: str = "euclidean",
                 method: str = "hnsw", post_processing: int = 2,
                 n_jobs: int = 1, verbose: int = 0):

        if nmslib is None:  # pragma: no cover
            raise ImportError("Please install the `nmslib` package, before using this class.\n"
                              "$ pip install nmslib") from None

        super().__init__(n_candidates=n_candidates,
                         metric=metric,
                         n_jobs=n_jobs,
                         verbose=verbose)
        self.method = method
        self.post_processing = post_processing
        self.space = None

    def fit(self, X, y=None) -> LegacyHNSW:
        """ Setup the HNSW index from training data.

        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored

        Returns
        -------
        self: LegacyHNSW
            An instance of HNSW with a built graph
        """
        X = check_array(X)

        method = self.method
        post_processing = self.post_processing

        if self.metric in ["euclidean", "l2", "minkowski", "squared_euclidean", "sqeuclidean"]:
            if self.metric in ["squared_euclidean", "sqeuclidean"]:
                self.metric = "sqeuclidean"
            else:
                self.metric = "euclidean"
            self.space = "l2"
        elif self.metric in ["cosine", "cosinesimil"]:
            self.space = "cosinesimil"
        else:
            raise ValueError(f'Invalid metric "{self.metric}". Please try "euclidean" or "cosine".')

        hnsw_index = nmslib.init(method=method,
                                 space=self.space)
        hnsw_index.addDataPointBatch(X)
        hnsw_index.createIndex({"post": post_processing,
                                "indexThreadQty": self.n_jobs,
                                },
                               print_progress=(self.verbose >= 2))
        self.index_ = hnsw_index
        self.n_samples_fit_ = len(self.index_)

        assert self.space in ["l2", "cosinesimil"], f"Internal: self.space={self.space} not allowed"

        return self

    def kneighbors(self, X: np.ndarray = None,
                   n_candidates: int = None,
                   return_distance: bool = True) -> Union[Tuple[np.array, np.array], np.array]:
        """ Retrieve k nearest neighbors.

        Parameters
        ----------
        X: np.array or None, optional, default = None
            Query objects. If None, search among the indexed objects.
        n_candidates: int or None, optional, default = None
            Number of neighbors to retrieve.
            If None, use the value passed during construction.
        return_distance: bool, default = True
            If return_distance, will return distances and indices to neighbors.
            Else, only return the indices.
        """
        check_is_fitted(self, ["index_", ])

        if X is None:
            raise NotImplementedError("Please provide X to hnsw.kneighbors().")

        # Check the n_neighbors parameter
        if n_candidates is None:
            n_candidates = self.n_candidates
        n_candidates = check_n_candidates(n_candidates)

        # Fetch the neighbor candidates
        neigh_ind_dist = self.index_.knnQueryBatch(X,
                                                   k=n_candidates,
                                                   num_threads=self.n_jobs)

        # If fewer candidates than required are found for a query,
        # we save index=-1 and distance=NaN
        n_test = X.shape[0]
        neigh_ind = -np.ones((n_test, n_candidates),
                             dtype=np.int32)
        neigh_dist = np.empty_like(neigh_ind,
                                   dtype=X.dtype) * np.nan

        for i, (ind, dist) in enumerate(neigh_ind_dist):
            neigh_ind[i, :ind.size] = ind
            neigh_dist[i, :dist.size] = dist

        if self.space == 'l2' and self.metric == 'sqeuclidean':
            neigh_dist **= 2

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
