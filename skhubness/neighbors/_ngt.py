# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer
# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations
from functools import partial
import logging
from pathlib import Path
from typing import Union, Tuple

try:
    import ngtpy
except ImportError:
    ngtpy = None  # pragma: no cover

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tqdm.auto import tqdm
from .approximate_neighbors import ApproximateNearestNeighbor
from ..utils.check import check_n_candidates
from ..utils.io import create_tempfile_preferably_in_dir

__all__ = [
    "LegacyNNG",
    "NGTTransformer",
]


class NGTTransformer(BaseEstimator, TransformerMixin):
    """ Approximate nearest neighbors retrieval with NGT.

    Compatible with sklearn's KNeighborsTransformer.
    NGT (Neighborhood Graph and Tree for Indexing High-dimensional Data) provides
    "commands and a library for performing high-speed approximate nearest neighbor searches
    against a large volume of data (several million to several 10 million items of data)
    in high dimensional vector data space (several ten to several thousand dimensions)" (Yahoo Japan).

    Parameters
    ----------
    n_neighbors : int, default = 5
        Number of neighbors to retrieve
    metric : str, default = "euclidean"
        Distance metric, allowed are "manhattan", "L1", "euclidean", "L2", "minkowski",
        "Angle", "Normalized Angle", "Hamming", "Jaccard", "Cosine" or "Normalized Cosine".
    optimize : bool, default = False
        Use ONNG method by optimizing the ANNG graph.
        May require long time for index creation.
    edge_size_for_creation : int, default = 80
        Increasing ANNG edge size improves retrieval accuracy at the cost of more time
    edge_size_for_search : int, default = 40
        Increasing ANNG edge size improves retrieval accuracy at the cost of more time
    epsilon : float, default 0.1
        Trade-off in ANNG between higher accuracy (larger epsilon) and shorter query time (smaller epsilon)
    num_incoming : int
        Number of incoming edges in ONNG graph
    num_outgoing : int
        Number of outgoing edges in ONNG graph
    n_jobs : int, default = 1
        Number of parallel jobs
    mmap_dir : str, default = 'auto'
        Memory-map the index to the given directory. This is required to make the class pickleable.

        - If None, keep everything in main memory (NON pickleable index),
        - if mmap_dir is a string, it is interpreted as a directory to store the index into,
        - if "auto", create a temp dir for the index, preferably in /dev/shm on Linux.
          Note: The directory/the index will NOT be deleted automatically.

    verbose : int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures

    Notes
    -----
    NGTTransformer stores the index to a directory specified in `mmap_dir`.
    The index is persistent, and will NOT be deleted automatically.
    It is the user's responsibility to take care of deletion, when required.
    """
    valid_metrics = [
        "manhattan",
        "L1",
        "euclidean",
        "L2",
        "minkowski",
        "Angle",
        "Normalized Angle",
        "Hamming",
        "Jaccard",
        "Cosine",
        "Normalized Cosine",
    ]
    _metric = {
        **{x: x for x in valid_metrics},
        "manhattan": "L1",
        "euclidean": "L2",
        "minkowski": "L2",
    }

    def __init__(
            self,
            n_neighbors: int = 5,
            metric: str = "euclidean",
            mmap_dir: str = "auto",
            optimize: bool = False,
            edge_size_for_creation: int = 80,
            edge_size_for_search: int = 40,
            num_incoming: int = -1,
            num_outgoing: int = -1,
            epsilon: float = 0.1,
            n_jobs: int = 1,
            verbose: int = 0,
    ):

        if ngtpy is None:  # pragma: no cover
            raise ImportError(
                "Please install the ngt package before using NGTTransformer.\n"
                "pip install ngt",
            ) from None

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.mmap_dir = mmap_dir
        self.optimize = optimize
        self.edge_size_for_creation = edge_size_for_creation
        self.edge_size_for_search = edge_size_for_search
        self.num_incoming = num_incoming
        self.num_outgoing = num_outgoing
        self.epsilon = epsilon
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _mmap_index(self):
        # Set up a directory to save the index to
        prefix = "skhubness_"
        suffix = ".ngt"
        if self.mmap_dir == "auto":
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix,
                suffix=suffix,
                directory="/dev/shm",
            )
            logging.warning(f"The index will be stored in {index_path}. "
                            f"It will NOT be deleted automatically, when this instance is destructed.")
        elif isinstance(self.mmap_dir, str):
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix,
                suffix=suffix,
                directory=self.mmap_dir,
            )
        elif self.mmap_dir is None:
            index_path = create_tempfile_preferably_in_dir(
                prefix=prefix,
                suffix=suffix,
            )
        else:
            raise TypeError("NGTTransformer requires to write an index to the filesystem. "
                            "Please provide a valid path with parameter `mmap_dir`.")
        return index_path

    def fit(self, X, y=None) -> NGTTransformer:
        """ Build the NGT index and insert data from X.

        Parameters
        ----------
        X: array-like
            Data to be indexed
        y: ignored

        Returns
        -------
        self: NGTTransformer
            An instance of NGTTransformer with a built index
        """
        X: np.ndarray = check_array(X)  # noqa
        n_samples, n_features = X.shape
        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features
        self.X_dtype_ = X.dtype

        metric = NGTTransformer._metric.get(self.metric, None)
        if metric is None:
            # Some metrics are capitalized, e.g., Hamming, Cosine, Normalized Cosine.
            metric = NGTTransformer._metric.get(str(self.metric).capitalize(), None)
            if metric is None:
                raise ValueError(f"Invalid metric: {self.metric}. Please use one of: {NGTTransformer.valid_metrics}")
        self.effective_metric_ = metric

        # Create the ANNG index, insert data
        index_path = self._mmap_index()
        ngtpy.create(
            path=index_path,
            dimension=self.n_features_in_,
            edge_size_for_creation=self.edge_size_for_creation,
            edge_size_for_search=self.edge_size_for_search,
            distance_type=self.effective_metric_,
        )
        index_obj = ngtpy.Index(index_path)
        index_obj.batch_insert(
            X,
            num_threads=self.n_jobs,
        )
        index_obj.save()

        # Convert ANNG to ONNG
        if self.optimize:
            optimizer = ngtpy.Optimizer(
                num_of_outgoings=self.num_outgoing,
                num_of_incomings=self.num_incoming,
                log_disabled=self.verbose < 1,
            )
            index_path_onng = str(Path(index_path).with_suffix(".ngtonng"))
            optimizer.execute(index_path, index_path_onng)
            index_path = index_path_onng

        # Keep index in memory or store in path
        if self.mmap_dir is None:
            self.neighbor_index_ = index_obj
        else:
            self.neighbor_index_ = index_path

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
        X: Union[np.ndarray, csr_matrix] = check_array(X, accept_sparse=False)  # noqa

        n_samples_transform, n_features_transform = X.shape
        if n_features_transform != self.n_features_in_:
            raise ValueError(f"Shape of X ({n_features_transform} features) does not match "
                             f"shape of fitted data ({self.n_features_in_} features.")

        if isinstance(self.neighbor_index_, str):
            index = ngtpy.Index(self.neighbor_index_)
        else:
            index = self.neighbor_index_

        n_neighbors = self.n_neighbors + 1

        tqdm_fmt = partial(
            tqdm,
            desc="NGT transform",
            disable=self.verbose < 1,
            unit=" vector",
            unit_scale=True,
        )
        indices = []
        distances = []
        for i, query in enumerate(tqdm_fmt(X)):
            response = index.search(
                query=query,
                size=n_neighbors,
                with_distance=True,
                epsilon=self.epsilon,
            )
            ind, dist = [np.array(arr) for arr in zip(*response)]
            indices.append(ind)
            distances.append(dist)

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


class LegacyNNG(BaseEstimator, ApproximateNearestNeighbor):
    """Wrapper for ngtpy and NNG variants.

    By default, the graph is an ANNG. Only when the `optimize` parameter is set,
    the graph is optimized to obtain an ONNG.

    Parameters
    ----------
    n_candidates: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = 'euclidean'
        Distance metric, allowed are 'manhattan', 'L1', 'euclidean', 'L2', 'minkowski',
        'Angle', 'Normalized Angle', 'Hamming', 'Jaccard', 'Cosine' or 'Normalized Cosine'.
    index_dir: str, default = 'auto'
        Store the index in the given directory.
        If None, keep the index in main memory (NON pickleable index),
        If index_dir is a string, it is interpreted as a directory to store the index into,
        if 'auto', create a temp dir for the index, preferably in /dev/shm on Linux.
        Note: The directory/the index will NOT be deleted automatically.
    optimize: bool, default = False
        Use ONNG method by optimizing the ANNG graph.
        May require long time for index creation.
    edge_size_for_creation: int, default = 80
        Increasing ANNG edge size improves retrieval accuracy at the cost of more time
    edge_size_for_search: int, default = 40
        Increasing ANNG edge size improves retrieval accuracy at the cost of more time
    epsilon: float, default 0.1
        Trade-off in ANNG between higher accuracy (larger epsilon) and shorter query time (smaller epsilon)
    num_incoming: int
        Number of incoming edges in ONNG graph
    num_outgoing: int
        Number of outgoing edges in ONNG graph
    n_jobs: int, default = 1
        Number of parallel jobs
    verbose: int, default = 0
        Verbosity level. If verbose > 0, show tqdm progress bar on indexing and querying.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures

    Notes
    -----
    LegacyNNG stores the index to a directory specified in `index_dir`.
    The index is persistent, and will NOT be deleted automatically.
    It is the user's responsibility to take care of deletion,
    when required.
    """
    valid_metrics = ["manhattan", "L1", "euclidean", "L2", "minkowski", "sqeuclidean",
                     "Angle", "Normalized Angle", "Cosine", "Normalized Cosine", "Hamming", "Jaccard"]
    internal_distance_type = {"manhattan": "L1",
                              "euclidean": "L2",
                              "minkowski": "L2",
                              "sqeuclidean": "L2",
                              }

    def __init__(self, n_candidates: int = 5,
                 metric: str = "euclidean",
                 index_dir: str = "auto",
                 optimize: bool = False,
                 edge_size_for_creation: int = 80,
                 edge_size_for_search: int = 40,
                 num_incoming: int = -1,
                 num_outgoing: int = -1,
                 epsilon: float = 0.1,
                 n_jobs: int = 1,
                 verbose: int = 0):

        if ngtpy is None:  # pragma: no cover
            raise ImportError("Please install the `ngt` package, before using this class.\n"
                              "$ pip3 install ngt") from None

        super().__init__(n_candidates=n_candidates,
                         metric=metric,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         )
        self.index_dir = index_dir
        self.optimize = optimize
        self.edge_size_for_creation = edge_size_for_creation
        self.edge_size_for_search = edge_size_for_search
        self.num_incoming = num_incoming
        self.num_outgoing = num_outgoing
        self.epsilon = epsilon

    def fit(self, X, y=None) -> LegacyNNG:
        """ Build the ngtpy.Index and insert data from X.

        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored

        Returns
        -------
        self: LegacyNNG
            An instance of NNG with a built index
        """
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
            self.y_train_ = y

        self.n_samples_fit_ = X.shape[0]
        self.n_features_in_ = X.shape[1]
        self.X_dtype_ = X.dtype

        # Map common distance names to names used by ngt
        try:
            self.effective_metric_ = LegacyNNG.internal_distance_type[self.metric]
        except KeyError:
            self.effective_metric_ = self.metric
        if self.effective_metric_ not in LegacyNNG.valid_metrics:
            if (metric_cap := str(self.effective_metric_).capitalize()) in LegacyNNG.valid_metrics:
                self.effective_metric_ = metric_cap
            else:
                raise ValueError(f"Unknown distance/similarity measure: {self.effective_metric_}. "
                                 f"Please use one of: {LegacyNNG.valid_metrics}.")

        # Set up a directory to save the index to
        prefix = "skhubness_"
        suffix = ".anng"
        if self.index_dir in ["auto"]:
            index_path = create_tempfile_preferably_in_dir(prefix=prefix,
                                                           suffix=suffix,
                                                           directory="/dev/shm")
            logging.warning(f"The index will be stored in {index_path}. "
                            f"It will NOT be deleted automatically, when this instance is destructed.")
        elif isinstance(self.index_dir, str):
            index_path = create_tempfile_preferably_in_dir(prefix=prefix,
                                                           suffix=suffix,
                                                           directory=self.index_dir)
        elif self.index_dir is None:
            index_path = create_tempfile_preferably_in_dir(prefix=prefix,
                                                           suffix=suffix)
        else:
            raise TypeError("LegacyNNG requires to write an index to the filesystem. "
                            "Please provide a valid path with parameter `index_dir`.")

        # Create the ANNG index, insert data
        ngtpy.create(path=index_path,
                     dimension=self.n_features_in_,
                     edge_size_for_creation=self.edge_size_for_creation,
                     edge_size_for_search=self.edge_size_for_search,
                     distance_type=self.effective_metric_,
                     )
        index_obj = ngtpy.Index(index_path)
        index_obj.batch_insert(X, num_threads=self.n_jobs)
        index_obj.save()

        # Convert ANNG top ONNG
        if self.optimize:
            optimizer = ngtpy.Optimizer()
            optimizer.set(num_of_outgoings=self.num_outgoing,
                          num_of_incomings=self.num_incoming)
            index_path_onng = str(Path(index_path).with_suffix(".onng"))
            optimizer.execute(index_path, index_path_onng)
            index_path = index_path_onng

        # Keep index in memory or store in path
        if self.index_dir is None:
            self.index_ = index_obj
        else:
            # index_obj.save()
            self.index_ = index_path

        return self

    def kneighbors(self, X=None, n_candidates=None, return_distance=True) -> Union[Tuple[np.array, np.array], np.array]:
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
        check_is_fitted(self, 'index_')
        if X is not None:
            X = check_array(X)

        n_test = self.n_samples_fit_ if X is None else X.shape[0]
        dtype = self.X_dtype_ if X is None else X.dtype

        if n_candidates is None:
            n_candidates = self.n_candidates
        n_candidates = check_n_candidates(n_candidates)

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        if X is None:
            n_neighbors = n_candidates + 1
            start = 1
        else:
            n_neighbors = n_candidates
            start = 0

        # If fewer candidates than required are found for a query,
        # we save index=-1 and distance=NaN
        neigh_ind = -np.ones((n_test, n_candidates),
                             dtype=np.int32)
        if return_distance:
            neigh_dist = np.empty_like(neigh_ind,
                                       dtype=dtype) * np.nan

        if isinstance(self.index_, str):
            index = ngtpy.Index(self.index_)
        else:
            index = self.index_

        disable_tqdm = False if self.verbose else True
        if X is None:
            for i in tqdm(range(n_test),
                          desc="Query LegacyNNG",
                          disable=disable_tqdm,
                          ):
                query = index.get_object(i)
                response = index.search(query=query,
                                        size=n_neighbors,
                                        with_distance=return_distance,
                                        epsilon=self.epsilon,
                                        )
                if return_distance:
                    ind, dist = [np.array(arr) for arr in zip(*response)]
                else:
                    ind = response
                ind = ind[start:]
                neigh_ind[i, :len(ind)] = ind
                if return_distance:
                    dist = dist[start:]
                    neigh_dist[i, :len(dist)] = dist
        else:  # if X was provided
            for i, x in tqdm(enumerate(X),
                             desc="Query LegacyNNG",
                             disable=disable_tqdm,
                             ):
                response = index.search(query=x,
                                        size=n_neighbors,
                                        with_distance=return_distance,
                                        epsilon=self.epsilon,
                                        )
                if return_distance:
                    ind, dist = [np.array(arr) for arr in zip(*response)]
                else:
                    ind = response
                ind = ind[start:]
                neigh_ind[i, :len(ind)] = ind
                if return_distance:
                    dist = dist[start:]
                    neigh_dist[i, :len(dist)] = dist

        if return_distance and self.metric == "sqeuclidean":
            neigh_dist **= 2

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
