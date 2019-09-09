# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Author: Roman Feldbauer (adaptions for scikit-hubness)
# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations
import logging
from typing import Union, Tuple
try:
    import ngtpy
except (ImportError, ModuleNotFoundError) as e:
    logging.warning("The package 'ngt' is required to run this example.")  # pragma: no cover
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tqdm.auto import tqdm
from .approximate_neighbors import ApproximateNearestNeighbor
from ..utils.check import check_n_candidates
from ..utils.io import create_tempfile_preferably_in_dir

print(__doc__)

__all__ = ['ONNG',
           ]


class ONNG(BaseEstimator, ApproximateNearestNeighbor):
    """Wrapper for ngtpy and ONNG

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
    ONNG stores the index to a directory specified in `index_dir`.
    The index is persistent, and will NOT be deleted automatically.
    It is the user's responsibility to take care of deletion,
    when required.
    """
    valid_metrics = ['manhattan', 'L1', 'euclidean', 'L2', 'minkowski', 'sqeuclidean',
                     'Angle', 'Normalized Angle', 'Cosine', 'Normalized Cosine', 'Hamming', 'Jaccard']
    internal_distance_type = {'manhattan': 'L1',
                              'euclidean': 'L2',
                              'minkowski': 'L2',
                              'sqeuclidean': 'L2',
                              }

    def __init__(self, n_candidates: int = 5,
                 metric: str = 'euclidean',
                 index_dir: str = 'auto',
                 edge_size_for_creation: int = 40,
                 edge_size_for_search: int = 10,
                 n_jobs: int = 1,
                 verbose: int = 0):
        super().__init__(n_candidates=n_candidates,
                         metric=metric,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         )
        self.index_dir = index_dir
        self.edge_size_for_creation = edge_size_for_creation
        self.edge_size_for_search = edge_size_for_search

    def fit(self, X, y=None) -> ONNG:
        """ Build the ngtpy.Index and insert data from X.

        Parameters
        ----------
        X: np.array
            Data to be indexed
        y: any
            Ignored

        Returns
        -------
        self: ONNG
            An instance of ONNG with a built index
        """
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
            self.y_train_ = y

        self.n_samples_train_ = X.shape[0]
        self.n_features_ = X.shape[1]
        self.X_dtype_ = X.dtype

        # Map common distance names to names used by ngt
        try:
            self.effective_metric_ = ONNG.internal_distance_type[self.metric]
        except KeyError:
            self.effective_metric_ = self.metric
        if self.effective_metric_ not in ONNG.valid_metrics:
            raise ValueError(f'Unknown distance/similarity measure: {self.effective_metric_}. '
                             f'Please use one of: {ONNG.valid_metrics}.')

        # Set up a directory to save the index to
        if self.index_dir in ['auto']:
            index_path = create_tempfile_preferably_in_dir(prefix='skhubness_',
                                                           suffix='.onng',
                                                           directory='/dev/shm')
            logging.warning(f'The index will be stored in {index_path}. '
                            f'It will NOT be deleted automatically, when this instance is destructed.')
        elif isinstance(self.index_dir, str):
            index_path = create_tempfile_preferably_in_dir(prefix='skhubness_',
                                                           suffix='.onng',
                                                           directory=self.index_dir)
        elif self.index_dir is None:
            index_path = create_tempfile_preferably_in_dir(prefix='skhubness_',
                                                           suffix='.onng')
        else:
            raise TypeError(f'ONNG requires to write an index to the filesystem. '
                            f'Please provide a valid path with parameter `index_dir`.')

        # Create the ONNG index, insert data
        # TODO add ngt optimizer
        ngtpy.create(path=index_path,
                     dimension=self.n_features_,
                     edge_size_for_creation=self.edge_size_for_creation,
                     edge_size_for_search=self.edge_size_for_search,
                     distance_type=self.effective_metric_,
                     )
        index_obj = ngtpy.Index(index_path)
        index_obj.batch_insert(X, num_threads=self.n_jobs)

        # Keep index in memory or store in path
        if self.index_dir is None:
            self.index_ = index_obj
        else:
            index_obj.save()
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

        n_test = self.n_samples_train_ if X is None else X.shape[0]
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

        if X is None:
            if self.verbose:
                query_ind = tqdm(range(n_test),
                                 desc='Query ONNG',
                                 total=n_test,
                                 )
            else:
                query_ind = range(n_test)
            for i in query_ind:
                query = index.get_object(i)
                response = index.search(query=query,
                                        size=n_neighbors,
                                        with_distance=return_distance,
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
            if self.verbose:
                enumerate_X = tqdm(enumerate(X),
                                   desc='Query ONNG',
                                   total=n_test,
                                   )
            else:
                enumerate_X = enumerate(X)
            for i, x in enumerate_X:
                response = index.search(query=x,
                                        size=n_neighbors,
                                        with_distance=return_distance,
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

        if return_distance and self.metric == 'sqeuclidean':
            neigh_dist **= 2

        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind
