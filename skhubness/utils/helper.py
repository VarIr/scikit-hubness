import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "k_neighbors_graph",
]


# TODO should probably go to `neighbors` subpackage
def k_neighbors_graph(
        hub_reduced_dist: np.ndarray,
        original_X: csr_matrix,
        sort_distances: bool = True,
) -> csr_matrix:
    """ Create a possibly sorted sparse kneighbors graph from hubness-reduced distances and indices.

    Parameters
    ----------
    hub_reduced_dist : array-like of shape (n_query, n_neighbors)
        Array of (usually hubness-reduced) distances.
    original_X : csr_matrix of shape (n_query, n_indexed)
        The original k-neighbors graph (usually from before hubness reduction);
        Might be sorted according to `hub_reduced_dist`.
    sort_distances : bool, default = True
        Sort the new k-neighbors graph

    Returns
    -------
    kneighbors_graph : csr_matrix
    """
    n_query, _ = original_X.shape
    hub_reduced_ind = original_X.indices.reshape(n_query, -1)

    if sort_distances:
        sorted_ind = np.argsort(hub_reduced_dist, axis=1)
        hub_reduced_dist = np.take_along_axis(hub_reduced_dist, sorted_ind, axis=1)
        hub_reduced_ind = np.take_along_axis(hub_reduced_ind, sorted_ind, axis=1)
        del sorted_ind

    # Construct CSR matrix k-neighbors graph
    A_data = hub_reduced_dist.ravel()
    A_indices = hub_reduced_ind.ravel()
    A_indptr = original_X.indptr

    kneighbors_graph = csr_matrix(
        (A_data, A_indices, A_indptr),
        shape=original_X.shape,
    )

    return kneighbors_graph
