from itertools import product
import pytest
from skhubness.analysis import Hubness
from skhubness.data import load_dexter
from skhubness.neighbors import kneighbors_graph

HUBNESS_ALGORITHMS = ('mp',
                      'ls',
                      )
MP_PARAMS = tuple({'method': method} for method in ['normal', 'empiric'])
LS_PARAMS = tuple({'method': method} for method in ['standard', 'nicdm'])
HUBNESS_ALGORITHMS_WITH_PARAMS = ((*product(['mp'], MP_PARAMS),
                                   *product(['ls'], LS_PARAMS),
                                   ))


@pytest.mark.parametrize('hubness_param', HUBNESS_ALGORITHMS_WITH_PARAMS)
@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_neighbors_dexter(hubness_param, metric):
    hubness, param = hubness_param
    X, y = load_dexter()

    # Hubness in standard spaces
    hub = Hubness(k=10, metric=metric).estimate(X)
    k_skew_orig = hub.k_skewness_

    # Hubness in secondary distance spaces (after hub. red.)
    graph = kneighbors_graph(X, n_neighbors=10, metric=metric,
                             hubness=hubness, hubness_params=param)
    hub = Hubness(k=10, metric='precomputed')
    hub.estimate(graph, has_self_distances=True)
    k_skew_hr = hub.k_skewness_

    assert k_skew_hr < k_skew_orig * 8/10,\
        f'k-occurrence skewness was not reduced by at least 20% for dexter with {hubness}'
