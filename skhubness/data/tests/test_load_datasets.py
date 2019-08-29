
from skhubness.data import load_dexter


def test_load_dexter():
    X, y = load_dexter()
    n_samples = 300
    n_features = 20_000
    assert X.shape == (n_samples, n_features), f'Wrong shape: X.shape = {X.shape}, should be (300, 20_000).'
    assert y.shape == (n_samples, ), f'Wrong shape: y.shape = {y.shape}, should be (300, ).'
