"""Tests for transforms.py module.

TODO:
* Add random vectors and vector lengths
* Create global list of transforms
* Implement delta2 tests
* Implement transform_data tests

"""
import numpy as np
import pytest

from distrx.transforms import transform_delta


def test_transform_delta_input_len():
    """Raise ValueError if lengths of mu and sigma don't match."""
    for transform in ['log', 'logit', 'exp', 'expit']:
        with pytest.raises(ValueError):
            transform_delta([0.1]*2, [0.1]*3, transform)


def test_transform_delta_sigma():
    """Raise ValueError if `sigma` contains non-positive values."""
    for transform in ['log', 'logit', 'exp', 'expit', [np.sin, np.cos]]:
        vals = [0.1, -0.1]
        with pytest.raises(ValueError):
            transform_delta(vals, vals, transform)


def test_transform_delta_transform():
    """Raise ValueError for invalid `transform`."""
    with pytest.raises(ValueError):
        transform_delta([0.1], [0.1], 'dummy')


def test_transform_delta_output_type():
    """Output should be numpy arrays."""
    vals = [0.1]*2
    for transform in ['log', 'logit', 'exp', 'expit']:
        mu, sigma = transform_delta(vals, vals, transform)
        assert isinstance(mu, np.ndarray)
        assert isinstance(sigma, np.ndarray)


def test_transform_delta_outout_len():
    """Length of output vectors should match."""
    vals = [0.1]*2
    for transform in ['log', 'logit', 'exp', 'expit']:
        mu, sigma = transform_delta(vals, vals, transform)
        assert len(mu) == len(sigma)
