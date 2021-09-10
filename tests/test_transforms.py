"""Tests for transforms.py module.

TODO:
* Add random vectors and vector lengths
* Implement delta2 tests
* Implement transform_data tests

"""
import numpy as np
import pytest

from distrx.transforms import transform_data, transform_delta


TRANSFORM_LIST = ['log', 'logit', 'exp', 'expit']
FUNCTION_LIST = [transform_data, transform_delta]


@pytest.mark.parametrize("transform", TRANSFORM_LIST)
def test_method_name_valid(transform):
    """Raise ValueError for invalue `method`."""
    vals = [0.1]*2
    with pytest.raises(ValueError):
        transform_data(vals, vals, transform, method='dummy') 


@pytest.mark.parametrize("function", FUNCTION_LIST)
def test_input_len_match(function):
    """Raise ValueError if lengths of input vectors don't match."""
    for transform in TRANSFORM_LIST:
        with pytest.raises(ValueError):
            function([0.1]*2, [0.1]*3, transform)


@pytest.mark.parametrize("function", FUNCTION_LIST)
def test_sigma_positive(function):
    """Raise ValueError if `sigma` contains non-positive values."""
    for transform in TRANSFORM_LIST:
        vals = [0.1, -0.1]
        with pytest.raises(ValueError):
            transform_delta(vals, vals, transform)


@pytest.mark.parametrize("function", FUNCTION_LIST)
def test_transform_name_valid(function):
    """Raise ValueError for invalid `transform`."""
    with pytest.raises(ValueError):
        transform_delta([0.1], [0.1], 'dummy')


@pytest.mark.parametrize("function", FUNCTION_LIST)
def test_output_type(function):
    """Output should be numpy arrays."""
    vals = [0.1]*2
    for transform in TRANSFORM_LIST:
        mu, sigma = transform_delta(vals, vals, transform)
        assert isinstance(mu, np.ndarray)
        assert isinstance(sigma, np.ndarray)


@pytest.mark.parametrize("function", FUNCTION_LIST)
def test_outout_len_match(function):
    """Length of output vectors should match."""
    vals = [0.1]*2
    for transform in TRANSFORM_LIST:
        mu, sigma = transform_delta(vals, vals, transform)
        assert len(mu) == len(sigma)
