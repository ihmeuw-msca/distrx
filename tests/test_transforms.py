"""Tests for transforms.py module."""
import numpy as np
import pytest

from distrx.transforms import transform_data, delta_method


TRANSFORM_LIST = ['log', 'logit', 'exp', 'expit']
FUNCTION_LIST = [transform_data, delta_method]
VALS = [0.1]*2


@pytest.mark.parametrize('transform', TRANSFORM_LIST)
def test_method_name_valid(transform):
    """Raise ValueError for invalue `method`."""
    with pytest.raises(ValueError):
        transform_data(VALS, VALS, transform, method='dummy')


@pytest.mark.parametrize('function', FUNCTION_LIST)
def test_input_len_match(function):
    """Raise ValueError if lengths of input vectors don't match."""
    for transform in TRANSFORM_LIST:
        with pytest.raises(ValueError):
            function(VALS, VALS*2, transform)


@pytest.mark.parametrize('function', FUNCTION_LIST)
def test_sigma_negative(function):
    """Raise ValueError if `sigma` contains negative values."""
    vals = VALS + [-0.1]
    for transform in TRANSFORM_LIST:
        with pytest.raises(ValueError):
            function(vals, vals, transform)


@pytest.mark.parametrize('function', FUNCTION_LIST)
def test_sigma_zero(function):
    """Display warning if `sigma` contains zeros."""
    vals = VALS + [0.0]
    for transform in TRANSFORM_LIST:
        with pytest.warns(UserWarning):
            function(vals, vals, transform)


@pytest.mark.parametrize('function', FUNCTION_LIST)
def test_transform_name_valid(function):
    """Raise ValueError for invalid `transform`."""
    with pytest.raises(ValueError):
        function(VALS, VALS, 'dummy')


@pytest.mark.parametrize('function', FUNCTION_LIST)
def test_output_type(function):
    """Output should be numpy arrays."""
    for transform in TRANSFORM_LIST:
        mu, sigma = function(VALS, VALS, transform)
        assert isinstance(mu, np.ndarray)
        assert isinstance(sigma, np.ndarray)


@pytest.mark.parametrize('function', FUNCTION_LIST)
def test_outout_len_match(function):
    """Length of output vectors should match."""
    for transform in TRANSFORM_LIST:
        mu, sigma = function(VALS, VALS, transform)
        assert len(mu) == len(sigma)
