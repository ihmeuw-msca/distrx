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
@pytest.mark.parametrize('transform', TRANSFORM_LIST)
def test_input_len_match(function, transform):
    """Raise ValueError if lengths of input vectors don't match."""
    with pytest.raises(ValueError):
        function(VALS, VALS*2, transform)


@pytest.mark.parametrize('function', FUNCTION_LIST)
@pytest.mark.parametrize('transform', TRANSFORM_LIST)
def test_sigma_negative(function, transform):
    """Raise ValueError if `sigma` contains negative values."""
    vals = VALS + [-0.1]
    with pytest.raises(ValueError):
        function(vals, vals, transform)


@pytest.mark.parametrize('function', FUNCTION_LIST)
@pytest.mark.parametrize('transform', TRANSFORM_LIST)
def test_sigma_zero(function, transform):
    """Display warning if `sigma` contains zeros."""
    vals = VALS + [0.0]
    with pytest.warns(UserWarning):
        function(vals, vals, transform)


@pytest.mark.parametrize('function', FUNCTION_LIST)
def test_transform_name_valid(function):
    """Raise ValueError for invalid `transform`."""
    with pytest.raises(ValueError):
        function(VALS, VALS, 'dummy')


@pytest.mark.parametrize('function', FUNCTION_LIST)
@pytest.mark.parametrize('transform', TRANSFORM_LIST)
def test_output_type(function, transform):
    """Output should be numpy arrays."""
    mu, sigma = function(VALS, VALS, transform)
    assert isinstance(mu, np.ndarray)
    assert isinstance(sigma, np.ndarray)


@pytest.mark.parametrize('function', FUNCTION_LIST)
@pytest.mark.parametrize('transform', TRANSFORM_LIST)
def test_outout_len_match(function, transform):
    """Length of output vectors should match."""
    mu, sigma = function(VALS, VALS, transform)
    assert len(mu) == len(sigma)
