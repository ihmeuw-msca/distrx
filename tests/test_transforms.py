"""Tests for transforms.py module."""

import numpy as np
import pandas as pd
import pytest

from distrx.transforms import (
    delta_method,
    transform_data,
    transform_percentage_change,
    transform_percentage_change_counts1,
)

TRANSFORM_DICT = {
    "log": [np.log, lambda x: 1.0 / x],
    "logit": [lambda x: np.log(x / (1.0 - x)), lambda x: 1.0 / (x * (1.0 - x))],
    "exp": [np.exp, np.exp],
    "expit": [
        lambda x: 1.0 / (1.0 + np.exp(-x)),
        lambda x: np.exp(-x) / (1.0 + np.exp(-x)) ** 2,
    ],
}
TRANSFORM_LIST = list(TRANSFORM_DICT.keys())
FUNCTION_LIST = [transform_data, delta_method]
VALS = [0.1] * 2


@pytest.mark.parametrize("transform", TRANSFORM_LIST)
def test_method_name_valid(transform):
    """Raise ValueError for invalue `method`."""
    with pytest.raises(ValueError):
        transform_data(VALS, VALS, transform, method="dummy")


@pytest.mark.parametrize("function", FUNCTION_LIST)
@pytest.mark.parametrize("transform", TRANSFORM_LIST)
def test_input_len_match(function, transform):
    """Raise ValueError if lengths of input vectors don't match."""
    with pytest.raises(ValueError):
        function(VALS, VALS * 2, transform)


@pytest.mark.parametrize("function", FUNCTION_LIST)
@pytest.mark.parametrize("transform", TRANSFORM_LIST)
def test_sigma_negative(function, transform):
    """Raise ValueError if `sigma` contains negative values."""
    vals = VALS + [-0.1]
    with pytest.raises(ValueError):
        function(vals, vals, transform)


@pytest.mark.parametrize("function", FUNCTION_LIST)
@pytest.mark.parametrize("transform", TRANSFORM_LIST)
def test_sigma_zero(function, transform):
    """Display warning if `sigma` contains zeros."""
    vals = VALS + [0.0]
    with pytest.warns(UserWarning):
        function(vals, vals, transform)


@pytest.mark.parametrize("function", FUNCTION_LIST)
def test_transform_name_valid(function):
    """Raise ValueError for invalid `transform`."""
    with pytest.raises(ValueError):
        function(VALS, VALS, "dummy")


@pytest.mark.parametrize("function", FUNCTION_LIST)
@pytest.mark.parametrize("transform", TRANSFORM_LIST)
def test_output_type(function, transform):
    """Output should be numpy arrays."""
    mu, sigma = function(VALS, VALS, transform)
    assert isinstance(mu, np.ndarray)
    assert isinstance(sigma, np.ndarray)


@pytest.mark.parametrize("function", FUNCTION_LIST)
@pytest.mark.parametrize("transform", TRANSFORM_LIST)
def test_outout_len_match(function, transform):
    """Length of output vectors should match."""
    mu, sigma = function(VALS, VALS, transform)
    assert len(mu) == len(sigma)


@pytest.mark.parametrize("transform", TRANSFORM_LIST)
def test_delta_result(transform):
    """Check expected results."""
    mu = np.random.uniform(0.1, 1.0, size=10)
    sigma = np.random.uniform(0.1, 1.0, size=10)
    mu_ref = TRANSFORM_DICT[transform][0](mu)
    sigma_ref = sigma * TRANSFORM_DICT[transform][1](mu)
    mu_trans, sigma_trans = delta_method(mu, sigma, transform)
    assert np.allclose(mu_trans, mu_ref)
    assert np.allclose(sigma_trans, sigma_ref)


def test_percentage_change():
    x = np.random.normal(1, 0.1, 1000)
    y = np.random.normal(1.1, 0.1, 1000)
    z = np.random.normal(1, 0.1, 1001)
    p, sigma = transform_percentage_change(x, y)
    assert 0 < p and p < 1
    assert 0 < sigma and sigma < 1
    with pytest.raises(ValueError):
        transform_percentage_change(x, z)


def test_percentage_change_counts():
    x = np.random.choice([0, 1], size=1000, p=[0.1, 0.9])
    y = np.random.choice([0, 1], size=1100, p=[0.2, 0.8])
    mu, sigma = transform_percentage_change_counts1(
        (x == 1).sum(), len(x), (y == 1).sum(), len(y)
    )
    assert -1 <= mu and mu < np.inf
    assert 0 < sigma and sigma < 1


def test_percentage_change_input():
    # scalar input
    c_x, n_x = 100, 1000
    c_y, n_y = 200, 1050
    # with pytest.raises(TypeError):
    transform_percentage_change_counts1(c_x, n_x, c_y, n_y)

    # base list input
    c_x = [100, 200]
    n_x = [1000, 1000]
    c_y = [300, 400]
    n_y = [1050, 1050]
    # with pytest.raises(TypeError):
    transform_percentage_change_counts1(c_x, n_x, c_y, n_y)

    # dataframe input
    df = pd.DataFrame({"c_x": c_x, "n_x": n_x, "c_y": c_y, "n_y": n_y})
    # with pytest.raises(TypeError):
    transform_percentage_change_counts1(
        df["c_x"], df["n_x"], df["c_y"], df["n_y"]
    )
