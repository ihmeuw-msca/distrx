"""Tests for transforms.py module."""
import types

import pytest

from distrx.transforms import get_transform


def test_transform_transform_value():
    """Raise ValueError for invalid `transform`."""
    for order in [0, 1, 2]:
        with pytest.raises(ValueError):
            get_transform('dummy', order)


def test_transform_order_value():
    """Raise ValueError for invalid `order`."""
    for transform in ['log', 'logit']:
        with pytest.raises(ValueError):
            get_transform(transform, 3)


def test_transform_ouput_len():
    """Length of output should correspond to `order`."""
    for transform in ['log', 'logit']:
        assert len(get_transform(transform, 1)) == 2
        assert len(get_transform(transform, 2)) == 3


def test_transform_output_type():
    """Type of output should correspond to `order`."""
    for transform in ['log', 'logit']:
        assert isinstance(get_transform(transform), types.FunctionType)
        assert isinstance(get_transform(transform, 0), types.FunctionType)
        for order in [1, 2]:
            for function in get_transform(transform, order):
                assert isinstance(function, types.FunctionType)
