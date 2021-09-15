"""Transform data from one space to another.

Transform data, in the form of sample statistics and their standard
errors, from one space to another using a given transform function.

TODO:
* Add user-defined transform function
* Add functions for confidence intervals
* Add decorators for accepting floats or vectors

"""
from typing import Tuple

import numpy as np
import numpy.typing as npt


TRANSFORM_DICT = {
    'log': [
        np.log,
        lambda x: 1/x
    ], 'logit': [
        lambda x: np.log(x/(1 - x)),
        lambda x: 1/(x*(1 - x))
    ], 'exp': [
        np.exp,
        np.exp
    ], 'expit': [
        lambda x: 1/(1 + np.exp(-x)),
        lambda x: np.exp(-x)/(1 + np.exp(-x))**2
    ]
}
METHOD_LIST = ['delta']


def transform_data(mu: npt.ArrayLike, sigma: npt.ArrayLike, transform: str,
                   method: str = 'delta') -> Tuple[np.ndarray, np.ndarray]:
    """Transform data from one space to another.

    Transform data, in the form of sample statistics and their standard
    errors, from one space to another using a given transform function.
    No assumptions are made about the underlying distributions of the
    given data.

    Parameters
    ----------
    mu : array_like
        Sample statistics.
    sigma : array_like
        Standard errors.
    transform : {'log', 'logit', 'exp', 'expit'}
        Transform function.
    method : {'delta'}, optional
        Method used to transform data.

    Returns
    -------
    mu_trans : numpy.ndarray
        Sample stastistics in the transform space.
    sigma_trans : numpy.ndarray
        Standard errors in the transform space.

    """
    mu, sigma = np.array(mu), np.array(sigma)
    _check_input(mu, sigma, transform, method)
    if method == 'delta':
        return delta_method(mu, sigma, transform)


def delta_method(mu: npt.ArrayLike, sigma: npt.ArrayLike, transform: str) -> \
                 Tuple[np.ndarray, np.ndarray]:
    """Transform data using the delta method.

    Transform data, in the form of sample statistics and their standard
    errors, from one space to another using a given transform function
    and the delta method. No assumptions are made about the underlying
    distributions of the given data.

    Parameters
    ----------
    mu : array_like
        Sample statistics.
    sigma : array_like
        Standard errors.
    transform : {'log', 'logit', 'exp', 'expit'}
        Transform function.

    Returns
    -------
    mu_trans : numpy.ndarray
        Sample statistics in the transform space.
    sigma_trans : numpy.ndarray
        Standard errors in the transform space.

    Notes
    -----
    The delta method expands a function of a random variable about its
    mean with a one-step Taylor approximation and then takes the
    variance.

    """
    mu, sigma = np.array(mu), np.array(sigma)
    _check_input(mu, sigma, transform, 'delta')
    mu_trans = TRANSFORM_DICT[transform][0](mu)
    sigma_trans = sigma*TRANSFORM_DICT[transform][1](mu)
    return mu_trans, sigma_trans


def _check_input(mu: npt.ArrayLike, sigma: npt.ArrayLike, transform: str,
                 method: str) -> None:
    """Run checks on input data.

    Parameters
    ----------
    mu : array_like
        Sample statistics.
    sigma : array_like
        Standard errors.
    transform : {'log', 'logit', 'exp', 'expit'}
        Transform function.
    method : {'delta'}
        Method used to transform data.

    """
    _check_lengths_match(mu, sigma)
    _check_sigma_positive(sigma)
    _check_transform_valid(transform)
    _check_method_valid(method)


def _check_lengths_match(mu: npt.ArrayLike, sigma: npt.ArrayLike) -> None:
    """Check that `mu` and `sigma` have the same lengths.

    Parameters
    ----------
    mu : array_like
        Sample statistics.
    sigma : array_like
        Standard errors.

    """
    if len(mu) != len(sigma):
        raise ValueError("Lengths of mu and sigma don't match.")


def _check_sigma_positive(sigma: npt.ArrayLike) -> None:
    """Check that `sigma` is positive.

    Parameters
    ----------
    sigma : array_like
        Standard errors.

    """
    if np.any(sigma <= 0):
        raise ValueError("Sigma values must be positive.")


def _check_transform_valid(transform: str) -> None:
    """Check that `transform` is in TRANSFORM_DICT.

    Parameters
    ----------
    transform : {'log', 'logit', 'exp', 'expit'}
        Transform function.

    """
    if transform not in TRANSFORM_DICT:
        raise ValueError(f"Invalid transform '{transform}'.")


def _check_method_valid(method: str) -> None:
    """Check that `method` is in ['delta', 'delta2'].

    Parameters
    ----------
    method : {'delta', 'delta2'}
        Method used to transform data.

    """
    if method not in METHOD_LIST:
        raise ValueError(f"Invalid method '{method}'.")
