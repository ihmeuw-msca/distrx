"""Transform data from one space to another.

Transform data, in the form of sample statistics and their standard
errors, from one space to another using a given transform function.

TODO:
* Implement transform_delta2
* Implement transform_data
* Add user-defined transform function

"""
from typing import Tuple

import numpy as np
import numpy.typing as npt


TRANSFORM_DICT = {
    'log': [
        np.log,
        lambda x: 1/x,
        lambda x: -1/x**2,
    ], 'logit': [
        lambda x: np.log(x/(1 - x)),
        lambda x: 1/(x*(1 - x)),
        lambda x: (2*x - 1)/(x**2*(1 - x)**2)
    ], 'exp': [
        np.exp,
        np.exp,
        np.exp
    ], 'expit': [
        lambda x: 1/(1 + np.exp(-x)),
        lambda x: np.exp(-x)/(1 + np.exp(-x))**2,
        lambda x: np.exp(-x)*(np.exp(-x) - 1)/(1 + np.exp(-x))**3
    ]
}


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
    method : {'delta, 'delta2'}, optional
        Method used to transform data.

    Returns
    -------
    mu_trans : numpy.ndarray
        Sample stastistics in the transform space.
    sigma_trans : numpy.ndarray
        Standard errors in the transform space.

    """
    mu, sigma = np.array(mu), np.array(sigma)
    check_input(mu, sigma, transform, method)
    if method == 'delta':
        return transform_delta(mu, sigma, transform)


def transform_delta(mu: npt.ArrayLike, sigma: npt.ArrayLike,
                    transform: str) -> Tuple[np.ndarray, np.ndarray]:
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
    check_input(mu, sigma, transform)
    mu_trans = TRANSFORM_DICT[transform][0](mu)
    sigma_trans = sigma*TRANSFORM_DICT[transform][1](mu)**2
    return mu_trans, sigma_trans


def transform_delta2(mu: npt.ArrayLike, sigma: npt.ArrayLike,
                     transform: str) -> Tuple[np.ndarray, np.ndarray]:
    """Transform data using the second-order delta method.

    Transform data, in the form of sample statistics and their standard
    errors, from one space to another using a given transform function
    and the second-order delta method. No assumptions are made about
    the underlying distributions of the given data.

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
    The second-order delta method expands a function of a random
    variable about its mean with a two-step Taylor approximation and
    then takes the variance. This method is useful if the derivative of
    the transform function is zero (so the first-order delta method
    cannot be applied), or the sample size is small.

    """
    mu, sigma = np.array(mu), np.array(sigma)
    check_input(mu, sigma, transform)
    return


def check_input(mu: npt.ArrayLike, sigma: npt.ArrayLike, transform: str,
                method: str = None) -> None:
    """Run checks on input data.

    Parameters
    ----------
    mu : array_like
        Sample statistics.
    sigma : array_like
        Standard errors.
    transform : {'log', 'logit', 'exp', 'expit'}
        Transform function.
    method : {None, 'delta', 'delta2'}, optional
        Method used to transform data.

    """
    check_lengths_match(mu, sigma)
    check_sigma_positive(sigma)
    check_transform_valid(transform)
    if method is not None:
        check_method_valid(method)


def check_lengths_match(mu: npt.ArrayLike, sigma: npt.ArrayLike) -> None:
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


def check_sigma_positive(sigma: npt.ArrayLike) -> None:
    """Check that `sigma` is positive.

    Parameters
    ----------
    sigma : array_like
        Standard errors.

    """
    if np.any(sigma <= 0):
        raise ValueError("Sigma values must be positive.")


def check_transform_valid(transform: str) -> None:
    """Check that `transform` is in TRANSFORM_DICT.

    Parameters
    ----------
    transform : {'log', 'logit', 'exp', 'expit'}
        Transform function.

    """
    if transform not in TRANSFORM_DICT:
        raise ValueError(f"Invalid transform '{transform}'.")


def check_method_valid(method: str) -> None:
    """Check that `method` is in ['delta', 'delta2'].

    Parameters
    ----------
    method : {'delta', 'delta2'}
        Method used to transform data.

    """
    if method not in ['delta', 'delta2']:
        raise ValueError(f"Invalid method '{method}'.")
