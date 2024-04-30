"""Transform data from one space to another.

Transform data, in the form of sample statistics and their standard
errors, from one space to another using a given transform function.

TODO:
* Add user-defined transform function
* Add functions for confidence intervals
* Add decorators for accepting floats or vectors

"""
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt


class FirstOrder:
    def __init__(self, transform: str, mu: npt.ArrayLike, sigma: npt.ArrayLike) -> None:
        # self.transforms = ["log", "logit", "exp", "expit"]
        self.transform = input(transform)
        match self.transform:
            case "log":
                self.mu_trans, self.sigma_trans = self.log_trans(mu, sigma)
            case "logit":
                self.mu_trans, self.sigma_trans = self.logit_trans(mu, sigma)
            case "exp":
                self.mu_trans, self.sigma_trans = self.exp_trans(mu, sigma)
            case "expit":
                self.mu_trans, self.sigma_trans = self.expit_trans(mu, sigma)
            case _:
                raise ValueError(f"Invalid transform '{transform}'.")

    def log_trans(mu: npt.ArrayLike, sigma: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        return np.log(mu), sigma / mu

    def logit_trans(mu: npt.ArrayLike, sigma: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        return np.log(mu / (1.0 - mu)), sigma / (mu * (1.0 - mu))

    def exp_trans(mu: npt.ArrayLike, sigma: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        return np.exp(mu), np.exp(mu)

    def expit_trans(mu: npt.ArrayLike, sigma: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        return 1.0 / (1.0 + np.exp(-mu)), sigma * np.exp(-mu) / (1.0 + np.exp(-mu)) ** 2

    def get_mu_trans(self):
        return self.mu_trans

    def get_sigma_trans(self):
        return self.sigma_trans


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
    _check_input(method, transform, mu, sigma)
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
    _check_input('delta', transform, mu, sigma)
    transformer = FirstOrder(transform, mu, sigma)
    return transformer.get_mu_trans(), transformer.get_sigma_trans()


def _check_input(method: str, transform: str, mu: npt.ArrayLike,
                 sigma: npt.ArrayLike) -> None:
    """Run checks on input data.

    Parameters
    ----------
    method : {'delta'}
        Method used to transform data.
    transform : {'log', 'logit', 'exp', 'expit'}
        Transform function.
    mu : array_like
        Sample statistics.
    sigma : array_like
        Standard errors.

    """
    _check_method_valid(method)
    _check_lengths_match(mu, sigma)
    _check_sigma_positive(sigma)


def _check_method_valid(method: str) -> None:
    """Check that `method` is in METHOD_LIST.

    Parameters
    ----------
    method : {'delta'}
        Method used to transform data.

    """
    if method not in METHOD_LIST:
        raise ValueError(f"Invalid method '{method}'.")


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
    if np.any(sigma == 0.0):
        warnings.warn("Sigma vector contains zeros.")
    if np.any(sigma < 0.0):
        raise ValueError("Sigma values must be positive.")
