"""Transform data from one space to another.

Transform data, in the form of sample statistics and their standard
errors, from one space to another using a given transform function.

TODO:
* Add user-defined transform function
* Add functions for confidence intervals
* Add decorators for accepting floats or vectors

"""

import warnings
from typing import Tuple

import numpy as np
import numpy.typing as npt
from msca.c2fun import c2fun_dict


class FirstOrder:
    """
    Contains 4 most common 1st order delta method transformation functions
    """

    def __init__(self, transform: str) -> None:
        """Initializes an object to perform 1st order delta method transformations

        Parameters
        ----------
        transform : str
            Function of choice

        Raises
        ------
        ValueError
            Is thrown function of choice not implemented
        """
        self.transform = transform

    def __call__(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        match self.transform:
            case "log":
                return self.log_trans(mu, sigma)
            case "logit":
                return self.logit_trans(mu, sigma)
            case "exp":
                return self.exp_trans(mu, sigma)
            case "expit":
                return self.expit_trans(mu, sigma)
            case _:
                raise ValueError(f"Invalid transform '{self.transform}'.")

    def log_trans(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs delta method on data under log transform

        Parameters
        ----------
        mu : npt.ArrayLike
            Sample statistics
        sigma : npt.ArrayLike
            Standard errors

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed mean and standard error
        """
        log = c2fun_dict["log"]
        # log(mu), sigma / mu
        return log(mu), sigma * log(mu, order=1)

    def logit_trans(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs delta method on data under logit transform

        Parameters
        ----------
        mu : npt.ArrayLike
            Sample statistics
        sigma : npt.ArrayLike
            Standard errors

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed mean and standard error
        """
        logit = c2fun_dict["logit"]
        # log(mu / (1 - mu)), sigma / (mu * (1 - mu))
        return logit(mu), sigma * logit(mu, order=1)

    def exp_trans(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs delta method on data under exponential transform

        Parameters
        ----------
        mu : npt.ArrayLike
            Sample statistics
        sigma : npt.ArrayLike
            Standard errors

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed mean and standard error
        """
        exp = c2fun_dict["exp"]
        # exp(mu), sigma * exp(mu)
        return exp(mu), sigma * exp(mu, order=1)

    def expit_trans(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs delta method on data under expit transform

        Parameters
        ----------
        mu : npt.ArrayLike
            Sample statistics
        sigma : npt.ArrayLike
            Standard errors

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed mean and standard error
        """
        expit = c2fun_dict["expit"]
        # 1 / (1 + exp(-mu)), sigma * exp(x) / (1 + exp(x))^2
        return expit(mu), sigma * expit(mu, order=1)


METHOD_LIST = ["delta"]


def transform_data(
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
    transform: str,
    method: str = "delta",
) -> Tuple[np.ndarray, np.ndarray]:
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
    if method == "delta":
        return delta_method(mu, sigma, transform)


def delta_method(
    mu: npt.ArrayLike, sigma: npt.ArrayLike, transform: str
) -> Tuple[np.ndarray, np.ndarray]:
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
    _check_input("delta", transform, mu, sigma)
    transformer = FirstOrder(transform)
    return transformer(mu, sigma)


def transform_percentage_change_experiment(
    x_vec: npt.ArrayLike, y_vec: npt.ArrayLike
) -> Tuple[float, float]:
    """percentage change with transformed standard error

    Parameters
    ----------
    x_vec : array_like
        observations from first sample
    y_vec : array_like
        observations from second sample

    Returns
    -------
    p_hat : float
        bias corrected percentage change
    sigma_trans : float
        standard error in the transformed space

    Raises
    ------
    ValueError
        covariance is not possible to calculate when x and y are different lengths
    """
    if len(x_vec) != len(y_vec):
        raise ValueError("x_vec must be the same length as y_vec")

    mu_x, mu_y = np.mean(x_vec), np.mean(y_vec)
    cov = np.cov(x_vec, y_vec)
    sigma2_x, sigma2_y, sigma_xy = cov[0, 0], cov[1, 1], cov[0, 1]

    delta_hat = (mu_y - mu_x) / mu_x

    sigma_trans = (
        (sigma2_y / mu_x**2)
        - (2 * mu_y * sigma_xy / (mu_x**3))
        + (mu_y**2 * sigma2_x / (mu_x**4))
    )

    return delta_hat, np.sqrt(sigma_trans)


def handle_input_pct(c_x, n_x, c_y, n_y):
    """helper function to convert to numpy arrays"""
    return np.array([c_x]), np.array([n_x]), np.array([c_y]), np.array([n_y])


def transform_percentage_change(
    c_x: npt.ArrayLike,
    n_x: npt.ArrayLike,
    c_y: npt.ArrayLike,
    n_y: npt.ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """percentage change variance transformation for incidence data

    Parameters
    ----------
    c_x : npt.ArrayLike
        incidence counts in first sample
    n_x : npt.ArrayLike
        sample size(s) of first sample
    c_y : npt.ArrayLike
        incidence counts in second sample
    n_y : npt.ArrayLike
        sample size(s) of second sample

    Returns
    -------
    (delta_hat, sigma_tx)
        sample percentage change of prevalence and corresponding transformed standard error
    """
    c_x, n_x, c_y, n_y = handle_input_pct(c_x, n_x, c_y, n_y)

    mu_x = c_x / n_x
    mu_y = c_y / n_y
    sigma2_x = (c_x * (1 - mu_x) ** 2 + (n_x - c_x) * mu_x**2) / (n_x - 1)
    sigma2_y = (c_y * (1 - mu_y) ** 2 + (n_y - c_y) * mu_y**2) / (n_y - 1)

    # ruff makes this really ugly for some reason
    sigma2_tx = (sigma2_y / (n_y * mu_x**2)) + (
        mu_y**2 * sigma2_x / (n_x * mu_x**4)
    )

    return ((mu_y / mu_x) - 1), np.sqrt(sigma2_tx)


def _check_input(
    method: str, transform: str, mu: npt.ArrayLike, sigma: npt.ArrayLike
) -> None:
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
