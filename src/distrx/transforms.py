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


class FirstOrder:
    """
    Contains 4 most common 1st order delta method transformation functions
    """

    def __init__(
        self, transform: str, mu: npt.ArrayLike, sigma: npt.ArrayLike
    ) -> None:
        """Initializes an object to perform 1st order delta method transformations

        Parameters
        ----------
        transform : str
            Function of choice
        mu : npt.ArrayLike
            Summary statistics
        sigma : npt.ArrayLike
            Standard errors

        Raises
        ------
        ValueError
            Is thrown function of choice not implemented
        """
        self.transform = transform
        self.mu = mu
        self.sigma = sigma
        match transform:
            case "log":
                self.mu_trans, self.sigma_trans = self.log_trans(
                    self.mu, self.sigma
                )
            case "logit":
                self.mu_trans, self.sigma_trans = self.logit_trans(
                    self.mu, self.sigma
                )
            case "exp":
                self.mu_trans, self.sigma_trans = self.exp_trans(
                    self.mu, self.sigma
                )
            case "expit":
                self.mu_trans, self.sigma_trans = self.expit_trans(
                    self.mu, self.sigma
                )
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
        return np.log(mu), sigma / mu

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
        return np.log(mu / (1.0 - mu)), sigma / (mu * (1.0 - mu))

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
        return np.exp(mu), sigma * np.exp(mu)

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
        return 1.0 / (1.0 + np.exp(-mu)), sigma * np.exp(-mu) / (
            1.0 + np.exp(-mu)
        ) ** 2

    def get_mu_trans(self) -> np.ndarray:
        """Getter for transformed mean

        Returns
        -------
        np.ndarray
            Transformed mean
        """
        return self.mu_trans

    def get_sigma_trans(self) -> np.ndarray:
        """Getter for transformed standard error

        Returns
        -------
        np.ndarray
            Transformed standard error
        """
        return self.sigma_trans


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
    transformer = FirstOrder(transform, mu, sigma)
    return transformer.get_mu_trans(), transformer.get_sigma_trans()


def transform_percentage_change(
    x_vec: npt.ArrayLike, y_vec: npt.ArrayLike
) -> Tuple[float, float]:
    """bias corrected percentage change with transformed standard error

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
    mu_x = np.mean(x_vec)
    mu_y = np.mean(y_vec)
    n = len(x_vec)
    cov = np.cov(x_vec, y_vec)
    sigma2_x = cov[0, 0]
    sigma2_y = cov[1, 1]
    sigma_xy = cov[0, 1]

    delta_hat = (mu_y - mu_x) / mu_x
    # TODO: add option instead of doing this by default
    bias_corr = (mu_y * sigma2_x) / ((n * mu_x) ** 2)
    p_hat = delta_hat + bias_corr

    sigma_trans = (
        (sigma2_y / mu_x**2)
        - (2 * mu_y * sigma_xy / (mu_x**3))
        + (mu_y**2 * sigma2_x / (mu_x**4))
    )

    return p_hat, np.sqrt(sigma_trans)


def transform_percentage_change_counts1(
    c_x: int, n_x: int, c_y: int, n_y: int
) -> float:
    """alternative percentage change transformation with only counts provided

    Parameters
    ----------
    c_x : int
        raw count in one sample (e.g. of incidence)
    n_x : int
        sample size
    c_y : int
        raw count in second sample (e.g. of incidence)
    n_y : int
        sample size

    Returns
    -------
    sigma_trans: array_like
        standard errors in the transform space
    """
    mu_x = c_x / n_x
    mu_y = c_y / n_y
    # sigma2_x = (c_x * (1 - mu_x) ** 2 + (n_x - c_x) * mu_x**2) / (n_x - 1)
    # sigma2_y = (c_y * (1 - mu_y) ** 2 + (n_y - c_y) * mu_y**2) / (n_y - 1)
    sigma2_x = n_x * mu_x * (1 - mu_x)
    sigma2_y = n_y * mu_y * (1 - mu_y)

    # sigma_trans = (sigma2_y / mu_x**2) + (mu_y**2 * sigma2_x / (mu_x**4))
    sigma_trans = (sigma2_y / c_x**2) + (c_y**2 * sigma2_x / (c_x**4))
    print(sigma2_x, sigma2_y)

    return ((c_y / c_x) - 1), np.sqrt(sigma_trans)


def transform_percentage_change_counts2(
    c_x: int, n_x: int, c_y: int, n_y: int
) -> float:
    """alternative percentage change transformation with only counts provided

    Parameters
    ----------
    c_x : int
        raw count in one sample (e.g. of incidence)
    n_x : int
        sample size
    c_y : int
        raw count in second sample (e.g. of incidence)
    n_y : int
        sample size

    Returns
    -------
    sigma_trans: array_like
        standard errors in the transform space
    """
    mu_x = c_x / n_x
    mu_y = c_y / n_y
    sigma2_x = (c_x * (1 - mu_x) ** 2 + (n_x - c_x) * mu_x**2) / (n_x - 1)
    sigma2_y = (c_y * (1 - mu_y) ** 2 + (n_y - c_y) * mu_y**2) / (n_y - 1)
    # print("look", sigma2_x, sigma2_y)

    sigma_trans = (sigma2_y / mu_x**2) + (mu_y**2 * sigma2_x / (mu_x**4))
    # sigma_trans = (sigma2_y / c_x**2) + (c_y**2 * sigma2_x / (c_x**4))

    return ((mu_y / mu_x) - 1), np.sqrt(sigma_trans)


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
