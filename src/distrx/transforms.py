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

METHOD_LIST = ["delta"]


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
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike, n: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        match self.transform:
            case "log":
                return self.log_trans(mu, sigma, n)
            case "logit":
                return self.logit_trans(mu, sigma, n)
            case "exp":
                return self.exp_trans(mu, sigma, n)
            case "expit":
                return self.expit_trans(mu, sigma, n)
            case _:
                raise ValueError(f"Invalid transform '{self.transform}'.")

    def log_trans(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike, n: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs delta method on data under log transform

        .. math::

            \log(mu), \frac{\sigma}{\mu} \cdot \frac{1}{\sqrt{n}}

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
        return log(mu), sigma * log(mu, order=1) / np.sqrt(n)

    def logit_trans(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike, n: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs delta method on data under logit transform

        .. math::

            \log(\frac{\mu}{1 - \mu}), \frac{\sigma}{\mu \cdot (1 - \mu)} \cdot \frac{1}{\sqrt{n}}

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
        return logit(mu), sigma * logit(mu, order=1) / np.sqrt(n)

    def exp_trans(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike, n: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs delta method on data under exponential transform

        .. math::

            \exp(\mu), \sigma \cdot \exp(\mu) \cdot \frac{1}{\sqrt{n}}

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
        return exp(mu), sigma * exp(mu, order=1) / np.sqrt(n)

    def expit_trans(
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike, n: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs delta method on data under expit transform

        .. math::

            \frac{1}{1 + \exp(-\mu)}, \sigma \cdot \frac{\exp(\mu)}{(1 + \exp(\mu))^2} \cdot \frac{1}{\sqrt{n}}

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
        return expit(mu), sigma * expit(mu, order=1) / np.sqrt(n)


class FirstOrderBivariate:
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
        self,
        c_x: npt.ArrayLike,
        n_x: npt.ArrayLike,
        c_y: npt.ArrayLike,
        n_y: npt.ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        match self.transform:
            case "percentage_change":
                return self.percentage_change_trans(c_x, n_x, c_y, n_y)
            case _:
                raise ValueError(f"Invalid transform '{self.transform}'.")

    def percentage_change_trans(
        self,
        c_x: npt.ArrayLike,
        n_x: npt.ArrayLike,
        c_y: npt.ArrayLike,
        n_y: npt.ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """percentage change variance transformation for incidence data

        .. math::

            \frac{p_y}{p_x} - 1, \sqrt{\frac{\sigma_y^2}{n_y\mu_x^2} + \frac{\mu_y^2\sigma_x^2}{n_x\mu_x^4}}

        Parameters
        ----------
        c_x : npt.ArrayLike
            incidence counts in first sample
        n_x : npt.ArrayLike
            sample sizes of first sample
        c_y : npt.ArrayLike
            incidence counts in second sample
        n_y : npt.ArrayLike
            sample sizes of second sample

        Returns
        -------
        (delta_hat, sigma_tx)
            sample percentage change of prevalence and corresponding transformed standard error
        """

        mu_x = c_x / n_x
        mu_y = c_y / n_y
        sigma2_x = (c_x * (1 - mu_x) ** 2 + (n_x - c_x) * mu_x**2) / (n_x - 1)
        sigma2_y = (c_y * (1 - mu_y) ** 2 + (n_y - c_y) * mu_y**2) / (n_y - 1)

        sigma2_tx = (sigma2_y / (n_y * mu_x**2)) + (
            mu_y**2 * sigma2_x / (n_x * mu_x**4)
        )

        return ((mu_y / mu_x) - 1), np.sqrt(sigma2_tx)


def transform_univariate(
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
    n: npt.ArrayLike,
    transform: str,
    method: str = "delta",
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform univariate data from one space to another.

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
    method : {'delta'}
        Method used to transform data.

    Returns
    -------
    mu_trans : numpy.ndarray
        Sample stastistics in the transform space.
    sigma_trans : numpy.ndarray
        Standard errors in the transform space.

    """

    mu, sigma = np.array(mu), np.array(sigma)
    _check_input(mu, sigma, n)
    match method:
        case "delta":
            transformer = FirstOrder(transform)
            return transformer(mu, sigma, n)
        case _:
            raise ValueError(f"Invalid method '{method}'.")


def transform_bivariate(
    c_x: npt.ArrayLike,
    n_x: npt.ArrayLike,
    c_y: npt.ArrayLike,
    n_y: npt.ArrayLike,
    transform: str,
    method: str = "delta",
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform bivariate data to a univariate space

    Transform data, in the form of counts and sample size from 2 groups,
    to point estimates and appropriate standard errors using a given transform
    function. No assumptions are made about the underlying distributions of
    the data.

    Parameters
    ----------
    c_x : npt.ArrayLike
        _description_
    n_x : npt.ArrayLike
        _description_
    c_y : npt.ArrayLike
        _description_
    n_y : npt.ArrayLike
        _description_
    transform : str
        _description_
    method : str, optional
        _description_, by default "delta"

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    c_x, n_x, c_y, n_y = (
        np.array(c_x),
        np.array(n_x),
        np.array(c_y),
        np.array(n_y),
    )
    match method:
        case "delta":
            transformer = FirstOrderBivariate(transform)
            return transformer(c_x, n_x, c_y, n_y)
        case _:
            raise ValueError(f"Invalid method '{method}'.")


# def transform_percentage_change_experiment(
#     x_vec: npt.ArrayLike, y_vec: npt.ArrayLike
# ) -> Tuple[float, float]:
#     """percentage change with transformed standard error

#     Parameters
#     ----------
#     x_vec : array_like
#         observations from first sample
#     y_vec : array_like
#         observations from second sample

#     Returns
#     -------
#     p_hat : float
#         bias corrected percentage change
#     sigma_trans : float
#         standard error in the transformed space

#     Raises
#     ------
#     ValueError
#         covariance is not possible to calculate when x and y are different lengths
#     """
#     if len(x_vec) != len(y_vec):
#         raise ValueError("x_vec must be the same length as y_vec")

#     mu_x, mu_y = np.mean(x_vec), np.mean(y_vec)
#     cov = np.cov(x_vec, y_vec)
#     sigma2_x, sigma2_y, sigma_xy = cov[0, 0], cov[1, 1], cov[0, 1]

#     delta_hat = (mu_y - mu_x) / mu_x

#     sigma_trans = (
#         (sigma2_y / mu_x**2)
#         - (2 * mu_y * sigma_xy / (mu_x**3))
#         + (mu_y**2 * sigma2_x / (mu_x**4))
#     )

#     return delta_hat, np.sqrt(sigma_trans)


def _check_input(
    mu: npt.ArrayLike, sigma: npt.ArrayLike, n: npt.ArrayLike
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
    # _check_lengths_match(mu, sigma)
    _check_sigma_n_positive(sigma, n)


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


def _check_sigma_n_positive(sigma: npt.ArrayLike, n: npt.ArrayLike) -> None:
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
    if np.any(n == 0.0):
        warnings.warn("Sigma vector contains zeros.")
    if np.any(n < 0.0):
        raise ValueError("Sigma values must be positive.")
