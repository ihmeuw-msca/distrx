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
        self, mu: npt.ArrayLike, sigma: npt.ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        c2_transformation = c2fun_dict[self.transform]
        return c2_transformation(mu), sigma * c2_transformation(mu, order=1)


class FirstOrderBivariate:
    def __init__(self, operation: str) -> None:
        """Initializes an object to perform 1st order delta method transformations

        Parameters
        ----------
        operation : str
            operation of choice

        Raises
        ------
        ValueError
            Is thrown function of choice not implemented
        """
        self.operation = operation

    def __call__(
        self,
        mu_x: npt.ArrayLike,
        sigma_x: npt.ArrayLike,
        mu_y: npt.ArrayLike,
        sigma_y: npt.ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        match self.operation:
            case "sum":
                return self.sum(mu_x, sigma_x, mu_y, sigma_y)
            case "difference":
                return self.diff(mu_x, sigma_x, mu_y, sigma_y)
            case "product":
                return self.prod(mu_x, sigma_x, mu_y, sigma_y)
            case "quotient":
                return self.quotient(mu_x, sigma_x, mu_y, sigma_y)
            case _:
                raise ValueError(f"Invalid transform '{self.transform}'.")

    def sum(self, mu_x, sigma_x, mu_y, sigma_y):
        sigma2_tx = sigma_x**2 + sigma_y**2
        return mu_x + mu_y, np.sqrt(sigma2_tx)

    def diff(self, mu_x, sigma_x, mu_y, sigma_y):
        sigma2_tx = sigma_x**2 + sigma_y**2
        return mu_x - mu_y, np.sqrt(sigma2_tx)

    def prod(self, mu_x, sigma_x, mu_y, sigma_y):
        sigma2_tx = mu_y**2 * sigma_x**2 + mu_x**2 * sigma_y**2
        return mu_x * mu_y, np.sqrt(sigma2_tx)

    def quotient(self, mu_x, sigma_x, mu_y, sigma_y):
        sigma2_tx = (sigma_y**2 / mu_x**2) + (mu_y**2 * sigma_x**2 / mu_x**4)
        return mu_y / mu_x, np.sqrt(sigma2_tx)


def transform_univariate(
    mu: npt.ArrayLike,
    sigma: npt.ArrayLike,
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

    mu, sigma = np.atleast_1d(np.array(mu)), np.atleast_1d(np.array(sigma))
    _check_input(mu, sigma)
    match method:
        case "delta":
            transformer = FirstOrder(transform)
            return transformer(mu, sigma)
        case _:
            raise ValueError(f"Invalid method '{method}'.")


def transform_bivariate(
    mu_x: npt.ArrayLike,
    sigma_x: npt.ArrayLike,
    mu_y: npt.ArrayLike,
    sigma_y: npt.ArrayLike,
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

    mu_x, sigma_x, mu_y, sigma_y = (
        np.array(mu_x),
        np.array(sigma_x),
        np.array(mu_y),
        np.array(sigma_y),
    )
    match method:
        case "delta":
            match transform:
                case "percentage_change":
                    pest, sigma = FirstOrderBivariate("quotient")(
                        mu_x, sigma_x, mu_y, sigma_y
                    )
                    return pest - 1, sigma
                case "ratio":
                    pest, sigma = FirstOrderBivariate("quotient")(
                        mu_x, sigma_x, mu_y, sigma_y
                    )
                    return pest, sigma
        case _:
            raise ValueError(f"Invalid method '{method}'.")


### HELPER FUNCTIONS
def process_counts(count, n):
    mu = count / n
    sigma2 = (count * (1 - mu) ** 2 + (n - count) * mu**2) / (n - 1)

    return mu, np.sqrt(sigma2 / n)


def _check_input(mu: npt.ArrayLike, sigma: npt.ArrayLike) -> None:
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
    _check_lengths_match(mu, sigma)
    _check_sigma_positive(sigma)


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
