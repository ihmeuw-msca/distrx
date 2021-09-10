"""Transform data from one space to another.

Transform data, in the form of sample statistics and their standard
errors, from one space to another using a given transform function.

TODO:
* Implement transform_delta2
* Implement transform_data
* Add typing in function definition?
* Add user-defined transform function

"""
import numpy as np


def transform_data(mu, sigma, transform, method='delta'):
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
    mu_transform : numpy.ndarray
        Sample stastistics in the transform space.
    sigma_transform : numpy.ndarray
        Standard errors in the transform space.

    """
    return


def transform_delta(mu, sigma, transform):
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
    mu_transform : numpy.ndarray
        Sample statistics in the transform space.
    sigma_transform : numpy.ndarray
        Standard errors in the transform space.

    Notes
    -----
    The delta method expands a function of a random variable about its
    mean with a one-step Taylor approximation and then takes the
    variance.

    """
    # Check mu and sigma
    mu = np.array(mu)
    sigma = np.array(sigma)
    if len(mu) != len(sigma):
        raise ValueError("Lengths of mu and sigma don't match.")
    if np.any(sigma <= 0.0):
        raise ValueError("Sigma values must be positive.")

    # Approximate transformed data
    transform = get_transform(transform, 1)
    return transform[0](mu), sigma*transform[1](mu)**2


def transform_delta2(mu, sigma, transform):
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
    mu_transform : numpy.ndarray
        Sample statistics in transform space.
    sigma_transform : numpy.ndarray
        Standard errors in transform space.

    Notes
    -----
    The second-order delta method expands a function of a random
    variable about its mean with a two-step Taylor approximation and
    then takes the variance. This method is useful if the derivative of
    the transform function is zero (so the first-order delta method
    cannot be applied), or the sample size is small.

    """
    return


def get_transform(transform, order=0):
    """Get transform function and its derivative(s).

    Returns transform function if `order` is 0.
    Otherwise returns an array of functions, including the transform
    function and its derivatives up to the specified order.

    Parameters
    ----------
    transform : {'log', 'logit', 'exp', 'expit'}
        Transform function.
    order : {0, 1, 2}, optional
        Highest order of derivative needed.

    Returns
    -------
    transforms : function or array_like of function
        Transform function and its derivative(s).

    """
    # Check input
    if transform not in ['log', 'logit', 'exp', 'expit']:
        raise ValueError(f"Invalid transform function '{transform}'.")
    if order not in [0, 1, 2]:
        raise ValueError(f"Invalid order '{order}'.")

    # Define transform functions
    transform_dict = {
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

    # Get function or list of functions
    if order == 0:
        return transform_dict[transform][order]
    return transform_dict[transform][:order+1]
