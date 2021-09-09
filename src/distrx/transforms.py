"""Transform data from one space to another.

Transform data, in the form of sample statistics and their standard
errors, from one space to another using a given transform function.

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
    mu : float or array_like of float
        Vector of sample statistics.
    sigma : float or array_like of float
        Vector of standard errors.
    transform : {'log', 'logit', array_like of function}
        Transform function. Users may define a transform function by
        supplying a function and its derivative. If `method` is
        'delta2', the second derivative is also required.
    method : {'delta, 'delta2'}, optional
        Method used to transform data.

    Returns
    -------
    mu_transform : float or array_like of float
        Vector of sample stastistics in the transform space.
    sigma_transform : float or array_like of float
        Vector of standard errors in the transform space.

    """
    pass


def transform_delta(mu, sigma, transform):
    """Transform data using the delta method.

    Parameters
    ----------
    mu : float or array_like of float
        Vector of sample statistics.
    sigma : float or array_like of float
        Vector of standard errors.
    transform : {'log', 'logit', array_like of function}
        Transform function. Users may define a transform function by
        supplying a function and its derivative.

    Returns
    -------
    mu_transform : float or array_like of float
        Vector of sample statistics in the transform space.
    sigma_transform : float or array_like of float
        Vector of standard errors in the transform space.

    Notes
    -----
    Description of delta method.

    """
    pass


def transform_delta2(mu, sigma, transform):
    """Transform data using the second-order delta method.

    Parameters
    ----------
    mu : float or array_like of float
        Vector of sample statistics.
    sigma : float or array_like of float
        Vector of standard errors.
    transform : {'log', 'logit', array_like of function}
        Transform function. Users may define a transform function by
        supplying a function and its first two derivatives.

    Returns
    -------
    mu_transform : float or array_like of float
        Vector of sample statistics in transform space.
    sigma_transform : float or array_like of float
        Vector of standard errors in transform space.

    Notes
    -----
    Description of second-order delta method.

    """
    pass


def get_transform(transform, order=0):
    """Get transform function and its derivative(s).

    Parameters
    ----------
    transform : {'log', 'logit'}
        Transform function.
    order : {0, 1, 2}, optional
        Highest order of derivative needed.

    Returns
    -------
    transform : function or array_like of function
        Transform function and its derivative(s).

    """
    pass
