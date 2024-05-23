==========
Quickstart
==========

Example
-------

.. code-block:: python

    from distrx.transforms import distrx.transform_data

    mu = [1, 2, 3]
    sigma = [0.2, 0.4, 0.6]
    transform = "log"
    method = "delta"
    mu_transformed, sigma_transformed = transform_data(mu, sigma, transform, method)
    assert mu_transformed = [0, 0.69, 1.1]
    assert sigma_transformed = [0.2, 0.2, 0.2]
