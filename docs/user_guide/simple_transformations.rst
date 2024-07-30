==========================
Univariate Transformations
==========================

There are currently 4 univariate transformations implemented in distrx:
    * log
    * logit
    * exp
    * expit

These transformations are implemented using the first order delta method, which works in these
cases as all of the transformations listed are continuous and differentiable. To briefly summarize,
the delta method transforms the variance by multiplying the original standard error by the first
order Taylor expansion of the transformation function.

Example: Log Transform
----------------------

Suppose that we have some means and standard errors (SEs) of systolic blood pressure (SBP) from
several different samples. The data may look something like the following,

.. csv-table::
   :header: mean, se, n
   :widths: 10, 10, 10
   :align: center

   122, 10, 106
   140, 14, 235
   113, 8, 462
   124, 15, 226
   134, 7, 509

and our goal is to obtain the appropriate SEs for the data after applying the log transform.

The first step is to import the required function from the distrx package.

.. code-block:: python

    from distrx import transform_univariate

Different transformation functions can be chosen through specifying a string parameter of which
transform you would like to apply to your data. In this case, it is the following.

.. code-block:: python

    mu_tx, sigma_tx = transform_univariate(mu=df["means"],
                                           sigma=df["se"],
                                           n=df["n"],
                                           transform="log")

``mu_tx`` and ``sigma_tx`` are simply the means with the transformation function applied and their
corresponding standard errors, respectively. ``sigma_tx`` has already been scaled by :math:`\sqrt{n}`
so the we **should not** scale it by square root of the sample size to obtain a confidence interval.