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

Suppose that we have some means and standard deviations (SDs) of systolic blood pressure (SBP) from
several different samples. The data may look something like the following,

.. csv-table::
   :header: mean, SD, n
   :widths: 10, 10, 10
   :align: center

   122, 10, 106
   140, 14, 235
   113, 8, 462
   124, 15, 226
   134, 7, 509

and our goal is to obtain the appropriate standard errors (SEs) for the mean after applying the log
transform.

Since we are interested in the transformed SEs and *not* the transformed SDs, we must provide the
SEs to distrx. **If you already have SEs and are performing the same task, you should skip this
step!**

.. code-block:: python

    df["SE"] = df["SD"] / df["n"]

Now, import the appropriate function from distrx.

.. code-block:: python

    from distrx import transform_univariate

Different transformation functions can be chosen through specifying a string parameter of which
transform you would like to apply to your data. In this case, it is the following.

.. code-block:: python

    mu_tx, sigma_tx = transform_univariate(mu=df["means"],
                                           sigma=df["SE"],
                                           transform="log")

``mu_tx`` and ``sigma_tx`` are simply the means with the transformation function applied and their
appropriately transformed standard errors, respectively. If a CI for the mean is desired, simply
use ``mu_tx +/- Q * sigma_tx``.