=========================
Bivariate Transformations
=========================

There are currently 2 bivariate transformations implemented in distrx:
    * percentage change
    * ratio

These transformations are implemented using the first order delta method. See INSERT CONCEPTS for
derivation if desired. Note that all functions are in terms of sample statistics (e.g. mean), not
raw counts, even though some functions do take counts as input.

Example: Percentage Change
--------------------------

Suppose we have samples in 2 different years measuring the incidence of cancer cases in each year
in various state counties. The data may look something like the following,

.. csv-table::
   :header: county, cases_1, sample_1, cases_2, sample_2
   :widths: 10, 10, 10, 10, 10
   :align: center

   "King", 252, 400, 258, 250
   "Snohomish", 12, 300, 90, 500
   "Pierce", 505, 1000, 219, 1000
   "Kitsap", 88, 124, 67, 204

and our goal is to find the percentage change in the prevalence of cancer with its appropriate SE.

Since we have counts and distrx expects mean/standard error (SE), we must first convert the data
appropriately. Counts data is common at IHME, so a function is provided to return sample mean and
SE given incidence count and sample size. We can import it and save the necessary variables like so.

.. code-block:: python

    from distrx import process_counts
    mu_x, sigma_x = process_counts(cases_1, sample_1)
    mu_y, sigma_y = process_counts(cases_2, sample_2)


Then, we can import the required function from the distrx package.

.. code-block:: python

    from distrx import transform_bivariate

Different transformation functions can be chosen through specifying a string parameter of which
transform you would like to apply to your data. In this case, it is the following.

.. code-block:: python

    mu_tx, sigma_tx = transform_bivariate(mu_x=mu_x,
                                          sigma_x=sigma_x,
                                          mu_y=mu_y,
                                          sigma_y=sigma_y,
                                          transform="percentage_change")

``mu_tx`` and ``sigma_tx`` are simply the percentage change for each county and their corresponding
standard errors, respectively. If a CI for the mean is desired, simply use
``mu_tx +/- Q * sigma_tx``.
