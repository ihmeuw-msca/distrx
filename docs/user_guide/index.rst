User guide
==========

.. toctree::
   :hidden:
   :numbered:

   simple_transformations
   percentage_change

.. note::

   Follow the titles in the sidebar for more information on the general type of transformation you
   are performing on your data to use this user guide


This user guide introduces and explains some key concepts behind transforming data. A common pitfall
modelers fall into is to simply apply the transformation they are using for their data (e.g. log) to
their standard errors as well. However, due to the nature of standard error calculation, this
will provide completely incorrect standard errors. In order to compute the standard error of data
after functions have been applied, or when functions of multiple variables are involved, more
complex transformation methods such as those based on the delta method are required.