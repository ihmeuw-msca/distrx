distrx documentation
===================

.. toctree::
   :hidden:

   getting_started/index
   user_guide/index
   api_reference/index
   developer_guide/index

.. note::

   In this page, please use one or two paragraphs to summarize the main purpose of the package.
   Following the summary, please provide guidence of how people should use this documentation.

Statistics like variance and SE (standard error) are a measure of spread in data.
When shifting all of your data up or down (i.e. adding and subtracting a constant from your data),
you need not worry about tinkering with the variance or SE calculations.
However, when applying a nonlinear transformation to data (e.g. a log transform),
one’s intuition may be to simply apply the same function to the resulting statistics.
While this might work for things like the sample mean,
but doing this for SE will produce completely incorrect uncertainty estimates.
The distrx package aims to provide functions based in sound statistical theory
to allow users to apply common transformation functions in the health sciences field
to produce transformations and maintain correct uncertainty estimates.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - :ref:`Getting started`
     - :ref:`User guide`

   * - If you are new to distrx, this is where you should go. It contains main use cases and examples to help you get started.
     - The user guide provides in-depth information on key concepts of data transformation with useful background information and explanation.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - :ref:`API reference`
     - :ref:`Developer guide`

   * - If you are looking for information on a specific module, function, class or method, this part of the documentation is for you.
     - Want to improve the existing functionalities and documentation? The contributing guidelines will guide you through the process.
