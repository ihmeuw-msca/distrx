Simple Transformations
=====

There are currently 4 simple transformations implemented in distrx:
    * log
    * logit
    * exp
    * expit

These transformations are implemented using the first order delta method, which works in these
cases as all of the transformations listed are continuous and differentiable.