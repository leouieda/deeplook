DeepLook
========

**Framework for building inverse problems.**

.. image:: http://img.shields.io/travis/leouieda/deeplook/master.svg?style=flat-square
    :alt: Travis CI build status
    :target: https://travis-ci.org/leouieda/deeplook
.. image:: http://img.shields.io/coveralls/leouieda/deeplook/master.svg?style=flat-square
    :alt: Test coverage status
    :target: https://coveralls.io/r/leouieda/deeplook?branch=master

**This is still in very early development and prototyping.**

The sections below are notes on how things will be structured, project goals,
and general ideas.


Project goals
-------------

Easy to reuse. No depending on how things are implemented in Fatiando. Pure
functions that use standard data structures.

Use dask for streaming computations.

Easy to extend and customize. Classes might be too rigid. Inheritance is not
flexible.

Non-linear problems come first. Linear problems are a special case.

Regularization of any kind is treated equally.

Different misfits are treated equally.

Flexibility for building implicit models as well.

Easy to use cross validation.

User implements as little code as possible.


Brainstorming
-------------

Functional is probably going to be better than OOP.

Current code has many side effects and gambiarras to make it look like sklearn.

Not worth copying sklearn API when it clearly wasn't made for non-linear
problems.

I like the keras API but it's also quite complex code and not easy to start
developing.

No side effects will make testing so much easier.

Look into automatic differentiation.


License
-------

This is free software: you can redistribute it and/or modify it
under the terms of the **BSD 3-clause License**. A copy of this license is
provided in ``LICENSE.txt``.
