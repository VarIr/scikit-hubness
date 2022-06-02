..
    DUPLICATED in root (for Github) and root/docs/development (for RTD)
    Until a clean way of redirecting/mirroring/etc is found,
    ALL CHANGES must be applied to BOTH VERSIONS!

============
Contributing
============

`scikit-hubness` is free open source software.
Contributions from the community are highly appreciated.
Even small contributions improve the software's quality.

If you are not familiar with programming languages and tools,
you may contribute by filing bugs or any problems as a
`GitHub issue <https://github.com/VarIr/scikit-hubness/issues>`_.


Git and branching model
=======================

We use `git` for version control (CVS), as do most projects nowadays.
If you are not familiar with git, there are lots of tutorials on
`GitHub Guide <https://guides.github.com/>`_.
All the important basics are covered in the
`GitHub Git handbook <https://guides.github.com/introduction/git-handbook/>`_.

There is one main branches: ``main``.
For any changes, a new branch should be created.
If you want to add a new feature, fix a noncritical bug, fix a critical bug,
branch off ``main``, introduce your changes, and create a pull request.

(Previously, development of `scikit-hubness` (mostly) followed the
`git flow branching model <https://nvie.com/posts/a-successful-git-branching-model/>`_.
This was found to be unnecessarily complicated for a project of this size).


Workflow
========

In case of large changes to the software, please first get in contact
with the authors for coordination, for example by filing an
`issue <https://github.com/VarIr/scikit-hubness/issues>`_.
If you want to fix small issues (typos in the docs, obvious errors, etc.)
you can - of course - directly submit a pull request (PR).

#. Create a fork of `scikit-hubness` in your GitHub account.
    Simply click "Fork" button on `<https://github.com/VarIr/scikit-hubness>`_.


#. Clone your fork on your computer.
    $ ``git clone git@github.com:YOUR-ACCOUNT-GOES-HERE/scikit-hubness.git && cd scikit-hubness``

#. Add remote upstream.
    $ ``git remote add upstream git@github.com:VarIr/scikit-hubness.git``

#. Create feature/bugfix branch.
    In case of feature or noncritical bugfix:
    $ ``git checkout main && git checkout -b featureXYZ``

    In case of critical bug:
    $ ``git checkout -b bugfix123 main``

#. Implement feature/fix bug/fix typo/...
    Happy coding!

#. Create a commit with meaningful message
    If you only modified existing files, simply
    ``$ git commit -am "descriptive message what this commit does (in present tense) here"``

#. Push to GitHub
    e.g. $ ``git push origin featureXYZ``

#. Create pull request (PR)
    Git will likely provide a link to directly create the PR.
    If not, click "New pull request" on your fork on GitHub.

#. Wait...
    Several devops checks will be performed automatically
    (e.g. continuous integration (CI) with Github Actions).

    The authors will get in contact with you,
    and may ask for changes.

#. Respond to code review.
    If there were issues with continuous integration,
    or the authors asked for changes, please create a new commit locally,
    and simply push again to GitHub as you did before.
    The PR will be updated automatically.

#. Maintainers merge PR, when all issues are resolved.
    Thanks a lot for your contribution!


Code style and further guidelines
=================================

* Please make sure all code complies with `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_

* All code should be documented sufficiently
  (functions, classes, etc. must have docstrings with general description, parameters,
  ideally return values, raised exceptions, notes, etc.)

* Documentation style is
  `NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.

* New code must be covered by unit tests using `pytest <https://docs.pytest.org/en/latest/>`_.

* If you fix a bug, please provide regression tests (fail on old code, succeed on new code).

* It may be helpful to install `scikit-hubness` in editable mode for development.
  When you have already cloned the package, switch into the corresponding directory, and

  .. code-block:: bash

      pip install -e .

  (don't omit the trailing period).
  This way, any changes to the code are reflected immediately.
  That is, you don't need to install the package each and every time,
  when you make changes while developing code.


Testing
=======

In `scikit-hubness`, we aim for high code coverage. Between 90% and 100% of all code lines
should be visited at least once when running the complete test suite.
This is primarily to ensure:

* correctness of the code (to some extent) and
* maintainability (new changes don't break old code).

Creating a new PR, ideally all code would be covered by tests.
Sometimes, this is not feasible or only with large effort.
Pull requests will likely be accepted, if the overall code coverage
at least does not decrease.

Unit tests are automatically performed for each PR using CI tools online.
This may take some time, however.
To run the tests locally, you need `pytest` installed.
From the scikit-hubness directory, call

.. code-block:: bash

    pytest skhubness/

to run all the tests. You can also restrict the tests to the subpackage
you are working on, down to single tests.
For example

.. code-block:: bash

    pytest skhubness/reduction/tests/test_local_scaling.py --showlocals -v

only runs tests for hubness reduction with local scaling.

In order to check code coverage locally, you need the
`pytest-cov plugin <https://github.com/pytest-dev/pytest-cov>`_.

.. code-block:: bash

    pytest skhubness/reduction/ --cov=skhubness/reduction/
