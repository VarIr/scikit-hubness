Installation
============

From PyPI
---------

The current release of `scikit-hubness` can be installed from PyPI:

.. code-block:: bash

   pip install scikit-hubness


From Source
-----------

You can always grab the latest version directly from GitHub:

.. code-block:: bash

    cd install_dir
    git clone git@github.com:VarIr/scikit-hubness.git
    cd scikit-hubness
    pip install -e .

This is the recommended approach, if you want to contribute to the development of `scikit-hubness`.


Supported platforms
-------------------

`scikit-hubness` currently supports all major operating systems:

  - Linux
  - MacOS X
  - Windows

Note, that some functionality of `scikit-hubness` is not available on Windows
(e.g. locality-sensitive hashing (LSH) is provided by `falconn`,
which itself does not support Windows. Please use HNSW instead).
