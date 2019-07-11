Installation
============

From PyPI
---------

The current release of `hubness` can be installed from PyPI:

.. code-block:: bash

   pip install hubness


From Source
-----------

You can always grab the latest version directly from GitHub:

.. code-block:: bash

    cd install_dir
    git clone git@github.com:VarIr/hubness.git
    cd hubness
    pip install -e .

This is the recommended approach, if you want to contribute to the development of `hubness`.


Supported platforms
-------------------

`hubness` currently supports all major operating systems:

  - Linux
  - MacOS X
  - Windows

Note, that some functionality of `hubness` is not available on Windows
(e.g. locality-sensitive hashing (LSH) is provided by `falconn`,
which itself does not support Windows. Please use HNSW instead).
