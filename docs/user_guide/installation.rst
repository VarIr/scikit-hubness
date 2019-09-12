Installation
============

From PyPI
---------

The current release of `scikit-hubness` can be installed from PyPI:

.. code-block:: bash

   pip install scikit-hubness


Dependencies
------------

All strict dependencies of `scikit-hubness` are automatically installed
by `pip`. Some optional dependencies (certain ANN libraries) may not
yet be available from PyPI. If you require one of these libraries,
please refer to the library's documentation for building instructions.
For example, at the time of writing, `puffinn` was not available on PyPI.
Building and installing is straight-forward:

.. code-block:: bash

    git clone https://github.com/puffinn/puffinn.git
    cd puffinn
    python3 setup.py build
    pip install .


From Source
-----------

You can always grab the latest version of `scikit-hubness` directly from GitHub:

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

Note, that not all approximate nearest neighbor libraries used in `scikit-hubness`
are available on all platforms. The table below indicates, which libaries and
algorithms are currently supported on your operating system.

+---------+-------------+-------+-------+---------+
| library | algorithm   | Linux | MacOS | Windows |
+---------+-------------+-------+-------+---------+
| nmslib  | hnsw        |   x   |   x   |    x    |
+---------+-------------+-------+-------+---------+
| annoy   | rptree      |   x   |   x   |    x    |
+---------+-------------+-------+-------+---------+
| ngtpy   | onng        |   x   |   x   |         |
+---------+-------------+-------+-------+---------+
| falconn | falconn_lsh |   x   |   x   |         |
+---------+-------------+-------+-------+---------+
| puffinn | lsh         |   x   |       |         |
+---------+-------------+-------+-------+---------+
