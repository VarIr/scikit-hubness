#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUBNESS package available at
https://github.com/OFAI/hubness/
The HUBNESS package is licensed under the terms of the GNU GPLv3.

(c) 2018, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI) and
University of Vienna, Division of Computational Systems Biology (CUBE)
Contact: <roman.feldbauer@ofai.at>


Installation:
-------------
In the console (terminal application) change to the folder containing this file.

To build the package hub_toolbox:
python3 setup.py build

To install the package (with administrator permissions):
sudo python3 setup.py install

To test the installation:
sudo python3 setup.py test

If this succeeds with an 'OK' message, you are ready to go.
Otherwise you may consider filing a bug report on github.
(Some skipped tests are perfectly fine, though.)
"""
import re
import os
import sys
REQ_MAJOR = 3
REQ_MINOR = 6
if sys.version_info < (REQ_MAJOR, REQ_MINOR):
    sys.stdout.write(
        (f"The HUBNESS package requires Python {REQ_MAJOR}.{REQ_MINOR} or higher."
         f"\nPlease try to run as python3 setup.py or update your Python "
         f"environment.\n Consider using Anaconda for easy package handling."))
    sys.exit(1)

try:
    import numpy
    import scipy
    import sklearn
except ImportError:
    sys.stdout.write("The HUBNESS package requires numpy, scipy and scikit-learn. "
                     "Please make sure these packages are available locally. "
                     "Consider using Anaconda for easy package handling.\n")
    sys.exit(1)
try:
    import pandas
    import joblib
except ImportError:
    sys.stdout.write("Some modules of the HUBNESS package require pandas and joblib. "
                     "Please make sure these packages are available locally. "
                     "Consider using Anaconda for easy package handling.\n")
try:
    import nmslib
    import falconn
except ImportError:
    sys.stdout.write("Approximate hubness reduction requires 'nmslib' and 'falconn' "
                     "libraries for approximate nearest neighbor search. "
                     "Please make sure these packages are available locally. "
                     "Consider using Anaconda for easy package handling.\n")
setup_options = {}

try:
    from setuptools import setup
    setup_options['test_suite'] = 'tests'
except ImportError:
    from distutils.core import setup
    import warnings
    warnings.warn("setuptools not found, resorting to distutils. "
                  "Unit tests won't be discovered automatically.")

# Parsing current version number
# Adapted from the Lasagne project at
# https://github.com/Lasagne/Lasagne/blob/master/setup.py
here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    # Read ASCII file with builtin open() so __version__ is str in Python 2 and 3
    with open(os.path.join(here, 'hubness', '__init__.py'), 'r') as f:
        init_py = f.read()
    version = re.search("__version__ = '(.*)'", init_py).groups()[0]
except IOError:
    version = ''

setup(
    name="hubness",
    version=version,
    author="Roman Feldbauer",
    author_email="roman.feldbauer@ofai.at",
    maintainer="Roman Feldbauer",
    maintainer_email="roman.feldbauer@ofai.at",
    description="Hubness reduction and analysis tools",
    license="GNU GPLv3",
    keywords=["machine learning", "data science", "data mining"],
    url="https://github.com/OFAI/hubness",
    packages=['hubness', 'tests'],
    package_data={'examples': ['data/*']},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering"
    ],
    **setup_options
)
