#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause

""" scikit-hubness: A Python package for nearest neighbor retrieval in high-dimensional space.

This file is part of the scikit-hubness package available at
https://github.com/VarIr/scikit-hubness/
The scikit-hubness package is licensed under the terms the BSD 3-Clause license.

(c) 2018-2019, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI) and
University of Vienna, Division of Computational Systems Biology (CUBE)
Contact: <roman.feldbauer@univie.ac.at>
"""

import codecs
from os import path
import re
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# Single-sourcing the package version: Read from __init__
def read(*parts):
    with codecs.open(path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="scikit-hubness",  # https://pypi.org/project/scikit-hubness/
    version=find_version("skhubness", "__init__.py"),  # version number should comply with PEP 440
    description="Hubness reduction and analysis tools",  # "summary" metadata field
    long_description=long_description,  # "Description" metadata field; what people will see on PyPI
    long_description_content_type='text/markdown',  # "Description-Content-Type" metadata field
    url="https://github.com/VarIr/scikit-hubness",  # "Home-Page" metadata field
    author="Roman Feldbauer",
    author_email="roman.feldbauer@univie.ac.at",
    maintainer="Roman Feldbauer",
    maintainer_email="roman.feldbauer@univie.ac.at",
    classifiers=[  # https://pypi.org/classifiers/
        'Development Status :: 4 - Beta',
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords="machine-learning high-dimensional-data hubness nearest-neighbor "
             "data-science data-mining artificial-intelligence ",  # string of words separated by whitespace, not a list
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # previously used : packages=['skhubness', 'tests'],
    python_requires='>=3.7',  # 'pip install' will check this
    install_requires=['numpy',    # These packages will be installed by pip.
                      'scipy >= 1.2',    # For comparison with requirements.txt see also:
                      'scikit-learn >= 0.21',  # https://packaging.python.org/en/latest/requirements.html
                      'tqdm',
                      'pybind11',  # Required for nmslib build
                      'joblib >= 0.12',
                      'nmslib',
                      'falconn;platform_system!="Windows"',  # falconn is not available on Windows; see also PEP 508
                      ],
    extras_require={  # Install using the 'extras' syntax: $ pip install sampleproject[dev]
        # 'dev': ['check-manifest'],
        'test': ['coverage', 'pytest', 'nose'],
    },
    package_data={'examples': ['data/*',
                               'skhubness/data/dexter/*'], },
    include_package_data=True,  # to include data in wheel
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/VarIr/scikit-hubness/issues',
        'Documentation': 'https://scikit-hubness.readthedocs.io',
        'Say Thanks!': 'https://saythanks.io/to/VarIr',
        'Source': 'https://github.com/VarIr/scikit-hubness/',
    },
)
