#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Python hubness package for nearest neighbor retrieval in high-dimensional space.

This file is part of the HUBNESS package available at
https://github.com/OFAI/hubness/
The HUBNESS package is licensed under the terms of the GNU GPLv3.

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
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
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
    name="hubness",  # https://pypi.org/project/hubness/
    version=find_version("hubness", "__init__.py"),  # version number should comply with PEP 440
    description="Hubness reduction and analysis tools",  # "summary" metadata field
    long_description=long_description,  # "Description" metadata field; what people will see on PyPI
    long_description_content_type='text/x-rst',  # "Description-Content-Type" metadata field
    url="https://github.com/OFAI/hubness",  # "Home-Page" metadata field
    author="Roman Feldbauer",
    author_email="roman.feldbauer@univie.ac.at",
    maintainer="Roman Feldbauer",
    maintainer_email="roman.feldbauer@univie.ac.at",
    classifiers=[  # https://pypi.org/classifiers/
        'Development Status :: 4 - Beta',
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords="machine-learning high-dimensional-data hubness nearest-neighbor "
             "data-science data-mining artificial-intelligence ",  # string of words separated by whitespace, not a list
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # previously used : packages=['hubness', 'tests'],
    python_requires='>=3.6',  # 'pip install' will check this
    install_requires=['numpy',    # These packages will be installed by pip.
                      'scipy',    # For comparison with requirements.txt see also:
                      'sklearn',  # https://packaging.python.org/en/latest/requirements.html
                      'pandas',
                      'tqdm',
                      'joblib',
                      'nmslib',
                      'falconn',
                      ],
    extras_require={  # Install using the 'extras' syntax: $ pip install sampleproject[dev]
        # 'dev': ['check-manifest'],
        'test': ['coverage', 'pytest', 'nose'],
    },
    package_data={'examples': ['data/*'], },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/OFAI/hubness/issues',
        'Documentation': 'https://hubness.readthedocs.io',
        'Say Thanks!': 'https://saythanks.io/to/VarIr',
        'Source': 'https://github.com/OFAI/hubness/',
    },
)
