# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

import mock
MOCK_MODULES = ['falconn', 'nmslib',
                ]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- Project information -----------------------------------------------------

project = 'scikit-hubness'
copyright = '2019, Roman Feldbauer'
author = 'Roman Feldbauer'

# The full version, including alpha/beta/rc tags
from skhubness import __version__
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['recommonmark',
              'numpydoc',
              'sphinx_automodapi.automodapi',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.graphviz',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.todo',
              # 'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.githubpages',
              'sphinx.ext.mathjax',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.linkcode',
              ]

# Due to sphinx-automodapi
numpydoc_show_class_members = False

# Napoleon settings
napoleon_include_init_with_doc = False

# autodoc options
autodoc_default_options = {'members': True, 'inherited-members': True}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Mock packages that are not installed on rtd
autodoc_mock_imports = MOCK_MODULES

# The master toctree document. (see https://stackoverflow.com/a/56859983/6555620)
master_doc = 'index'

autosummary_generate = True


# The following is used by sphinx.ext.linkcode to provide links to github
from docs.github_link import make_linkcode_resolve
linkcode_resolve = make_linkcode_resolve('sklearn',
                                         'https://github.com/scikit-learn/'
                                         'scikit-learn/blob/{revision}/'
                                         '{package}/{path}#L{lineno}')

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# default: html_theme = 'alabaster'
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    html_theme = 'default'
else:
    html_theme = 'sphinx_pdj_theme'
import sphinx_pdj_theme
html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
