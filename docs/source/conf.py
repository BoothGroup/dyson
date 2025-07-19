"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""


# Project information

project = "dyson"
copyright = "2025, Booth Group, King's College London"
author = "Oliver J. Backhouse, Basil Ibrahim, Marcus K. Allen, George H. Booth"


# General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# Options for HTML output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
default_role = "autolink"


# Options for autosummary

autosummary_generate = True


# Options for intersphinx

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pyscf": ("https://pyscf.org/doc/latest/", None),
    "rich": ("https://rich.readthedocs.io/en/stable/", None),
}
