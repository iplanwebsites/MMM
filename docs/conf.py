"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

import tomli

sys.path.insert(0, str(Path("..").resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

with (Path(__file__).parent.parent / "pyproject.toml").open("rb") as f:
    data = tomli.load(f)

project = data["project"]["name"]
copyright = "2024, Metacreation Lab"  # noqa: A001
author = "Metacreation Lab"
version = data["project"]["version"]


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    # "sphinxcontrib.tikz",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"{project}'s docs"
# html_logo = "assets/logo_stroke.png"
# html_favicon = "assets/favicon.png"
# tikz_proc_suite = "GhostScript"  # required for readthedocs, produce png, not svg
