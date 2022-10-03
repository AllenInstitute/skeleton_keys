# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'skeleton_keys'
copyright = '2022, Nathan Gouwens, Forrest Collman, Matt Mallory, Clare Gamlin'
author = 'Nathan Gouwens, Forrest Collman, Matt Mallory, Clare Gamlin'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
#     'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',

]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["fenics", 'mshr']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']


from argschema.autodoc import process_schemas

def setup(app):
    app.connect('autodoc-process-docstring',process_schemas)
