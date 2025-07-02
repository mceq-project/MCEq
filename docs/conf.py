# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MCEq"
copyright = "2025, Anatoli Fedynitch, Stefan Fröse"
author = "Anatoli Fedynitch, Stefan Fröse"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "sphinx_gallery.load_style",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


source_suffix = ".rst"
master_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_title = f"{project}"
htmlhelp_basename = project + "docs"

html_theme_options = {
    "navbar_links": [
        {"name": "Quickstart", "url": "quickstart/index", "internal": True},
        {"name": "Example Gallery", "url": "example-gallery", "internal": True},
        {"name": "API Reference", "url": "api-reference/index", "internal": True},
        {"name": "Cite Us", "url": "citeus", "internal": True},
        {"name": "References", "url": "citations", "internal": True},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mceq-project/mceq",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_sidebars = {
    "quickstart/index": [],  # disable sidebar on Quickstart
    "citeus": [],  # disable sidebar on Cite Us
    "citations": [],  # disable sidebar on References
    "example-gallery": [],  # disable sidebar on References
    "v12v11_diff": [],  # disable sidebar on References
}

html_show_sourcelink = False
