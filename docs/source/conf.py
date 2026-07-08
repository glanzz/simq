# Configuration file for the Sphinx documentation builder.
#
# This builds the project-wide Ferriq documentation site (Rust workspace +
# Python bindings). It is deployed to GitHub Pages by
# .github/workflows/docs.yml.
#
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = "Ferriq"
copyright = "2025, Ferriq Contributors"
author = "Ferriq Contributors"
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",            # Markdown support
    "sphinx_design",          # Grids, cards, tabs for the landing page
    "sphinx_copybutton",      # "Copy" button on code blocks
    "sphinx_sitemap",         # sitemap.xml for search engines
    "sphinxext.opengraph",    # OpenGraph/social + SEO meta tags
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST (Markdown) configuration
myst_enable_extensions = [
    "colon_fence",     # ::: directives
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_inline",
    "substitution",
]
myst_heading_anchors = 3

# Syntax highlighting default for untagged code blocks
highlight_language = "rust"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "Ferriq — High-Performance Quantum Computing SDK"
html_short_title = "Ferriq"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"
html_logo = "_static/logo.svg"

# Canonical base URL of the deployed site (GitHub Pages). Required by
# sphinx-sitemap and used to emit <link rel="canonical"> tags — both matter
# for search-engine indexing.
html_baseurl = "https://glanzz.github.io/ferriq/"

# GitHub Pages serves files starting with underscores only if Jekyll
# processing is disabled; the deploy workflow adds .nojekyll, but keep the
# default Sphinx output layout regardless.
html_copy_source = False
html_show_sourcelink = False

html_theme_options = {
    "source_repository": "https://github.com/glanzz/ferriq",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#6d28d9",
        "color-brand-content": "#6d28d9",
    },
    "dark_css_variables": {
        "color-brand-primary": "#a78bfa",
        "color-brand-content": "#a78bfa",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/glanzz/ferriq",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0"'
                ' viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0'
                " 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-."
                "01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1"
                ".13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87"
                ".87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-"
                ".87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82."
                "64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-"
                ".82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3."
                "75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21."
                '15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>'
                "</svg>"
            ),
            "class": "",
        },
    ],
}

# Extra files copied verbatim to the site root (robots.txt for crawlers)
html_extra_path = ["robots.txt"]

# -- sphinx-sitemap ----------------------------------------------------------

# Single-language, single-version site: emit clean URLs into sitemap.xml
sitemap_locales = [None]
sitemap_url_scheme = "{link}"
sitemap_excludes = ["search.html", "genindex.html"]

# -- sphinxext-opengraph (SEO / social cards) --------------------------------

ogp_site_url = "https://glanzz.github.io/ferriq/"
ogp_site_name = "Ferriq Documentation"
ogp_description_length = 200
ogp_type = "website"
ogp_enable_meta_description = True

# -- sphinx-copybutton -------------------------------------------------------

# Strip shell prompts and Python REPL prompts when copying
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
