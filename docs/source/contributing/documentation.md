---
myst:
  html_meta:
    description: "Working on the SimQ documentation — building the Sphinx site locally, page conventions, GitHub Pages deployment, and SEO setup."
---

# Working on the docs

This site is built with [Sphinx](https://www.sphinx-doc.org/) using the
[Furo](https://pradyunsg.me/furo/) theme and Markdown (MyST). Sources live
in `docs/source/`; the site deploys to GitHub Pages automatically.

## Build locally

```bash
pip install -r docs/requirements.txt
cd docs
make html            # output in docs/build/html
python -m http.server -d build/html  # preview at http://localhost:8000
```

For live reload while editing:

```bash
pip install sphinx-autobuild
make livehtml
```

Treat warnings as errors before pushing — CI does:

```bash
sphinx-build -W --keep-going -b html source build/html
```

## Layout

```text
docs/
├── requirements.txt        # pinned Sphinx toolchain
├── Makefile / make.bat
└── source/
    ├── conf.py             # theme, SEO, sitemap configuration
    ├── index.md            # landing page
    ├── robots.txt          # copied to the site root
    ├── _static/            # CSS, logo, favicon
    ├── getting-started/    # installation + quickstarts
    ├── guide/              # user guide (one topic per page)
    ├── examples/           # runnable example catalogue
    ├── api/                # Rust + Python API reference pages
    ├── architecture/       # crate-level architecture
    └── contributing/       # this section
```

## Page conventions

- **Markdown (MyST)** for all pages; reStructuredText only if a directive
  demands it.
- Start every page with a `myst.html_meta.description` front-matter block —
  it becomes the `<meta name="description">` tag search engines display.
- One H1 per page; keep headings ≤ H3 so the sidebar stays scannable.
- Code samples must be real: lifted from working examples or doc tests, not
  invented. If an API changes, update the docs in the same PR.
- Link source files on GitHub with full URLs
  (`https://github.com/glanzz/simq/blob/main/...`).

## Deployment (GitHub Pages)

`.github/workflows/docs.yml` builds the site on every push to `main` or
`documentation` that touches `docs/**` and deploys it to GitHub Pages at
<https://glanzz.github.io/simq/>. Pull requests get a build-only check (no
deploy), so broken docs never land.

One-time repository setup (already done if the site is live):

1. GitHub → **Settings → Pages**
2. Set **Source** to **GitHub Actions** (not "Deploy from a branch")

## SEO

Search-engine indexing is set up in `conf.py`:

- `html_baseurl` emits canonical URLs on every page
- `sphinx-sitemap` generates `sitemap.xml`
- `robots.txt` (copied to the site root) allows all crawlers and points at
  the sitemap
- `sphinxext-opengraph` adds OpenGraph/social-card and meta-description
  tags
- Per-page descriptions come from the front-matter convention above

To accelerate indexing after major changes, submit
`https://glanzz.github.io/simq/sitemap.xml` in
[Google Search Console](https://search.google.com/search-console) and
[Bing Webmaster Tools](https://www.bing.com/webmasters).
