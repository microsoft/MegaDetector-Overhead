---
description: "How to build and deploy the MegaDetector-Overhead MkDocs documentation site locally and to GitHub Pages."
tags:
  - MkDocs
  - documentation
  - developer guide
  - MegaDetector-Overhead
---

# Developer Guide — Building the Docs

This page explains how to build and preview the MegaDetector-Overhead documentation site locally.

## Prerequisites

Install the documentation dependencies (separate from the ML requirements):

```bash
pip install -r docs-requirements.txt
```

## Preview locally

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The server hot-reloads on file changes.

## Build (offline check)

```bash
mkdocs build --strict
```

`--strict` treats warnings as errors. Fix any broken links or missing pages before opening a PR.

## Deploy to GitHub Pages

Deployment is automatic. The [GitHub Actions workflow](https://github.com/microsoft/MegaDetector-Overhead/blob/main/.github/workflows/deploy-docs.yml) triggers on every push to `main` that touches `docs/**`, `mkdocs.yml`, or `docs-requirements.txt`.

To deploy manually (maintainers only):

```bash
mkdocs gh-deploy --force
```

This builds the site and force-pushes to the `gh-pages` branch. Do not commit the `site/` directory — it is generated and is in `.gitignore`.

## Adding a new page

1. Create a new `.md` file under `docs/`
2. Add SEO front matter at the top:
   ```yaml
   ---
   description: "One sentence describing this page for search engines."
   tags:
     - relevant tag
     - another tag
   ---
   ```
3. Add the page to the `nav:` section in `mkdocs.yml`
4. Run `mkdocs build --strict` to verify no errors
