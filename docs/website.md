# Build the website

The static website uses Sphinx, Furo, and MyST. Sphinx builds the pages and
search index, Furo supplies navigation and responsive layouts, and MyST reads
the existing Markdown. The site uses the README, model card, guides, and saved
figures directly; do not maintain separate copies of their content.

From the repository root, with [mise](https://mise.jdx.dev/) installed:

```bash
mise install
mise run sync-docs
mise run docs-preview
```

Open `http://localhost:8000`. The generated files are in `build/site`.
`mise run sync-docs` selects only the `docs` group from `uv.lock`, without
installing the project or its model dependencies. The build does not import Linnaeus,
execute the examples, load models, train, or download datasets.

Keep navigation in the existing MyST `toctree` directives. Use `docs/conf.py`
for theme settings and `docs/_static/brand.css` for small styling changes.
Keep the existing logo and published figures as the visual sources.

Use `mise run docs` for a build without the preview server. Dependency versions and
commands live in `pyproject.toml`; see [development](development.md) for checks
and lock updates.

## GitHub Pages

The website workflow builds pull requests without publishing. Pushes to `main`
and manual runs on `main` build and deploy the static output to GitHub Pages.
Set the repository's Pages source to **GitHub Actions** before the first deploy.
The deployment uses the repository's `github-pages` environment and respects any
configured protection rules. It requires no custom server, external hosting
account, or repository secret.
