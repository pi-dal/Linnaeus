# Build the website

The site is a Nuxt 4 + UnoCSS app in `site/` (SSG, deployed to GitHub Pages).
It renders the Markdown in `docs/` directly via `@nuxt/content` — the `docs/`
Markdown stays the single source of truth; never maintain copies.

Toolchain: mise provisions Node 24 + pnpm 12 (`mise.toml`).

```bash
mise install          # node + pnpm (plus python/uv for the rest)
mise run site         # pnpm install + nuxi generate → site/.output/public
mise run site-preview # serve on http://localhost:8000
```

## Conventions

- Visual system: `DESIGN.md` (Linnaean botanical specimen world — paper,
  ink green, Fraunces serif, specimen-plate tables).
- Landing + genus table live in `site/app/pages/index.vue`; doc pages render
  `docs/*.md` through `app/pages/docs/[...slug].vue`.
- Keep plain-Markdown docs — no Sphinx/MyST directives (`{toctree}`,
  `{include}`, `{eval-rst}`); Nuxt Content renders standard Markdown + tables.
- Figures stay in `docs/figures/` and are referenced by relative path.

## GitHub Pages

`.github/workflows/website.yml` builds pull requests without publishing and
deploys `.output/public` on `main`. Set Pages source to **GitHub Actions**
before the first deploy; it uses the `github-pages` environment and needs no
secrets or external hosting.
