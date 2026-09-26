# PRODUCT.md — Linnaeus

## Product truth

Linnaeus is a 2B-parameter **decision model**: given a structured state and a
set of typed questions (choice / noul / score), it returns calibrated
probability distributions over caller-supplied candidates — without generating
free text. Forked from Dohnuts, re-based on Qwen3.5-2B, trained on CUDA,
deployable on Apple Silicon via MLX exports.

Hard facts (measured, do not inflate):
- Held-out macro accuracy 80.78% (26 groups, 180k) — upstream 78.21%
- JevBench v1.2.2: 73.16% (231 tasks) — top of the ~2B local-model class
- On-device: MLX 8bit 70.56%, VLM builds keep image decisions
- Artifacts on HF: adapter + merged + MLX 8/4bit + VLM 8/4bit
- Known weakness: hard/temporal_numeric (2/15), hard/long-policy borderlines

## Naming (brand commitment)

Named for **Carl Linnaeus** — the model is a *taxonomist for decisions*:
it takes an unstructured state and classifies it into the right species of
intent. The Linnaean frame is the product identity, not decoration:
candidates are specimens, the model assigns each question its correct class,
with calibrated confidence. Binomial/italic Latin naming, taxonomy tables,
and specimen-plate layouts are legitimate motif territory.

## Audience & mode

- Primary surface: project site — **Persuade** (landing) + **Read** (docs)
- Audience: developers evaluating small local decision/classification models
  for iOS/macOS apps and GPU serving; open-source ML practitioners.
- Success: visitor believes "a 2B model that decides instead of generates is
  real, measured, and runs on my phone" → clones repo or pulls HF weights.

## Constraints (inferred from brief — labeled assumptions)

- Stack pinned: **Nuxt + UnoCSS**, pnpm, GitHub Pages SSG
- Palette pinned: green family; mood: "生命的高级感" — botanical, alive,
  premium. Not neon/cyber — herbarium paper, ink green, specimen elegance.
- Docs content lives in `docs/*.md` — rendered, not duplicated.
- No upstream (Dohnuts) brand assets. Acknowledgements stay in repo text.
