# DESIGN.md — Linnaean botanical specimen world

The site reads as a page from a modern *Systema Naturae*: a quiet,
paper-toned catalog where every model artifact is a pressed specimen and
every benchmark number is a measurement plate. Premium through restraint —
generous whitespace, engraved serifs, hairline rules — never through gloss.

## Palette

| Token | Value | Role |
| --- | --- | --- |
| `paper` | `#F7F5EE` | warm ivory ground (herbarium sheet) |
| `paper-deep` | `#EFEBDD` | recessed panels, table zebra |
| `ink` | `#1A2B1E` | near-black green ink — body text |
| `moss` | `#2E5339` | primary green — headings, key UI |
| `leaf` | `#4A7C59` | interactive green — links, accents |
| `fern` | `#8FAE94` | muted secondary green — captions, meta |
| `vein` | `#D8D2BF` | hairline rules, table borders |
| `specimen` | `#B5541E` | rare accent — a single pressed-flower warm note, only for the one number that matters |

No gradients, no shadows-as-depth (use hairline borders + paper-depth),
no dark mode v1.

## Typography

- **Display / headings**: Fraunces (serif, high-contrast, engraved-plate feel)
- **Latin binomials & labels**: Fraunces italic — species names always italic
- **Body / UI**: Inter
- **Data / scores**: JetBrains Mono or IBM Plex Mono — tabular figures
- Heading scale restrained; Latin labels in small caps + letterspaced

## Motifs

- **Binomial naming**: artifacts rendered as *Linnaeus decisio* var. *octo*,
  var. *quatuor* — italic genus+species, roman variety
- **Specimen plate**: benchmark numbers inside hairline-bordered cards with
  a label strip (nº, name, provenance) like a catalog entry
- **Taxonomic key**: the predict() contract shown as an indented
  classification key (choice → noul → score), not a flowchart
- **Herbarium rule**: thin double rules under section headings;
  section markers as `§` or plate numbers ("Tab. I")
- **Leaf sprig**: a single fine-line botanical SVG mark (stem + leaflets)
  used sparingly as the logo-ish glyph

## Signature interaction

Minimal motion. Hover on specimen cards lifts the label strip opacity.
Numbers count nowhere — static measured values, printed like plates.
The one flourish: a thin growing stem line that draws on hero load.

## Surface anatomy (landing)

1. Hero — wordmark `LINNAEUS` + *Linnaeus decisio*, one-line mechanism
   ("a taxonomist for decisions"), the four numbers as a plate strip
2. Mechanism — state + questions → distributions, as taxonomic key
3. Specimen plates — benchmark tables (held-out / JevBench / on-device)
4. Genus table — the six HF artifacts as species entries
5. Docs index + footer (named for Linnaeus, upstream acknowledgement)
