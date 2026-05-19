# A Coupled Power–Thermal–Cyber Framework for the Actuarial Pricing and Insurance of Hyperscale Data Centers

Self-contained source + compiled-PDF bundle of Paper 1 of the Intelligent Actuaries research series, ready for upload to the website.

**Author:** Ali Denewade — Founder, *Intelligent Actuaries* / SOA examination pathway.
**Compiled PDF:** `dc_paper.pdf` (49 pages, ~462 KB, A4).

## Contents

| File | Bytes | SHA-256 (12-char prefix) | Purpose |
|---|---:|---|---|
| `dc_paper.pdf` | 462 KB | `91acb6f2db7a…` | Compiled paper |
| `dc_paper.tex` | 76 KB | `aae8b2917544…` | LaTeX root (preamble + introduction + reliability + SDE + frequency + severity + dependence + pricing + product design + pgfplots views) |
| `cooling_section.tex` | 30 KB | `9897d89f3281…` | §4 Cooling infrastructure as a first-class actuarial risk (`\input` from root) |
| `econ_section.tex` | 34 KB | `00fc95df552d…` | §14 Microeconomic and macroeconomic implications (`\input` from root) |
| `appendix.tex` | 19 KB | `9dad677bbb42…` | Appendix A: Electrical engineering foundations (`\input` from root) |
| `build.sh` | — | — | One-line LaTeX rebuild script |

No external images. Every figure in the paper is TikZ-generated inline.

## How to rebuild

Requires TeX Live 2023+ with `texlive-latex-extra`, `texlive-fonts-extra`, `texlive-pictures`, `texlive-science`, `texlive-publishers`, `texlive-extra-utils`, `latexmk`, `biber`, and `poppler-utils`.

```bash
./build.sh
```

or manually:

```bash
latexmk -pdf -interaction=nonstopmode -file-line-error -synctex=1 \
  -shell-escape -outdir=build dc_paper.tex
```

Three pdflatex passes are run automatically by `latexmk` so cross-references and the ToC settle on the first run. Build wall-clock is ~7–10 s on a modern x86_64.

## Visual theme

The paper uses the Intelligent Actuaries colour palette (bone background `#FAFAF7`, deep-warm-near-black headings `#1B1815`, burnt-sienna accents `#A04A1F`). PDF.js-based viewers (including the VS Code in-tab PDF preview) wash these colours; open the PDF in Evince, Okular, Adobe, or any Poppler-based viewer for accurate rendering.

## Citation

Please cite as:

```bibtex
@misc{Denewade2026DataCenters,
  author = {Ali Denewade},
  title  = {A Coupled Power--Thermal--Cyber Framework for the
            Actuarial Pricing and Insurance of Hyperscale Data Centers},
  year   = {2026},
  note   = {Working paper, Intelligent Actuaries research series, Paper~1.}
}
```

## Companion code

A GPU-accelerated Monte Carlo reproduction of every numerical result in the paper is available at
<https://github.com/alidenewade/actuarial-pricing-data-centers>.
