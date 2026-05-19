# dc_paper figures — standalone renders

Every figure, plot, chart, and diagram from
*A Coupled Power–Thermal–Cyber Framework for the Actuarial Pricing
and Insurance of Hyperscale Data Centers*
(Paper 1 of the Intelligent Actuaries research series), rendered as
standalone publication-grade artefacts ready for website upload.

## Folder layout

```
dc_paper_figures/
├── README.md      (this file)
├── pdf/           16 vector PDFs, one per figure (use these where possible)
├── png/           16 raster PNGs at 300 DPI (drop-in for HTML <img>)
└── src/           17 .tex files — fig_preamble.tex + one standalone .tex per figure
                   (compile any with `pdflatex -shell-escape FigNN_….tex`)
```

Total: **2.6 MB** (PDFs 752 KB, PNGs 1.7 MB, sources 80 KB).

Every PDF and PNG has the **Intelligent Actuaries** bone background
(`#FAFAF7`) baked in, the burnt-sienna accent (`#A04A1F`) preserved,
and Palatino+matching-math typography from `mathpazo`. No PDF.js or
sRGB-quantising viewer can wash these.

## Figure-by-figure index

The `FigNN_` prefix is source-order in the paper (top-to-bottom across
`dc_paper.tex` → `cooling_section.tex` → `econ_section.tex` → `appendix.tex`).
The `fig_<slug>` part is the LaTeX `\label{}` used in the paper, so
the in-paper `Figure~\ref{fig:slug}` citation can be cross-walked
unambiguously.

| Filename stem | LaTeX label | Source file | Short caption |
|---|---|---|---|
| `Fig01_fig_singleline` | `fig:singleline` | `dc_paper.tex` | Single-line wireframe of a Tier-IV $2N$ hyperscale data center |
| `Fig02_fig_fta` | `fig:fta` | `dc_paper.tex` | Simplified fault tree for the top event $T=$ "IT load lost" |
| `Fig03_fig_markov` | `fig:markov` | `dc_paper.tex` | Continuous-time Markov chain on plant states for a $2N$ data center |
| `Fig04_fig_arr` | `fig:arr` | `dc_paper.tex` | Arrhenius temperature acceleration of the baseline failure rate |
| `Fig05_fig_oep` | `fig:oep` | `dc_paper.tex` | Annual Occurrence Exceedance Probability curve for the worked example |
| `Fig06_fig_cooling-topology` | `fig:cooling-topology` | `cooling_section.tex` | Three cooling regimes the underwriter must distinguish (air / D2C / immersion) |
| `Fig07_fig_cool-hazard` | `fig:cool-hazard` | `cooling_section.tex` | Cooling hazard $\lambda^{\mathrm{cool}}$ as a function of wet-bulb temperature |
| `Fig08_fig_cooling-faulttree` | `fig:cooling-faulttree` | `cooling_section.tex` | Cooling-system fault tree (chiller / distribution / liquid–electrical / thermal runaway) |
| `Fig09_fig_micro` | `fig:micro` | `econ_section.tex` | Operator's optimisation $\min_K C(K)$ — total cost vs reliability capex |
| `Fig10_fig_supdem` | `fig:supdem` | `econ_section.tex` | Market equilibrium in the data-center insurance market |
| `Fig11_fig_supdem-shift` | `fig:supdem-shift` | `econ_section.tex` | Comparative statics: shifts in actuarial parameters move the equilibrium |
| `Fig12_fig_compute-mkt` | `fig:compute-mkt` | `econ_section.tex` | Compute-services market equilibrium with reliability shock |
| `Fig13_fig_contagion` | `fig:contagion` | `econ_section.tex` | Macro-financial contagion network triggered by a hyperscale data-center failure |
| `Fig14_fig_thevenin` | `fig:thevenin` | `appendix.tex` | Thévenin equivalent looking back from a fault point $F$ at the LV bus |
| `Fig15_fig_ups` | `fig:ups` | `appendix.tex` | Double-conversion (online) UPS topology |
| `Fig16_fig_itic` | `fig:itic` | `appendix.tex` | Stylised ITIC/CBEMA voltage tolerance envelope |

## Rebuilding any single figure

The `.tex` sources in `src/` are self-contained. From within `src/`:

```bash
pdflatex -shell-escape Fig03_fig_markov.tex
```

Each one loads `fig_preamble.tex`, which mirrors the relevant
preamble lines from `dc_paper.tex`: same colour palette, same TikZ
libraries, same math macros (`\Prb`, `\E`, `\1`, `\dd`, `\VaR`,
`\TVaR`, `\argmin`, `\argmax`, …), same Palatino+helvet+microtype
typography. Compiling a fig source produces a perfectly cropped
PDF on a bone background.

## Suggested website use

- **Articles / blog posts:** use the **PNGs** in `png/` as `<img>` sources. They're rasterised at 300 DPI so they hold up at any reasonable display size and look correct on retina displays.
- **Download links / paper repository:** use the **PDFs** in `pdf/` — they're vector, scale to any size without loss, and embed all fonts.
- **Re-stylable figures (e.g. dark-mode site):** edit the `\definecolor{iabone}{...}` line in `src/fig_preamble.tex` and recompile any subset; nothing else needs to change.

## Provenance

Generated 2026-05-19 from
`/home/adu-00/ali/actuarial-pricing-data-centers/paper/dc_paper.tex`
(post-polish-pass HEAD, the same source that built the 49-page paper PDF).
