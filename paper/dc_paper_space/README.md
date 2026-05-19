---
title: Hyperscale Data-Center Risk Simulation
emoji: ⚡
colorFrom: orange
colorTo: red
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
license: mit
short_description: Power-thermal-cyber actuarial pricing for hyperscale data centers
---

# Hyperscale Data-Center Risk Simulation

Interactive companion to
**A Coupled Power-Thermal-Cyber Framework for the Actuarial Pricing and
Insurance of Hyperscale Data Centers**
(Denewade 2026, Paper 1 of the Intelligent Actuaries research series).

DOI: [10.5281/zenodo.20279225](https://doi.org/10.5281/zenodo.20279225) ·
Paper: [intelligentactuaries.com/research/data-centers](https://intelligentactuaries.com/research/data-centers) ·
Code: [github.com/alidenewade/actuarial-pricing-data-centers](https://github.com/alidenewade/actuarial-pricing-data-centers)

## Try It Live

🚀 **[Run the simulation on Hugging Face Spaces](https://huggingface.co/spaces/intelligentactuaries/hyperscale-dc-risk-simulation)**

No installation required. Pick a mode, adjust parameters in the left
sidebar, and the actuarial engines recompute on every rerun. No
precomputed results.

## What This Simulation Demonstrates

The simulator exposes every quantitative engine in the paper as a
sidebar control: a four-state Markov plant model, a Negative-Binomial /
Generalised-Pareto compound annual loss, an
Occurrence-Exceedance-Probability (OEP) Monte Carlo, the technical-premium
calculator under five classical premium principles, and a side-by-side
comparison harness for two configurations under a shared seed. Every
output is a Plotly chart on the Intelligent Actuaries bone background.

## User Interface

The app uses [Streamlit](https://streamlit.io). All adjustment widgets
(sliders, selectors, seed inputs) live in the **left sidebar**,
organised by mode. The main view shows six tabs as a segmented control
at the top:

| Tab | Paper section | What you change | What you see |
|---|---|---|---|
| 🏗️  **Reliability** | §5 (Markov plant) | Per-train λ, μ, λ_f, μ_r; Tier topology | Stationary distribution π, availability A, MTBF, λ^out |
| 📉 **Compound loss** | §7 + §8 | NB(ν, λ) frequency + lognormal-body / GPD-tail severity | Histogram of S = ΣX_j, mean, p99.5, max |
| 🌪️  **OEP curve** | §16 (worked example) | Engineering anchor + severity + n_years Monte Carlo | Annual occurrence-exceedance curve with 1-in-200 line |
| 💰 **Pricing** | §10 (premium principles) | Risk-aversion knobs (a, b, h, λ*) | Pure, SD, Var, Esscher, Wang premiums + loading table |
| ⚖️  **Compare** | side-by-side underwriting | Two sets of (λ, ξ, σ, u, π_tail) under one seed | Summary table + overlaid OEP curves |
| 🖼️  **Figures** | gallery | (browse only) | All 16 paper figures rendered inline |

## What This Simulation Does Not Demonstrate

- The simulator is calibrated on stylised parameters, not on confidential
  industry claims data. The qualitative shape of every output is
  correct; specific dollar figures are illustrative.
- The Markov chain is steady-state. Operational reality often involves
  non-stationary regimes (maintenance windows, climate drift). For a
  fully time-varying treatment, see the doubly-stochastic Cox formulation
  in §7 of the paper.
- Tail-dependence is currently handled via a single Gaussian copula on the
  catastrophic layer. The Gumbel copula treatment of §9.3 is not yet
  exposed via the UI; it lives in the companion `dcrisk` Python package.

## Methodology

Every formula is taken verbatim from the paper. See:

- **Markov stationary U** — eq. (10) in §5
- **Compound aggregate** — eq. (22) in §7
- **GPD POT severity** — eq. (33) in §8
- **Premium principles** — §10.1-§10.5
- **Solvency II 1-in-200 anchor** — §13.1

The companion `dcrisk` Python package
([github.com/alidenewade/actuarial-pricing-data-centers](https://github.com/alidenewade/actuarial-pricing-data-centers))
provides a GPU-accelerated Monte Carlo back-end that reproduces every
numerical result in the paper to 3 significant figures.

## Visual Identity

The Space mirrors the chrome of the lab's other simulators
([nanoeconomics-simulation](https://huggingface.co/spaces/intelligentactuaries/nanoeconomics-simulation))
so the two read as siblings of the same research lab.

- Bone background `#FAFAF7`, deep warm near-black headings `#1B1815`,
  burnt-sienna accents `#A04A1F`. Same palette as the paper PDF and the
  `intelligentactuaries.com` website.
- Theme-aware CSS via `prefers-color-scheme` so Plotly text contrast
  remains readable under both light and dark Streamlit themes.
- Bundled `figures/` directory ships all 16 paper PNG figures so the
  Figures tab works fully offline.

## Run locally

```bash
git clone <this-space-url>
cd hyperscale-dc-risk-simulation
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501> in your browser.

For development with `uv`:

```bash
uv sync
uv run streamlit run app.py
```

## Research

This work is produced by the **Intelligent Actuaries Research Lab**.
For questions, collaborations, or feedback, contact
[research@intelligentactuaries.com](mailto:research@intelligentactuaries.com).

## Reference

Denewade, A. (2026). *A Coupled Power-Thermal-Cyber Framework for the
Actuarial Pricing and Insurance of Hyperscale Data Centers.* Zenodo.
DOI: [10.5281/zenodo.20279225](https://doi.org/10.5281/zenodo.20279225).

## Citation

```bibtex
@misc{Denewade2026DataCenters,
  author    = {Denewade, Ali},
  title     = {A Coupled Power--Thermal--Cyber Framework for the
               Actuarial Pricing and Insurance of Hyperscale Data Centers},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20279225},
  url       = {https://doi.org/10.5281/zenodo.20279225}
}
```

## License

MIT, © 2026 Ali Denewade / Intelligent Actuaries.
