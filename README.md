# dcrisk — Actuarial Pricing & Insurance for Hyperscale Data Centers

Coupled power–thermal–cyber simulation, dependence modelling, and actuarial
pricing for hyperscale data-center insurance, accompanying the working paper
*A Coupled Power–Thermal–Cyber Framework for the Actuarial Pricing and
Insurance of Hyperscale Data Centers*.

**Provenance.** The paper and the simulation code in this repository form a
single research artefact authored by **Ali Denewade** (actuary-in-training,
SOA pathway; sessional lecturer at the University of the Witwatersrand).
The LaTeX source in `paper/` is the canonical text; the Python package in
`src/dcrisk/` operationalises the models so every figure and table can be
regenerated from scratch.

## Two-machine workflow

| Role | Machine | Use |
| ---- | ------- | --- |
| Editing | HP Victus laptop *(fas-datart-2026)* | LaTeX writing, code edits, lightweight CPU runs |
| Heavy compute | Workstation *(adu-00, AMD Ryzen 9 9950X, RTX 5090 32 GB)* | Large Monte Carlo, copula bootstraps, GPU-accelerated SDE, Streamlit served over Tailscale |

The repo is a single git tree synced between both via the `main` branch.
GPU code paths and notebooks live in this same tree; CPU-only paths still
work unchanged on the laptop.

## Repository layout

```
.
├── paper/                       # LaTeX source
│   ├── dc_paper.tex             # main document
│   ├── appendix.tex
│   ├── econ_section.tex
│   ├── cooling_section.tex      # §4 — cooling thermodynamics + hazard
│   ├── build/                   # latexmk output (gitignored contents)
│   └── reference/
│       ├── dc_paper_original.pdf
│       └── README.md
├── src/dcrisk/                  # simulation + pricing library
│   ├── reliability/             # Markov chain, fault tree, Arrhenius hazard
│   ├── severity/                # lognormal body, GPD tail, mixture
│   ├── frequency/               # NB / Cox process
│   ├── copula/                  # Gaussian + Gumbel
│   ├── sde/                     # coupled (V, T, C) Euler-Maruyama (CPU + JAX-GPU)
│   ├── cooling/                 # COP, wet-bulb, lambda_cool, T_wb climate
│   ├── pricing/                 # pure premium, loadings, XoL
│   ├── econ/                    # operator optimum + market equilibrium
│   ├── monte_carlo/             # compound aggregator, Cox sampling, parallel dispatch
│   └── dashboards/              # Streamlit (binds to Tailscale IP on adu-00)
├── notebooks/
│   ├── 01_compound_nb_gpd.ipynb
│   ├── 02_markov_chain_reliability.ipynb
│   ├── 03_copula_dependence.ipynb
│   ├── 04_sde_simulation_cpu.ipynb
│   ├── 04b_sde_simulation_gpu.ipynb         # JAX on RTX 5090
│   ├── 05_worked_example_200MW.ipynb
│   ├── 06_supply_demand_equilibrium.ipynb
│   ├── 07_cooling_hazard.ipynb              # §4 figures
│   └── 08_oep_full_simulation.ipynb         # 10^4-10^5 years, GPU
├── tests/                       # pytest smoke + GPU smoke
├── pyproject.toml               # hatchling
├── Makefile                     # paper / sim / sim-gpu / app / test / bench
└── setup.log                    # full preflight + install history (per host)
```

## Quick start on adu-00

```bash
conda activate dcrisk-gpu       # created by the Step-3 setup
pip install -e .                # editable install of the dcrisk package
make paper                      # paper/build/dc_paper.pdf
make sim-gpu                    # only the GPU notebooks (04b, 08)
make test-gpu                   # JAX + Torch + CuPy see the RTX 5090
make app                        # Streamlit bound to Tailscale IP, port 8501
make app-tunnel                 # prints the URL to open from the laptop
make bench                      # CPU vs GPU SDE wall-clock
```

On the laptop (`fas-datart-2026`) the env is named `dcrisk` (no GPU stack)
and `make sim-gpu` / `make test-gpu` are expected to fail. Everything else
works identically.

## Mapping from paper to code

| Paper section / equation                | Module                                     |
| --------------------------------------- | ------------------------------------------ |
| §4 eq. (4)  PUE decomposition           | `dcrisk.cooling.thermo`                    |
| §4 eq. (5)  Stull wet-bulb              | `dcrisk.cooling.wetbulb`                   |
| §4 eq. (7)  cooling-loss hazard         | `dcrisk.cooling.hazard`                    |
| §4 eq. (12) non-stationary T_wb(t)      | `dcrisk.cooling.climate`                   |
| §5 eq. (7)  4-state Markov reliability  | `dcrisk.reliability.markov`                |
| §6  fault-tree top events               | `dcrisk.reliability.fault_tree`            |
| §6  Arrhenius + voltage stress          | `dcrisk.reliability.arrhenius`             |
| §6  Cox-process thinning                | `dcrisk.monte_carlo.cox_process`           |
| §7  NB / Cox frequency                  | `dcrisk.frequency.nb_gamma`                |
| §8  lognormal-GPD severity              | `dcrisk.severity.{lognormal,gpd,mixture}`  |
| §9  copula dependence                   | `dcrisk.copula.{gaussian,gumbel}`          |
| §10 eq. (11) coupled SDE                | `dcrisk.sde.ptcyber` + `.ptcyber_gpu`      |
| §11 premium principles                  | `dcrisk.pricing.loadings`                  |
| §12.1 XoL closed form (GPD)             | `dcrisk.pricing.xol`                       |
| §13 eq. (50) operator cost              | `dcrisk.econ.operator`                     |
| §13 eq. (52)-(53) equilibrium           | `dcrisk.econ.market`                       |
| §14 passthrough elasticity              | `dcrisk.econ.incidence`                    |
| §11 compound aggregator                 | `dcrisk.monte_carlo.compound`              |

## Series Paper 2 — Wearables

**Title:** *Mortality of the Quantified Self — A Bayesian Credibility Framework for Wearable-Derived Life Underwriting.*
**Status:** draft. Target post date: Monday 27 July 2026.
**Sources:** `paper/wearables.tex` (root), `paper/wearables_body.tex` (body sections, also concatenated into the root file — kept for reference), `paper/wearables_README.md` (per-paper review notes).
**Built PDF:** `paper/wearables.pdf` (20 pages after the polish pass) and `paper/build/wearables.pdf` after a fresh `make wearables`.

The wearables paper extends the Intelligent Actuaries framework from
property risk (Paper 1, dcrisk) into mortality risk. It applies the same
visual house style, the same Bayesian-credibility apparatus already
present in `src/dcrisk/`, and integrates with the regulatory layer
(POPIA, GDPR, HIPAA, NY DFS CL 2019-1). Build it with:

```bash
make wearables          # builds only the wearables paper
make papers             # builds both papers in this repo
```

## LaTeX build configuration

The paper compiles with the **Intelligent Actuaries** colour theme: bone
background `#FAFAF7`, deep warm near-black headings `#1B1815`, burnt-sienna
accents `#A04A1F`, coloured callout boxes (Novel contribution / Remark /
Worked example), and pgfplots charts in colour.

**Canonical build (from project root):**

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -file-line-error -synctex=1 \
  -shell-escape -outdir=build dc_paper.tex
# -> paper/build/dc_paper.pdf  (42 pages, ~457 KB)
# -> paper/build/dc_paper.synctex.gz   (for Evince SyncTeX)
```

Or just `make paper` from the project root, which runs the same `latexmk`.

**VS Code build (LaTeX Workshop):** `.vscode/settings.json` configures the
same `latexmk` invocation. Build is auto-triggered on save.

**Viewer note.** VS Code's built-in PDF tab (PDF.js) renders this paper's
near-white bone background and tcolorbox shading slightly differently from
Poppler-based viewers — the colours can look washed out in the in-tab
preview. The byte-for-byte identical PDF renders correctly in Evince, and
`.vscode/settings.json` is configured to open `paper/build/dc_paper.pdf` in
**external Evince** after each build. Forward-search (Ctrl+Alt+J from
LaTeX Workshop) and inverse-search (Ctrl+click in Evince) both work via
the `.synctex.gz` file produced by `-synctex=1`.

**If VS Code is unavailable**, the CLI build is byte-equivalent to what
LaTeX Workshop produces — diagnostic in `setup.log` confirms a normalised
SHA-256 match of `db266edd73f805bee867c5c2…` across the two builds.

## License

[MIT](./LICENSE) — © 2026 Ali Denewade.
