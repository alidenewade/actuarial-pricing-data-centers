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

---

## Repository layout

```
.
├── paper/                      # LaTeX source
│   ├── dc_paper.tex            # main document
│   ├── appendix.tex
│   ├── econ_section.tex
│   ├── build/                  # latexmk output (gitignored contents)
│   └── reference/
│       ├── dc_paper_original.pdf   # baseline build, retained for comparison
│       └── README.md
├── src/dcrisk/                 # simulation + pricing library
│   ├── reliability/            # Markov chain, fault tree, Arrhenius hazard
│   ├── severity/               # Lognormal body, GPD tail, mixture
│   ├── frequency/              # NB / Cox process
│   ├── copula/                 # Gaussian + Gumbel
│   ├── sde/                    # coupled (V, T, C) Euler–Maruyama
│   ├── pricing/                # pure premium, loadings, XoL
│   ├── econ/                   # operator optimum + market equilibrium
│   └── dashboards/             # Streamlit interactive front-end
├── notebooks/                  # six paper-aligned notebooks
├── tests/                      # pytest smoke + unit tests
├── pyproject.toml              # hatchling
├── Makefile                    # paper / sim / app / test / clean
└── README.md
```

---

## Quick start

```bash
# 1. Create the dedicated conda env (one-off, done by the setup script).
conda activate dcrisk

# 2. Install the package in editable mode.
pip install -e .

# 3. Build the paper.
make paper          # → paper/build/dc_paper.pdf

# 4. Run all notebooks end-to-end.
make sim

# 5. Launch the interactive dashboard.
make app            # → http://localhost:8501

# 6. Run tests.
make test
```

---

## Mapping from paper to code

| Paper section / equation        | Module                                     |
| ------------------------------- | ------------------------------------------ |
| §4, eq. (7) — 4-state Markov    | `dcrisk.reliability.markov`                |
| §5 — Arrhenius hazard           | `dcrisk.reliability.arrhenius`             |
| §6 — fault-tree top events      | `dcrisk.reliability.fault_tree`            |
| §7 — NB / Cox frequency         | `dcrisk.frequency.nb_gamma`                |
| §8 — lognormal–GPD severity     | `dcrisk.severity.{lognormal,gpd,mixture}`  |
| §9 — copula dependence          | `dcrisk.copula.{gaussian,gumbel}`          |
| §10, eq. (11) — coupled SDE     | `dcrisk.sde.ptcyber`                       |
| §11 — premium principles        | `dcrisk.pricing.loadings`                  |
| §12.1 — XoL closed form (GPD)   | `dcrisk.pricing.xol`                       |
| §13, eq. (50) — operator cost   | `dcrisk.econ.operator`                     |
| §13, eq. (52)–(53) — equilibrium | `dcrisk.econ.market`                      |
| §14 — passthrough elasticity    | `dcrisk.econ.incidence`                    |

---

## Hardware notes

Developed on an HP Victus laptop (AMD Ryzen AI 7 350, RTX 5060 8 GB Max-Q,
38 GB RAM, Ubuntu 24.04). The simulation code is **CPU-only by design** so
the same notebooks run unchanged on any modern x86_64 machine. GPU
acceleration, if ever needed, will live on a separate workstation.

---

## License

[MIT](./LICENSE) — © 2026 Ali Denewade.
