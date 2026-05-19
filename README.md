# dcrisk — Actuarial Pricing & Insurance for Hyperscale Data Centers

Coupled power–thermal–cyber simulation, dependence modelling, and actuarial
pricing for hyperscale data-center insurance, accompanying the working paper
*A Coupled Power–Thermal–Cyber Framework for the Actuarial Pricing and
Insurance of Hyperscale Data Centers* (Denewade, 2026).

The LaTeX source in `paper/` is the canonical text; the Python package in
`src/dcrisk/` operationalises the models so every figure and table can be
regenerated from scratch.

## Repository layout

```
.
├── paper/                       # LaTeX source + build
├── src/dcrisk/                  # simulation + pricing library
│   ├── reliability/             # Markov chain, fault tree, hazard
│   ├── severity/                # lognormal body, GPD tail, mixture
│   ├── frequency/               # NB / Cox process
│   ├── copula/                  # Gaussian + Gumbel
│   ├── sde/                     # coupled (V, T, C) Euler–Maruyama
│   ├── cooling/                 # COP, wet-bulb, hazard, climate
│   ├── pricing/                 # pure premium, loadings, XoL
│   ├── econ/                    # operator optimum + market equilibrium
│   ├── monte_carlo/             # compound aggregator, Cox sampling
│   └── dashboards/              # Streamlit companion
├── notebooks/                   # figure-by-figure reproductions
├── tests/                       # pytest smoke tests
├── pyproject.toml
└── Makefile                     # paper / sim / app / test
```

## Quick start

```bash
pip install -e .
make paper          # builds paper/build/dc_paper.pdf
make sim            # runs the notebook suite
make app            # Streamlit companion on :8501
make test
```

A GPU-accelerated path exists for the larger Monte Carlo runs; it is
optional and the CPU path produces the same results at lower throughput.

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
| §8  lognormal–GPD severity              | `dcrisk.severity.{lognormal,gpd,mixture}`  |
| §9  copula dependence                   | `dcrisk.copula.{gaussian,gumbel}`          |
| §10 eq. (11) coupled SDE                | `dcrisk.sde.ptcyber`                       |
| §11 premium principles                  | `dcrisk.pricing.loadings`                  |
| §12.1 XoL closed form (GPD)             | `dcrisk.pricing.xol`                       |
| §13 eq. (50) operator cost              | `dcrisk.econ.operator`                     |
| §13 eq. (52)–(53) equilibrium           | `dcrisk.econ.market`                       |
| §14 passthrough elasticity              | `dcrisk.econ.incidence`                    |
| §11 compound aggregator                 | `dcrisk.monte_carlo.compound`              |

## Series Paper 2 — Wearables

*Mortality of the Quantified Self — A Bayesian Credibility Framework for
Wearable-Derived Life Underwriting.* Draft in `paper/wearables.tex`. Build
with `make wearables`, or `make papers` to build both.

## License

[MIT](./LICENSE) — © 2026 Ali Denewade.
