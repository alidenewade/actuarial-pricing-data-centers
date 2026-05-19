# Wearables Paper — IA Research Series, Paper 2

**Title:** Mortality of the Quantified Self — A Bayesian Credibility Framework for Wearable-Derived Life Underwriting
**Author:** Ali Denewade
**Target post date:** Monday 27 July 2026 (after Exam P)
**Status:** Draft, 16 pages, compiles clean

## Files

- `wearables.tex` — self-contained LaTeX source. Compiles with `pdflatex wearables.tex` (run three times for cross-refs).
- `wearables.pdf` — the rendered 16-page paper.
- `wearables_body.tex` — body sections, kept separately for reference but already concatenated into `wearables.tex`.

## What's in the paper

13 sections covering:
1. Introduction (the wearable underwriting moment, why classical credibility needs extension, our contribution)
2. The wearable biometric signal taxonomy (with Figure 1 — signal flow diagram, and Table 1 — informativeness tiers)
3. The Holmström informativeness principle formalised (Proposition 3.1)
4. Bayesian credibility for wearable-augmented mortality (Poisson-Gamma update, closed-form Z(t), Figure 2 — credibility convergence)
5. From posterior intensity to premium (exponential-tilt adjustment)
6. Failure mode 1 — selection on wearable ownership (IPW correction)
7. Failure mode 2 — signal substitution and gameability (wear-time normalisation Z*(t))
8. Failure mode 3 — intergenerational fairness
9. Regulatory regime comparison (Table 2 — POPIA/GDPR/HIPAA/NY DFS)
10. Worked example: pricing a 25-year-old with 12 months of Apple Watch data ($340 → $304, 10.6% reduction)
11. Implications for product design, capital, and market structure
12. Open problems
13. Conclusions
+ Bibliography (9 entries)

## What you should review before posting

1. **The hazard ratios in Table 1** — these are illustrative ranges drawn from a synthesis of cohort studies. Before posting, verify the ranges against the actual referenced papers (Strain 2020 in Nature Medicine, Hippisley-Cox 2017 in BMJ, UK Biobank). Don't get caught with a wrong HR on a LinkedIn challenge.

2. **The worked example arithmetic** — the 28% wearable-implied intensity reduction, the effective prior precision argument, and the final 10.6% premium discount. These compound through several steps; sanity-check the maths before posting.

3. **POPIA references** — I cited POPIA §71, §69, §26, §72. Verify these section numbers against the current POPIA text. They were correct as of the 2021 promulgation but check for amendments.

4. **The Discovery Vitality framing** — the remark about Vitality's "two-decade lead" and POPIA preparation should be fact-checked against Discovery's actual founding timeline.

5. **Bibliography entries 8 and 10 (Strain and ZK Health)** — Strain 2020 is real and you can verify the citation, but the "ZK Health Working Group" reference is a generic placeholder. Either find a real ZKP-for-biometrics paper or remove that section.

## Reuse from Paper 1 (Data Centers)

The following machinery ports identically:
- IA visual theme (colours, typography, callout boxes)
- The Holmström principle was *invoked* in Paper 1; here it's *formalised* with a proposition and proof sketch
- The Bayesian credibility update was *used* in Paper 1 §6.3 (engineering prior + claims experience); here it's the central machinery
- The hazard-process pattern λ(t) = λ_0 · exp(covariate adjustment) is the same shape

## Compilation

```
pdflatex wearables.tex
pdflatex wearables.tex
pdflatex wearables.tex
```

Three passes needed for the TOC, cross-references, and bibliography to settle.

## Next paper in the series

Paper 3, scheduled for late September 2026 posting:
"Compound Drought–Heat Stress on Smallholder Crop Insurance — A Copula Framework for Sub-Saharan African Index Products"
