# paper/reference/

This directory holds the **baseline PDF build** of `dc_paper.tex`, retained as
a reproducibility reference.

## `dc_paper_original.pdf`

- Built before the project was scaffolded into its current layout — it is the
  exact PDF that was sitting next to the `.tex` sources at the start of the
  setup session (412 365 bytes, SHA-256
  `20676bf9afc8f3ea3a2e9884088207bf36438740c936b3a86d317bca15c070d4`).
- Kept here so future builds can be compared against a known-good output
  (page count, file size, SyncTeX alignment).
- **Do not edit.** If the paper changes substantively and this reference is
  no longer meaningful, replace it deliberately and update this README.

## Canonical build

The canonical PDF is produced by

```bash
make paper
```

which runs `latexmk -pdf -outdir=build` inside `paper/` and writes to
**`paper/build/dc_paper.pdf`**. Any CI or downstream consumer should pull
from there, not from this reference copy.
