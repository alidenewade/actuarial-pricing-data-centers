# paper/reference/

This directory holds the **baseline PDF build** of `dc_paper.tex`, retained as
a reproducibility reference.

## `dc_paper_original.pdf`

- The known-good build delivered with the **2026-05-18 cooling-section
  rewrite**: 456 988 bytes, 42 pages, SHA-256
  `e5d608e47f76b7680bc3a02a907fa41100bf02c017f26508bc91f206abc72a04`.
- Compiled on the laptop and copied across with the matching
  `dc_paper.tex` (74 897 B) and `cooling_section.tex` (29 731 B). Use it as
  the byte-for-byte reference for the next `make paper` on adu-00 (compare
  page count, ToC alignment, and figure placement).
- The previous baseline (412 365 B, SHA `20676bf9afc8…`) was the pre-rewrite
  build with the placeholder `cooling_section.tex`. It has been superseded;
  see `git log -- paper/reference/dc_paper_original.pdf` to retrieve it if
  ever needed.
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
