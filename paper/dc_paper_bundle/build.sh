#!/usr/bin/env bash
# Rebuild dc_paper.pdf from the bundled .tex sources.
# Requires TeX Live 2023+ with texlive-latex-extra, texlive-fonts-extra,
# texlive-pictures, texlive-science, texlive-publishers,
# texlive-extra-utils, latexmk, biber, and poppler-utils.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p build

latexmk -pdf -interaction=nonstopmode -file-line-error -synctex=1 \
  -shell-escape -outdir=build dc_paper.tex

# Copy the freshly-built PDF up next to the sources for convenience.
cp build/dc_paper.pdf ./dc_paper.pdf

echo
echo "[ok] dc_paper.pdf built ($(pdfinfo build/dc_paper.pdf | awk '/^Pages:/{print $2}') pages, $(stat -c %s build/dc_paper.pdf) bytes)"
