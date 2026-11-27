# Xurcroh Fortress research paper

`xurcroh-fortress.pdf` is the publication artifact for ResearchGate. The manuscript is written by Felix Devon (SHANNON LAB LLC) and uses only the repository's public architecture, model matrix, experiments, and hardware-validation record.

## Build

From the repository root:

```bash
./scripts/build-paper.sh
```

The script requires Python 3, `latexmk`, `pdflatex`, and `biber`. On Debian/Ubuntu, the required packages are:

```bash
apt-get install --no-install-recommends \
  latexmk texlive-latex-recommended texlive-latex-extra \
  texlive-pictures texlive-science texlive-bibtex-extra biber
```

The build regenerates PGFPlots sources from `paper/data/`, runs BibLaTeX/Biber and LaTeX to convergence, and writes `paper/xurcroh-fortress.pdf`. Generated `.tex` plots are committed so readers can inspect them without running Python.

## Evidence scope

All physical-GPU results are historical measurements for `3b48e2a + patches-01…23`, frozen content hash `7f31c2e0`, as published in `docs/xurcroh-validation-report.md`. They are not measurements independently rerun by this paper build, current cloud price quotes, or guarantees for later source revisions. The manuscript explicitly retains the report's security, external-assurance, and model-license limitations.
