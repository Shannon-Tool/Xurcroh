# Xurcroh Fortress research paper

`xurcroh-fortress.pdf` is the ResearchGate publication artifact. The manuscript is written by Felix Devon (SHANNON LAB LLC) and uses only the repository's public architecture, model matrix, experiments, and hardware-validation record.

**DOI:** [10.13140/RG.2.2.15046.31042](https://doi.org/10.13140/RG.2.2.15046.31042) · [ResearchGate publication](https://www.researchgate.net/publication/418452236_Xurcroh_Fortress_-_Verifiable_Multimodal_Inference_over_Distributed_Non-Colluding_GPU_Workers) · [BibTeX](citation.bib)

## Cite

Devon, Felix. “Xurcroh Fortress - Verifiable Multimodal Inference over Distributed, Non-Colluding GPU Workers.” SHANNON LAB LLC, November 2026. https://doi.org/10.13140/RG.2.2.15046.31042.

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
