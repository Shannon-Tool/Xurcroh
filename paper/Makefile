.PHONY: all figures clean

all:
	cd .. && ./scripts/build-paper.sh

figures:
	../.venv/bin/python scripts/generate_figures.py

clean:
	latexmk -C -jobname=xurcroh-fortress main.tex
