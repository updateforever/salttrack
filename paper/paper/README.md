# SALTTrack Paper

This folder contains the IEEE TIP manuscript source.

- `main.tex`: main manuscript entry.
- `sections/`: section files.
- `figures/`: paper figures.
- `tables/`: optional external table files.
- `refs.bib`: bibliography.
- `ieee_template/`: original IEEE template files kept for reference.

Compile from this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
