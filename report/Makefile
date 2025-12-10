# Makefile for LaTeX report compilation

MAIN = main
SLIDES = slides
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex neurips_2024.sty
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex
	@echo "PDF compiled successfully: $(MAIN).pdf"

# Full compilation with bibliography
full: $(MAIN).tex neurips_2024.sty references.bib
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

slides: $(SLIDES).tex neurips_2024.sty
	$(LATEX) $(SLIDES).tex
	$(LATEX) $(SLIDES).tex
	@echo "Slides compiled successfully: $(SLIDES).pdf"

view: $(MAIN).pdf
	open $(MAIN).pdf  # macOS
	# xdg-open $(MAIN).pdf  # Linux
	# start $(MAIN).pdf  # Windows

# Overleaf-compatible: just compile twice
overleaf: $(MAIN).tex
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex
