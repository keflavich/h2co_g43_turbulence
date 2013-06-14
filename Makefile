# latex Makefile
texpath=/usr/texbin/
PDFLATEX=${texpath}pdflatex -halt-on-error -synctex=1 --interaction=nonstopmode
LATEX=${PDFLATEX}
BIBTEX=bibtex
DVIPS=dvips
PS2PDF=ps2pdf

all: h2co_lowdens

h2co_lowdens: 
	@rm -f h2co_lowdens*.aux h2co_lowdens*.bbl h2co_lowdens*.blg h2co_lowdens*.dvi h2co_lowdens*.log h2co_lowdens*.lot h2co_lowdens*.lof
	${LATEX} h2co_lowdens.tex
	${BIBTEX} h2co_lowdens
	${LATEX} h2co_lowdens.tex
	${BIBTEX} h2co_lowdens
	${LATEX} h2co_lowdens.tex
	cp h2co_lowdens.pdf h2co_turbulence_`date +%Y%m%d`.pdf
