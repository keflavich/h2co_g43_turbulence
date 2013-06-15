# latex Makefile
texpath=/usr/texbin/
PDFLATEX=${texpath}pdflatex -halt-on-error -synctex=1 --interaction=nonstopmode
LATEX=${PDFLATEX}
BIBTEX=bibtex
DVIPS=dvips
PS2PDF=ps2pdf

all: h2co_turbulence_letter

h2co_turbulence_letter: 
	@rm -f h2co_turbulence_letter*.aux h2co_turbulence_letter*.bbl h2co_turbulence_letter*.blg h2co_turbulence_letter*.dvi h2co_turbulence_letter*.log h2co_turbulence_letter*.lot h2co_turbulence_letter*.lof
	${LATEX} h2co_turbulence_letter.tex
	${BIBTEX} h2co_turbulence_letter
	${LATEX} h2co_turbulence_letter.tex
	${BIBTEX} h2co_turbulence_letter
	${LATEX} h2co_turbulence_letter.tex
	cp h2co_turbulence_letter.pdf h2co_turbulence_`date +%Y%m%d`.pdf
