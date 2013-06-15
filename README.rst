Code Accompanying Paper XXXX
============================

.. todo:: Update paper title when it is finalized.

This repository contains the source code and source LaTeX for the paper XXXX
submitted on DATE.

measure_tau.py
--------------
Using the measured optical depth ratio and Mach number from the Formaldehyde
and CO lines, this module uses `pymc <pymc-devs.github.io>`_ to fit the values
of sigma and the other controlling parameters of the distribution.

turbulent_pdfs.py
-----------------
Defines a variety of probability distribution functions (lognormal and variants
thereupon)

hopkins_pdf.py
--------------
A python implementation of the `Hopkins 2013
<http://adsabs.harvard.edu/abs/2013MNRAS.430.1880H>`_ PDF.  See `this blog post
<http://keflavich.github.io/blog/hopkins-pdf-generalization.html>`__ for a
detailed description.  Note that this code generalizes the Hopkins distribution
to :math:`\rho_0 \ne 1`.
