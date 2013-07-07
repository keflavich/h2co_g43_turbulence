"""
Smooth the LVG models with distributions to get tau, then fit it with
optimization procedures.
"""

import numpy as np
import hopkins_pdf
import turbulent_pdfs
from turbulent_pdfs import lognormal_massweighted

try:
    from agpy import readcol
    radtab = readcol('radex_data/1-1_2-2_XH2CO=1e-9_troscompt.dat',asRecArray=True)
except ImportError:
    import astropy.table
    radtab = astropy.table.read('radex_data/1-1_2-2_XH2CO=1e-9_troscompt.dat',format='ascii')

# plotting stuff
import pylab as pl
pl.rc('font',size=20)

_datacache = {}

def select_data(abundance=-9.0, opr=1, temperature=20, tolerance=0.1):
    key = (abundance, opr, temperature, tolerance)
    if key in _datacache:
        return _datacache[key]
    else:
        #tolerance = {-10:0.1, -9.5: 0.3, -9: 0.1, -8:0.1, -8.5: 0.3}[abundance]
        OKtem = radtab['Temperature'] == temperature
        OKopr = radtab['opr'] == opr
        OKabund = np.abs((radtab['log10col'] - radtab['log10dens'] - np.log10(3.08e18)) - abundance) < tolerance
        OK = OKtem * OKopr * OKabund

        tau1x = radtab['TauLow'][OK]
        tau2x = radtab['TauUpp'][OK]
        dens = radtab['log10dens'][OK]
        col = radtab['log10col'][OK]

        _datacache[key] = tau1x,tau2x,dens,col

        return tau1x,tau2x,dens,col


def generate_tau_functions(**kwargs):
    tau1x,tau2x,dens,col = select_data(**kwargs)

    # ddens = (10**dens[1]-10**dens[0])
    #dlogdens = (dens[1]-dens[0])
    #dlndens = dlogdens * np.log(10)

    def tau(meandens, line=tau1x, sigma=1.0, hightail=False,
            hopkins=False, powertail=False, lowtail=False,
            compressive=False, divide_by_col=False, **kwargs):
        if compressive:
            distr = turbulent_pdfs.compressive_distr(meandens,sigma,**kwargs)
        elif lowtail:
            distr = turbulent_pdfs.lowtail_distr(meandens,sigma,**kwargs)
        elif powertail or hightail:
            distr = turbulent_pdfs.hightail_distr(meandens,sigma,**kwargs)
        elif hopkins:
            T = hopkins_pdf.T_of_sigma(sigma, logform=True)
            #distr = 10**dens * hopkins_pdf.hopkins(10**(dens), meanrho=10**(meandens), sigma=sigma, T=T) # T~0.05 M_C
            distr = hopkins_pdf.hopkins_masspdf_ofmeandens(10**dens, 10**meandens, sigma_volume=sigma, T=T, normalize=True)
            # Hopkins is integral-normalized, not sum-normalized
            #distr /= distr.sum()
        else:
            #distr = lognormal(10**dens, 10**meandens, sigma) * dlndens
            distr = lognormal_massweighted(10**dens, 10**meandens, sigma, normalize=True)
        if divide_by_col:
            return (distr*line/(10**col)).sum()
        else:
            return (distr*line).sum()

    def vtau(meandens,**kwargs):
        """ vectorized tau """
        if hasattr(meandens,'size') and meandens.size == 1:
            return tau(meandens, **kwargs)
        taumean = np.array([tau(x,**kwargs) for x in meandens])
        return taumean

    def vtau_ratio(meandens, line1=tau1x, line2=tau2x, **kwargs):
        t1 = vtau(meandens, line=line1, **kwargs)
        t2 = vtau(meandens, line=line2, **kwargs)
        return t1/t2

    return tau,vtau,vtau_ratio

def generate_simpletools(**kwargs):
    tau,vtau,vtau_ratio = generate_tau_functions(**kwargs)

    def tauratio(meandens, sigma, **kwargs):
        return vtau_ratio(np.log10(meandens), sigma=sigma, **kwargs)

    def tauratio_hopkins(meandens, sigma, **kwargs):
        return vtau_ratio(np.log10(meandens), sigma=sigma, hopkins=True, **kwargs)

    def tau(meandens, sigma, line, **kwargs):
        return vtau(np.log10(meandens), sigma=sigma, line=line, **kwargs)

    def tau_hopkins(meandens, sigma, line, **kwargs):
        return vtau(np.log10(meandens), sigma=sigma, hopkins=True, line=line, **kwargs)

    return tauratio,tauratio_hopkins,tau,tau_hopkins
