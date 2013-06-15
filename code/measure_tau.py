"""
Smooth the LVG models with distributions to get tau, then fit it with
optimization procedures.
"""

import numpy as np
import hopkins_pdf
import turbulent_pdfs
from turbulent_pdfs import lognormal

try:
    from agpy import readcol
    radtab = readcol('radex_data/1-1_2-2_XH2CO=1e-9_troscompt.dat',asRecArray=True)
except ImportError:
    import astropy.table
    radtab = astropy.table.read('radex_data/1-1_2-2_XH2CO=1e-9_troscompt.dat',format='ascii')

# plotting stuff
import pylab as pl
pl.rc('font',size=20)


def select_data(abundance=-9, opr=1, temperature=20, tolerance=0.1):
    OKtem = radtab['Temperature'] == temperature
    OKopr = radtab['opr'] == opr
    OKabund = np.abs((radtab['log10col'] - radtab['log10dens'] - np.log10(3.08e18)) - abundance) < tolerance
    OK = OKtem * OKopr * OKabund

    tau1x = radtab['TauLow'][OK]
    tau2x = radtab['TauUpp'][OK]
    dens = radtab['log10dens'][OK]
    col = radtab['log10col'][OK]

    return tau1x,tau2x,dens,col

def generate_tau_functions(**kwargs):
    tau1x,tau2x,dens,col = select_data(**kwargs)

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
            distr = 10**dens * hopkins_pdf.hopkins(10**(dens), meanrho=10**(meandens), sigma=sigma, T=T) # T~0.05 M_C
            # Hopkins is integral-normalized, not sum-normalized
            distr /= distr.sum()
        else:
            distr = lognormal(10**dens, 10**meandens, sigma)
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

if __name__ == "__main__":
    savepath = "/Users/adam/work/h2co/lowdens/figures/"
    # load defaults by default
    tau,vtau,vtau_ratio = generate_tau_functions()

    def tauratio(meandens, sigma):
        return vtau_ratio(np.log10(meandens), sigma=sigma)

    def tauratio_hopkins(meandens, sigma):
        return vtau_ratio(np.log10(meandens), sigma=sigma, hopkins=True)

    import pymc
    from agpy import pymc_plotting
    import itertools

    dolognormal=True
    dohopkins=True
    do_paperfigure=True

    if dolognormal:
        d = {}
        # fit values for GSRMC 43.30
        # meandens is "observed" but we want to trace it and let it vary...
        # with meandens observed=False,  'sigma': {'95% HPD interval': array([ 2.89972948,  3.69028675]),
        d['meandens'] = pymc.Uniform(name='meandens',lower=8,upper=150,value=15, observed=False)
        d['sigma'] = pymc.Uniform(name='sigma',lower=0,upper=25)
        d['tauratio_mu'] = pymc.Deterministic(name='tauratio_mu', eval=tauratio, parents={'meandens':d['meandens'],'sigma':d['sigma']}, doc='tauratio')
        # the observed values.  f=tau ratio = 6.65.  tau might be too high, but the "best fits" were tau=9 before, which is just not possible
        d['tauratio'] = pymc.Normal(name='tauratio',mu=d['tauratio_mu'],tau=4,value=6.65,observed=True)
        mc_simple = pymc.MCMC(d)
        mc_simple.sample(100000)

        graph_lognormal_simple = pymc.graph.graph(mc_simple)
        graph_lognormal_simple.write_pdf(savepath+"mc_lognormal_simple_graph.pdf")
        graph_lognormal_simple.write_png(savepath+"mc_lognormal_simple_graph.png")

        print "WARNING: This [mc_complicated] fails because apparently there is no model that simultaneously fits the tau ratio and the mach number."
        d['temperature'] = temperature = pymc.Uniform(name='temperature',lower=8,upper=15)
        def cs(temperature):
            return np.sqrt(1.38e-16 * temperature / (2.3*1.67e-24)) /1e5
        d['c_s'] = pymc.Deterministic(name='c_s', eval=cs, parents={'temperature':temperature}, doc='soundspeed')
        d['b'] = pymc.Uniform('b',0.3,1.0)
        def mach(sigma, c_s, b):
            # sigma**2 = np.log(1+b**2 * Mach**2 * beta/(beta+1))
            # mach = sqrt[(exp(sigma**2) -1)/b**2]
            return np.sqrt((np.exp(sigma**2) - 1)/b**2)
        d['mach_mu'] = pymc.Deterministic(name='mach_mu', eval=mach, parents={'b':d['b'], 'sigma':d['sigma'], 'c_s':d['c_s']}, doc="Mach number")
        # mach number could be as low as 3.7, as high as say 6.8, and the distribution between isn't super obvious...
        #d['mach'] = pymc.Normal(name='MachNumber',mu=d['mach_mu'],tau=16, value=6.6, observed=True)
        # try a much less restrictive mach number range...
        d['mach'] = pymc.Normal(name='MachNumber',mu=d['mach_mu'],tau=0.5, value=5.1, observed=True)
        mc_lognormal = pymc.MCMC(d)
        mc_lognormal.sample(100000)

        graph_lognormal = pymc.graph.graph(mc_lognormal)
        graph_lognormal.write_pdf(savepath+"mc_lognormal_graph.pdf")
        graph_lognormal.write_png(savepath+"mc_lognormal_graph.png")

        varslice=(10000,None,None)
        for fignum,(p1,p2) in enumerate(itertools.combinations(('tauratio_mu','sigma','meandens','temperature','b','mach_mu'),2)):
            pymc_plotting.hist2d(mc_lognormal, p1, p2, bins=30, clear=True, fignum=fignum, varslice=varslice, colorbar=True)
            pl.title("Lognormal with Mach")
            pl.savefig(savepath+"LognormalWithMach_%s_v_%s_mcmc.png" % (p1,p2))

        for fignum2,(p1,p2) in enumerate(itertools.combinations(('tauratio_mu','sigma','meandens'),2)):
            pymc_plotting.hist2d(mc_simple, p1, p2, bins=30, clear=True, fignum=fignum+fignum2+1, varslice=varslice, colorbar=True)
            pl.title("Lognormal - just tauratio")
            pl.savefig(savepath+"LognormalJustTau_%s_v_%s_mcmc.png" % (p1,p2))

    if dohopkins:
        d = {}
        d['meandens'] = pymc.Uniform(name='meandens',lower=8,upper=150,value=15, observed=False)
        d['sigma'] = pymc.Uniform(name='sigma',lower=0,upper=25)
        d['tauratio_mu'] = pymc.Deterministic(name='tauratio_mu', eval=tauratio_hopkins, parents=d, doc='tauratio_hopkins')
        d['tauratio'] = pymc.Normal(name='tauratio',mu=d['tauratio_mu'],tau=4,value=6.65,observed=True)
        mc_hopkins_simple = pymc.MCMC(d)
        mc_hopkins_simple.sample(1e5)

        graph_hopkins_simple = pymc.graph.graph(mc_hopkins_simple)
        graph_hopkins_simple.write_pdf(savepath+"mc_hopkins_simple_graph.pdf")
        graph_hopkins_simple.write_png(savepath+"mc_hopkins_simple_graph.png")

        def Tval(sigma):
            return hopkins_pdf.T_of_sigma(sigma, logform=True)
        d['Tval'] = pymc.Deterministic(name='Tval', eval=Tval, parents={'sigma':d['sigma']}, doc='Intermittency parameter T')
        d['b'] = pymc.Uniform('b',0.3,1.0)
        def mach(Tval, b):
            return 20*Tval / b
        d['mach_mu'] = pymc.Deterministic(name='mach_mu', eval=mach, parents={'Tval':d['Tval'], 'b': d['b']}, doc='Mach Number')
        d['mach'] = pymc.Normal(name='MachNumber',mu=d['mach_mu'], tau=0.5, value=5.1, observed=True)

        mc_hopkins = pymc.MCMC(d)
        mc_hopkins.sample(1e5)

        graph_hopkins = pymc.graph.graph(mc_hopkins)
        graph_hopkins.write_pdf(savepath+"mc_hopkins_graph.pdf")
        graph_hopkins.write_png(savepath+"mc_hopkins_graph.png")

        varslice=(1000,None,None)

        for fignum,(p1,p2) in enumerate(itertools.combinations(('tauratio_mu','sigma','meandens','Tval','b','mach_mu'),2)):
            pymc_plotting.hist2d(mc_hopkins, p1, p2, bins=30, clear=True, fignum=fignum, varslice=varslice, colorbar=True)
            pl.title("Hopkins with Mach")
            pl.savefig(savepath+"HopkinsWithMach_%s_v_%s_mcmc.png" % (p1,p2))

        for fignum2,(p1,p2) in enumerate(itertools.combinations(('tauratio_mu','sigma','meandens'),2)):
            pymc_plotting.hist2d(mc_hopkins_simple, p1, p2, bins=30, clear=True, fignum=fignum+fignum2+1, varslice=varslice, colorbar=True)
            pl.title("Hopkins - just tauratio")
            pl.savefig(savepath+"HopkinsJustTau_%s_v_%s_mcmc.png" % (p1,p2))

        print "Some statistics used in the paper: "
        print 'mc_lognormal_simple sigma: ',mc_simple.stats()['sigma']['quantiles']
        print 'mc_lognormal        sigma: ',mc_lognormal.stats()['sigma']['quantiles']
        print 'mc_hopkins_simple sigma: ',mc_hopkins_simple.stats()['sigma']['quantiles']
        print 'mc_hopkins        sigma: ',mc_hopkins.stats()['sigma']['quantiles']
        print 'mc_hopkins        Tval: ',mc_hopkins.stats()['Tval']['quantiles']
        print 'mc_hopkins        b: ',mc_hopkins.stats(quantiles=(0.1,1,2.5,5,50))['b']['quantiles']
        print 'mc_hopkins        m: ',mc_hopkins.stats()['mach_mu']['quantiles']

    if do_paperfigure:
        abundance = -9
        opr = 1
        tau1x,tau2x,dens,col = select_data(abundance=abundance, opr=opr, temperature=20, tolerance=0.1, extrapolate=False, mindens=-2, maxdens=9, )

        pl.figure(30)
        pl.clf()
        ax = pl.gca()
        ax.plot(dens,tau1x/tau2x,'k',linewidth=3.0,label='LVG',alpha=0.75)

        logmeandens = np.linspace(-2,7,300)
        meandens = 10**logmeandens

        stylecycle = itertools.cycle(('-','-.','--',':'))
        dashcycle = itertools.cycle(((None,None),(20,10),(20,20),(10,10),(20,10,40,10)))

        for sigma in np.arange(0.5,4.0,1):
            ax.plot(logmeandens,tauratio(meandens,sigma=sigma),color='k',linewidth=2, alpha=0.5,  label='$\\sigma_s=%0.1f$' % sigma, dashes=dashcycle.next())
        for sigma in np.arange(0.5,4.0,1):
            ax.plot(logmeandens,tauratio_hopkins(meandens,sigma=sigma),color='orange', label='$\\sigma_s=%0.1f$ Hopkins' % sigma, linewidth=2, alpha=0.5, dashes=dashcycle.next())

        ax.legend(loc='best',prop={'size':18})
        ax.axis([-1,7,0,15])
        ax.set_xlabel('$\\log_{10}$($n($H$_2)$ [cm$^{-3}$])',fontsize=24)
        ax.set_ylabel('$\\tau_{1-1}/\\tau_{2-2}$',fontsize=24)
        ax.figure.savefig(savepath+'lognormalsmooth_density_ratio_massweight_withhopkins_logopr%0.1f_abund%i.png' % (np.log10(opr),abundance),bbox_inches='tight')

        ax.errorbar([np.log10(15)],[6.65],xerr=np.array([[0.33,1]]).T,yerr=np.array([[0.5,0.5]]).T,
                    label="G43.16-0.03", color=(0,0,1,0.5), alpha=0.5, marker='o',
                    linewidth=2)
        ax.figure.savefig(savepath+'lognormalsmooth_density_ratio_massweight_withhopkins_logopr%0.1f_abund%i_withG43.png' % (np.log10(opr),abundance),bbox_inches='tight')
