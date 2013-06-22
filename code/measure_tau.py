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

import astropy.io.fits as pyfits

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

def generate_simpletools(**kwargs):
    tau,vtau,vtau_ratio = generate_tau_functions(**kwargs)

    def tauratio(meandens, sigma):
        return vtau_ratio(np.log10(meandens), sigma=sigma)

    def tauratio_hopkins(meandens, sigma):
        return vtau_ratio(np.log10(meandens), sigma=sigma, hopkins=True)

    def tau(meandens, sigma, line):
        return vtau(np.log10(meandens), sigma=sigma, line=line)

    def tau_hopkins(meandens, sigma, line):
        return vtau(np.log10(meandens), sigma=sigma, hopkins=True, line=line)

    return tauratio,tauratio_hopkins,tau,tau_hopkins

# pymc tool
def save_traces(mc, filename, clobber=False):
    keys = [v.__name__ for v in mc.variables if hasattr(v,'observed') and not v.observed]
    traces = {k:np.concatenate([v.squeeze() for v in mc.trace(k)._trace.values()])
            for k in keys}
    shape = len(traces[keys[0]])
    arr = np.empty(shape, dtype=[(k,np.float) for k in keys])
    for k in keys:
        arr[k] = traces[k]
    hdu = pyfits.BinTableHDU(arr)
    hdu.writeto(filename, clobber=clobber)
    return traces

if __name__ == "__main__":
    savepath = "/Users/adam/work/h2co/lowdens/figures/"
    # load defaults by default
    abundance = -9

    tauratio,tauratio_hopkins,tau,tau_hopkins = generate_simpletools(abundance=abundance)
    tau1x,tau2x,dens,col = select_data(abundance=abundance)
    def tauoneone(meandens, sigma):
        return tau(meandens, sigma, line=tau1x)
    def tautwotwo(meandens, sigma):
        return tau(meandens, sigma, line=tau2x)
    def tauoneone_hopkins(meandens, sigma):
        return tau_hopkins(meandens, sigma, line=tau1x)
    def tautwotwo_hopkins(meandens, sigma):
        return tau_hopkins(meandens, sigma, line=tau2x)

    import pymc
    from agpy import pymc_plotting
    import itertools
    import pymc_tools

    dolognormal=True
    dohopkins=True
    do_paperfigure=True
    do_tables=True

    if dolognormal:
        d = {}
        # fit values for GSRMC 43.30
        # meandens is "observed" but we want to trace it and let it vary...
        # with meandens observed=False,  'sigma': {'95% HPD interval': array([ 2.89972948,  3.69028675]),
        d['meandens'] = pymc.Uniform(name='meandens',lower=8,upper=150,value=15, observed=False)
        d['sigma'] = pymc.Uniform(name='sigma',lower=0,upper=25)
        # the observed values.  f=tau ratio = 6.65.  tau might be too high, but the "best fits" were tau=9 before, which is just not possible
        d['tauoneone_mu'] = pymc.Deterministic(name='tauoneone_mu', eval=tauoneone, parents={'meandens':d['meandens'],'sigma':d['sigma']}, doc='tauoneone')
        d['tauoneone'] = pymc.Normal(name='tauoneone',mu=d['tauoneone_mu'],tau=1./0.001165**2,value=0.1133,observed=True)
        d['tautwotwo_mu'] = pymc.Deterministic(name='tautwotwo_mu', eval=tautwotwo, parents={'meandens':d['meandens'],'sigma':d['sigma']}, doc='tautwotwo')
        d['tautwotwo'] = pymc.Normal(name='tautwotwo',mu=d['tautwotwo_mu'],tau=1./0.000525**2,value=0.01623,observed=True)
        @pymc.deterministic(trace=True,plot=True)
        def tau_ratio(oneone=d['tauoneone_mu'], twotwo=d['tau_twotwo_mu']):
            return oneone/twotwo
        d['tau_ratio'] = tau_ratio
        mc_simple = pymc.MCMC(d)
        mc_simple.sample(100000)

        graph_lognormal_simple = pymc.graph.graph(mc_simple)
        graph_lognormal_simple.write_pdf(savepath+"mc_lognormal_simple_graph.pdf")
        graph_lognormal_simple.write_png(savepath+"mc_lognormal_simple_graph.png")

        d['b'] = pymc.Uniform(name='b', value=0.5, lower=0.3, upper=1, observed=False)
        @pymc.deterministic(plot=True,trace=True)
        def mach(sigma=d['sigma'], b=d['b']):
            return np.sqrt((np.exp(sigma**2) - 1)/b**2)
        
        d['mach'] = mach
        d['mach_observed'] = pymc.Normal(name='mach_observed', mu=mach, tau=1./0.2**2, value=5.1, observed=True)

        mc_lognormal = pymc.MCMC(d)
        mc_lognormal.sample(100000)

        graph_lognormal = pymc.graph.graph(mc_lognormal)
        graph_lognormal.write_pdf(savepath+"mc_lognormal_graph.pdf")
        graph_lognormal.write_png(savepath+"mc_lognormal_graph.png")

        varslice=(10000,None,None)
        for fignum,(p1,p2) in enumerate(itertools.combinations(('tauoneone_mu','tautwotwo_mu','sigma','meandens','b','mach'),2)):
            pymc_plotting.hist2d(mc_lognormal, p1, p2, bins=30, clear=True, fignum=fignum, varslice=varslice, colorbar=True)
            pl.title("Lognormal with Mach")
            pl.savefig(savepath+"LognormalWithMach_%s_v_%s_mcmc.png" % (p1,p2))

        for fignum2,(p1,p2) in enumerate(itertools.combinations(('tauoneone_mu','tautwotwo_mu','sigma','meandens'),2)):
            pymc_plotting.hist2d(mc_simple, p1, p2, bins=30, clear=True, fignum=fignum+fignum2+1, varslice=varslice, colorbar=True)
            pl.title("Lognormal - just $\\tau$ fits")
            pl.savefig(savepath+"LognormalJustTau_%s_v_%s_mcmc.png" % (p1,p2))

        lognormal_statstable = pymc_tools.stats_table(mc_lognormal)
        lognormal_statstable.write('lognormal_statstable_abundance%s.fits' % abundance, overwrite=True)
        lognormal_simple_statstable = pymc_tools.stats_table(mc_simple)
        lognormal_statstable.write('lognormal_simple_statstable_abundance%s.fits' % abundance, overwrite=True)

        mc_lognormal_traces = save_traces(mc_lognormal, "mc_lognormal_traces", clobber=True)
        mc_lognormal_simple_traces = save_traces(mc_simple, "mc_lognormal_simple_traces", clobber=True)

        pl.figure(33); pl.clf()
        pl.title("Lognormal")
        pymc_plotting.plot_mc_hist(mc_lognormal,'b',lolim=True,alpha=0.5,bins=25,legloc='lower right')
        pl.xlabel('$b$')
        pl.savefig('LognormalWithMach_b_1D_restrictions.png')

    if dohopkins:
        d = {}
        d['meandens'] = pymc.Uniform(name='meandens',lower=8,upper=150,value=15, observed=False)
        d['sigma'] = pymc.Uniform(name='sigma',lower=0,upper=25)
        parents = {k:d[k] for k in ('meandens','sigma')}
        d['tauoneone_mu'] = pymc.Deterministic(name='tauoneone_mu', eval=tauoneone_hopkins, parents=parents, doc='tauoneone_hopkins')
        d['tauoneone'] = pymc.Normal(name='tauoneone',mu=d['tauoneone_mu'],tau=1./0.001165**2,value=0.1133,observed=True)
        d['tautwotwo_mu'] = pymc.Deterministic(name='tautwotwo_mu', eval=tautwotwo_hopkins, parents=parents, doc='tautwotwo_hopkins')
        d['tautwotwo'] = pymc.Normal(name='tautwotwo',mu=d['tautwotwo_mu'],tau=1./0.000525**2,value=0.01623,observed=True)
        @pymc.deterministic(trace=True,plot=True)
        def tau_ratio(oneone=d['tauoneone_mu'], twotwo=d['tau_twotwo_mu']):
            return oneone/twotwo
        d['tau_ratio'] = tau_ratio
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

        for fignum,(p1,p2) in enumerate(itertools.combinations(('tauonone_mu','tautwotwo_mu','sigma','meandens','Tval','b','mach_mu'),2)):
            pymc_plotting.hist2d(mc_hopkins, p1, p2, bins=30, clear=True, fignum=fignum, varslice=varslice, colorbar=True)
            pl.title("Hopkins with Mach")
            pl.savefig(savepath+"HopkinsWithMach_%s_v_%s_mcmc.png" % (p1,p2))

        for fignum2,(p1,p2) in enumerate(itertools.combinations(('tauonone_mu','tautwotwo_mu','sigma','meandens'),2)):
            pymc_plotting.hist2d(mc_hopkins_simple, p1, p2, bins=30, clear=True, fignum=fignum+fignum2+1, varslice=varslice, colorbar=True)
            pl.title("Hopkins - just $\\tau$ fits")
            pl.savefig(savepath+"HopkinsJustTau_%s_v_%s_mcmc.png" % (p1,p2))

        print "Some statistics used in the paper: "
        print 'mc_lognormal_simple sigma: ',mc_simple.stats()['sigma']['quantiles']
        print 'mc_lognormal        sigma: ',mc_lognormal.stats()['sigma']['quantiles']
        print 'mc_lognormal        b: ',mc_lognormal.stats(quantiles=(0.1,1,2.5,5,50))['b']['quantiles']
        print 'mc_hopkins_simple sigma: ',mc_hopkins_simple.stats()['sigma']['quantiles']
        print 'mc_hopkins        sigma: ',mc_hopkins.stats()['sigma']['quantiles']
        print 'mc_hopkins        Tval: ',mc_hopkins.stats()['Tval']['quantiles']
        print 'mc_hopkins        b: ',mc_hopkins.stats(quantiles=(0.1,1,2.5,5,50))['b']['quantiles']
        print 'mc_hopkins        m: ',mc_hopkins.stats()['mach_mu']['quantiles']

        hopkins_statstable = pymc_tools.stats_table(mc_hopkins)
        hopkins_statstable.write('hopkins_statstable_abundance%s.fits' % abundance, overwrite=True)
        hopkins_simple_statstable = pymc_tools.stats_table(mc_hopkins_simple)
        hopkins_statstable.write('hopkins_simple_statstable_abundance%s.fits' % abundance, overwrite=True)

        save_traces(mc_hopkins, "mc_hopkins_traces", clobber=True)
        save_traces(mc_hopkins_simple, "mc_hopkins_simple_traces", clobber=True)

        pl.figure(32); pl.clf()
        pl.title("Hopkins")
        pymc_plotting.plot_mc_hist(mc_hopkins,'b',lolim=True,alpha=0.5,bins=25,legloc='lower right')

    if do_paperfigure:
        for abundance in [-9,-8.5,-8.25,-8.0]:
            opr = 1
            temperature = 20
            tau1x,tau2x,dens,col = select_data(abundance=abundance, opr=opr, temperature=temperature)
            tauratio,tauratio_hopkins,tau,tau_hopkins = generate_simpletools(abundance=abundance, opr=opr, temperature=temperature)
            print len(tau1x)

            pl.figure(30)
            pl.clf()
            ax = pl.gca()
            ax.plot(dens,tau1x/tau2x,'k',linewidth=3.0,label=r'Dirac $\delta$',alpha=0.75)

            logmeandens = np.linspace(-2,7,300)
            meandens = 10**logmeandens

            stylecycle = itertools.cycle(('-','-.','--',':'))
            dashcycle = itertools.cycle(((None,None),(6,2),(10,4),(2,2),(5,5)))

            for sigma in np.arange(0.5,4.0,1):
                ax.plot(logmeandens,tauratio(meandens,sigma=sigma),color='k',linewidth=2, alpha=0.5,  label='$\\sigma_s=%0.1f$' % sigma, dashes=dashcycle.next())

            dashcycle = itertools.cycle(((None,None),(6,2),(10,4),(2,2),(5,5)))
            for sigma in np.arange(0.5,4.0,1):
                ax.plot(logmeandens,tauratio_hopkins(meandens,sigma=sigma),color='orange', label='$\\sigma_s=%0.1f$ Hopkins' % sigma, linewidth=2, alpha=0.5, dashes=dashcycle.next())

            ax.legend(loc='best',prop={'size':18})
            ax.axis([-1,7,0,15])
            ax.set_xlabel('$\\log_{10}$($n($H$_2)$ [cm$^{-3}$])',fontsize=24)
            ax.set_ylabel('$\\tau_{1-1}/\\tau_{2-2}$',fontsize=24)
            ax.figure.savefig(savepath+'lognormalsmooth_density_ratio_massweight_withhopkins_logopr%0.1f_abund%s.png' % (np.log10(opr),str(abundance)),bbox_inches='tight')

            dot,caps,bars = ax.errorbar([np.log10(15)],[6.98],xerr=np.array([[0.33,1]]).T,yerr=np.array([[0.36,0.36]]).T,
                        label="G43.16-0.03", color=(0,0,1,0.5), alpha=0.5, marker='o',
                        linewidth=2)
            caps[0].set_marker('$($')
            caps[1].set_marker('$)$')
            caps[0].set_color((1,0,0,0.6))
            caps[1].set_color((1,0,0,0.6))
            bars[0].set_color((1,0,0,0.6))
            ax.figure.savefig(savepath+'lognormalsmooth_density_ratio_massweight_withhopkins_logopr%0.1f_abund%s_withG43.png' % (np.log10(opr),str(abundance)),bbox_inches='tight')

            for ii,(line,taux) in enumerate(zip(('oneone','twotwo'),(tau1x,tau2x))):
                pl.figure(31+ii)
                pl.clf()
                ax = pl.gca()
                ax.semilogy(dens,tau1x,'k',linewidth=3.0,label=r'Dirac $\delta$',alpha=0.75)

                logmeandens = np.linspace(-2,7,300)
                meandens = 10**logmeandens

                stylecycle = itertools.cycle(('-','-.','--',':'))
                dashcycle = itertools.cycle(((None,None),(6,2),(10,4),(2,2),(5,5)))

                for sigma in np.arange(0.5,4.0,1):
                    ax.semilogy(logmeandens,tau(meandens,sigma=sigma,line=taux),color='k',linewidth=2, alpha=0.5,  label='$\\sigma_s=%0.1f$' % sigma, dashes=dashcycle.next())

                dashcycle = itertools.cycle(((None,None),(6,2),(10,4),(2,2),(5,5)))
                for sigma in np.arange(0.5,4.0,1):
                    ax.semilogy(logmeandens,tau_hopkins(meandens,sigma=sigma,line=taux),color='orange', label='$\\sigma_s=%0.1f$ Hopkins' % sigma, linewidth=2, alpha=0.5, dashes=dashcycle.next())

                ax.legend(loc='best',prop={'size':18})
                ax.axis([-2,7,1e-3,10])
                ax.set_xlabel('$\\log_{10}$($n($H$_2)$ [cm$^{-3}$])',fontsize=24)
                ax.set_ylabel('$\\tau_{1-1}/\\tau_{2-2}$',fontsize=24)
                ax.figure.savefig(savepath+'lognormalsmooth_density_tau_%s_massweight_withhopkins_logopr%0.1f_abund%s.png' % (line, np.log10(opr),str(abundance)),bbox_inches='tight')

                tau_meas = {'oneone': [0.113,0.0011], 'twotwo':[0.0162,0.00052]}
                dot,caps,bars = ax.errorbar([np.log10(15)],tau_meas[line][0],xerr=np.array([[0.33,1]]).T,yerr=tau_meas[line][1]*3,
                            label="G43.17+0.01", color=(0,0,1,0.5), alpha=0.5, marker='o',
                            linewidth=2)
                caps[0].set_marker('$($')
                caps[1].set_marker('$)$')
                caps[0].set_color((1,0,0,0.6))
                caps[1].set_color((1,0,0,0.6))
                bars[0].set_color((1,0,0,0.6))
                ax.figure.savefig(savepath+'lognormalsmooth_density_tau_%s_massweight_withhopkins_logopr%0.1f_abund%s_withG43.png' % (line, np.log10(opr),str(abundance)),bbox_inches='tight')

    if do_tables:
        # clearly not done yet
        one_sided = ['b']
        two_sided = ['sigma','Tval']
        tex = {'b':r'$b$', 'sigma': r'$\sigma_s | M$', 'Tval':r'$T$', 'sigmab':r'$\sigma_s$'}
        fmt = {'b':r'>%0.2f', 'sigma': r'%0.1f', 'Tval':r'%0.2f', 'sigmab':r'%0.1f'}
        with open('distribution_fit_table.tex','w') as f:
            for v in one_sided:
                line = [tex[v]]
                for table in [lognormal_statstable,hopkins_statstable,]:
                    if v in table['variable name']:
                        row = table[table['variable name']==v]
                        #line += ["%0.1g" % x for x in (row['q0.1'],row['q1.0'],row['q5.0'])]
                        line += ["-",fmt[v] % row['q5.0']]
                    else:
                        line += ["-"] * 2
                print >>f,"&".join(line),r"\\"
            v='sigma'
            line = [tex['sigmab']]
            for table in [lognormal_simple_statstable,hopkins_simple_statstable]:
                if v in table['variable name']:
                    row = table[table['variable name']==v]
                    line += [fmt[v] % row['q50.0'], ("$^{"+fmt[v]+"}_{"+fmt[v]+"}$") % (row['q2.5'],row['q97.5'])]
                else:
                    line += ["-"] * 2
            print >>f,"&".join(line),r"\\"
            for v in two_sided:
                line = [tex[v]]
                for table in [lognormal_statstable,hopkins_statstable,]:
                    if v in table['variable name']:
                        row = table[table['variable name']==v]
                        line += [fmt[v] % row['q50.0'], ("$^{"+fmt[v]+"}_{"+fmt[v]+"}$") % (row['q2.5'],row['q97.5'])]
                    else:
                        line += ["-"] * 2
                print >>f,"&".join(line),r"\\"


    print "Abundance %f done" % abundance
