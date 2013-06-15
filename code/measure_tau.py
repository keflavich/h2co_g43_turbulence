"""
Smooth the LVG models with distributions to get tau, then fit it with
optimization procedures.
"""

from agpy import readcol
import numpy as np
import hopkins_pdf
import turbulent_pdfs
from turbulent_pdfs import lognormal
from scipy.optimize import curve_fit

radtab = readcol('/Users/adam/work/h2co/radex/troscompt_April2013_linearXH2CO/1-1_2-2_XH2CO=1e-9_troscompt.dat',asStruct=True)

# plotting stuff
import pylab as pl
pl.rc('font',size=20)


def select_data(abundance=-9, opr=1, temperature=20, tolerance=0.1, extrapolate=False, mindens=-1, maxdens=8, dens_fit_uplim=1.5, dens_fit_lolim=7.5):
    OKtem = radtab.Temperature == temperature
    OKopr = radtab.opr == opr
    OKabund = np.abs((radtab.log10col - radtab.log10dens - np.log10(3.08e18)) - abundance) < tolerance
    OK = OKtem * OKopr * OKabund

    tau1x = radtab.TauLow[OK]
    tau2x = radtab.TauUpp[OK]
    dens = radtab.log10dens[OK]
    col = radtab.log10col[OK]

    if extrapolate: # extrapolation is not trustworthy
        def f(x,a,b):
            return a*x+b
        def f_exp(x,a,b):
            return a*np.exp(b*x)
        fitdata_lo = dens<dens_fit_uplim
        pars1x_lo = np.polyfit(dens[fitdata_lo], np.log10(tau1x[fitdata_lo]), 5)
        pars2x_lo = np.polyfit(dens[fitdata_lo], np.log10(tau2x[fitdata_lo]), 5)
        #DEBUG print "lo:", pars1x_lo,pars2x_lo
        fitdata_hi = dens>dens_fit_lolim
        pars1x_hi,cov = curve_fit(f_exp, dens[fitdata_hi], tau1x[fitdata_hi],p0=(1.4e-6,2.3),maxfev=2500)
        pars2x_hi,cov = curve_fit(f_exp, dens[fitdata_hi], tau2x[fitdata_hi],p0=(1.6e-6,2.3),maxfev=2500)
        #DEBUG print "hi:", pars1x_hi,pars2x_hi
        dens_spacing = dens[1]-dens[0]
        #new_size = dens.size + (dens[0]-mindens)/dens_spacing + (maxdens-dens[-1])/dens_spacing
        newdens = np.arange((mindens),(maxdens),dens_spacing,dtype='float')
        newcol = newdens - (dens-col).mean()
        #import pdb; pdb.set_trace()

        newtau1x = np.interp(newdens, dens, tau1x, left=-np.inf, right=np.inf)
        leftside=np.isinf(newtau1x) * (newtau1x < 0)
        #newtau1x[leftside] = 10**f(newdens[leftside],*pars1x_lo)
        newtau1x[leftside] = 10**np.polyval(pars1x_lo,newdens[leftside])
        rightside=np.isinf(newtau1x) * (newtau1x > 0)
        newtau1x[rightside] = f_exp(newdens[rightside],*pars1x_hi)

        newtau2x = np.interp(newdens, dens, tau2x, left=-np.inf, right=np.inf)
        leftside=np.isinf(newtau2x) * (newtau2x < 0)
        #newtau2x[leftside] = 10**f(newdens[leftside],*pars2x_lo)
        newtau2x[leftside] = 10**np.polyval(pars2x_lo,newdens[leftside])
        rightside=np.isinf(newtau2x) * (newtau2x > 0)
        newtau2x[rightside] = f_exp(newdens[rightside],*pars2x_hi)
        if np.any(np.isnan(newtau1x)) or np.any(np.isnan(newtau2x)):
            raise ValueError("Invalid values in tau array.")

        return newtau1x,newtau2x,newdens,newcol

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
    test_extrapolation=False
    if test_extrapolation: # this is junk now
        tau1x,tau2x,dens,col = select_data(extrapolate=True, mindens=-1, maxdens=10)
        pl.figure(55)
        pl.clf()
        pl.subplot(211)
        pl.semilogy(dens[:len(dens)/4],tau1x[:len(dens)/4])
        pl.semilogy(dens[:len(dens)/4],tau2x[:len(dens)/4])
        pl.subplot(212)
        pl.semilogy(dens[len(dens)*3/4.:],tau1x[len(dens)/4.*3:])
        pl.semilogy(dens[len(dens)*3/4.:],tau2x[len(dens)/4.*3:])
        pl.show()
        pl.figure(56)
        pl.clf()
        pl.semilogy(dens,tau1x)
        pl.semilogy(dens,tau2x)
        pl.figure(57)
        pl.clf()
        pl.semilogy(dens,tau1x/tau2x)
    if dolognormal:
        d = {}
        # fit values for GSRMC 43.30
        # meandens is "observed" but we want to trace it and let it vary...
        # with meandens observed=False,  'sigma': {'95% HPD interval': array([ 2.89972948,  3.69028675]),
        d['meandens'] = pymc.Uniform(name='meandens',lower=8,upper=150,value=15, observed=False)
        d['sigma'] = pymc.Uniform(name='sigma',lower=0,upper=25)
        d['f'] = pymc.Deterministic(name='f', eval=tauratio, parents={'meandens':d['meandens'],'sigma':d['sigma']}, doc='tauratio')
        # the observed values.  f=tau ratio = 6.65.  tau might be too high, but the "best fits" were tau=9 before, which is just not possible
        d['tauratio'] = pymc.Normal(name='tauratio',mu=d['f'],tau=4,value=6.65,observed=True)
        mc_simple = pymc.MCMC(d)
        mc_simple.sample(100000)

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
        d['m'] = pymc.Deterministic(name='m', eval=mach, parents={'b':d['b'], 'sigma':d['sigma'], 'c_s':d['c_s']}, doc="Mach number")
        # mach number could be as low as 3.7, as high as say 6.8, and the distribution between isn't super obvious...
        #d['mach'] = pymc.Normal(name='MachNumber',mu=d['m'],tau=16, value=6.6, observed=True)
        # try a much less restrictive mach number range...
        d['mach'] = pymc.Normal(name='MachNumber',mu=d['m'],tau=0.5, value=5.1, observed=True)
        mc = pymc.MCMC(d)
        mc.sample(100000)

        varslice=(10000,None,None)
        for fignum,(p1,p2) in enumerate(itertools.combinations(('f','sigma','meandens','temperature','b','m'),2)):
            pymc_plotting.hist2d(mc, p1, p2, bins=30, clear=True, fignum=fignum, varslice=varslice, colorbar=True)
            pl.title("Lognormal with Mach")
            pl.savefig(savepath+"LognormalWithMach_%s_v_%s_mcmc.png" % (p1,p2))

        for fignum2,(p1,p2) in enumerate(itertools.combinations(('f','sigma','meandens'),2)):
            pymc_plotting.hist2d(mc_simple, p1, p2, bins=30, clear=True, fignum=fignum+fignum2+1, varslice=varslice, colorbar=True)
            pl.title("Lognormal - just tauratio")
            pl.savefig(savepath+"LognormalJustTau_%s_v_%s_mcmc.png" % (p1,p2))

    if dohopkins:
        d = {}
        d['meandens'] = pymc.Uniform(name='meandens',lower=8,upper=150,value=15, observed=False)
        d['sigma'] = pymc.Uniform(name='sigma',lower=0,upper=25)
        d['f'] = pymc.Deterministic(name='f', eval=tauratio_hopkins, parents=d, doc='tauratio_hopkins')
        d['tauratio'] = pymc.Normal(name='tauratio',mu=d['f'],tau=4,value=6.65,observed=True)
        mc_hopkins_simple = pymc.MCMC(d)
        mc_hopkins_simple.sample(1e5)

        def Tval(sigma):
            return hopkins_pdf.T_of_sigma(sigma, logform=True)
        d['Tval'] = pymc.Deterministic(name='Tval', eval=Tval, parents={'sigma':d['sigma']}, doc='Intermittency parameter T')
        #def b(Tval,mach):
        #    return 20*Tval / mach
        # mach number could be as low as 3.7, as high as say 6.8, and the distribution between isn't super obvious...
        #d['mach'] = pymc.Normal(name='MachNumber',mu=d['m'],tau=16, value=6.6, observed=True)
        # try a much less restrictive mach number range...
        #d['mach'] = pymc.Normal(name='MachNumber',mu=5.1, tau=0.5, value=5.1, observed=False)
        #d['b'] = pymc.Deterministic(name='b', eval=b, parents={'Tval':d['Tval'], 'mach': d['mach']}, doc='Compressiveness Factor')

        # Alternative approach, but what is 'b'?
        d['b'] = pymc.Uniform('b',0.3,1.0)
        def mach(Tval, b):
            return 20*Tval / b
        d['m'] = pymc.Deterministic(name='m', eval=mach, parents={'Tval':d['Tval'], 'b': d['b']}, doc='Mach Number')
        d['mach'] = pymc.Normal(name='MachNumber',mu=d['m'], tau=0.5, value=5.1, observed=True)

        mc_hopkins = pymc.MCMC(d)
        mc_hopkins.sample(1e5)

        varslice=(1000,None,None)

        for fignum,(p1,p2) in enumerate(itertools.combinations(('f','sigma','meandens','Tval','b','m'),2)):
            pymc_plotting.hist2d(mc_hopkins, p1, p2, bins=30, clear=True, fignum=fignum, varslice=varslice, colorbar=True)
            pl.title("Hopkins with Mach")
            pl.savefig(savepath+"HopkinsWithMach_%s_v_%s_mcmc.png" % (p1,p2))

        for fignum2,(p1,p2) in enumerate(itertools.combinations(('f','sigma','meandens'),2)):
            pymc_plotting.hist2d(mc_hopkins_simple, p1, p2, bins=30, clear=True, fignum=fignum+fignum2+1, varslice=varslice, colorbar=True)
            pl.title("Hopkins - just tauratio")
            pl.savefig(savepath+"HopkinsJustTau_%s_v_%s_mcmc.png" % (p1,p2))

        print 'mc_lognormal_simple sigma: ',mc_simple.stats()['sigma']['quantiles']
        print 'mc_lognormal        sigma: ',mc.stats()['sigma']['quantiles']
        print 'mc_hopkins_simple sigma: ',mc_hopkins_simple.stats()['sigma']['quantiles']
        print 'mc_hopkins        sigma: ',mc_hopkins.stats()['sigma']['quantiles']
        print 'mc_hopkins        Tval: ',mc_hopkins.stats()['Tval']['quantiles']
        print 'mc_hopkins        b: ',mc_hopkins.stats(quantiles=(0.1,1,2.5,5,50))['b']['quantiles']
        print 'mc_hopkins        m: ',mc_hopkins.stats()['m']['quantiles']

    # with sigma, meandens, f:
    # {'f': {'95% HPD interval': array([ 5.58464897,  7.55134594]),
    #   'mc error': 0.0049541949426704524,
    #   'mean': 6.5820004847214033,
    #   'n': 100000,
    #   'quantiles': {2.5: 5.5914346590685096,
    #    25: 6.2486693628418006,
    #    50: 6.5856617740996537,
    #    75: 6.9156289027613322,
    #    97.5: 7.5600756529076545},
    #   'standard deviation': 0.50079175941484455},
    #  'meandens': {'95% HPD interval': array([   8.72927479,  143.38600329]),
    #   'mc error': 0.71342470721976947,
    #   'mean': 77.770131313399418,
    #   'n': 100000,
    #   'quantiles': {2.5: 11.355003009982402,
    #    25: 41.808273968827784,
    #    50: 77.125330673883411,
    #    75: 113.32613066974223,
    #    97.5: 146.33760001919006},
    #   'standard deviation': 41.122387048289958},
    #  'sigma': {'95% HPD interval': array([ 2.31057206,  3.41069483]),
    #   'mc error': 0.0056520577266689926,
    #   'mean': 2.8252845070200503,
    #   'n': 100000,
    #   'quantiles': {2.5: 2.3472596807723374,
    #    25: 2.608976879876399,
    #    50: 2.7897972317263817,
    #    75: 3.0117403887424472,
    #    97.5: 3.4699062572773967},
    #   'standard deviation': 0.29853881932899151}}
    #

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

        #if 'mc_hopkins_simple' in locals():
        #    mchss = mc_hopkins_simple.stats()
        #    med = mchss['sigma']['quantiles'][50]
        #    lo,hi = mchss['sigma']['95% HPD interval']
        #    ax.plot(logmeandens,tauratio_hopkins(meandens,sigma=med),color='orange', linewidth=0.5, alpha=0.5,zorder=10)
        #    ax.fill_between(logmeandens,tauratio_hopkins(meandens,sigma=lo),tauratio_hopkins(meandens,sigma=hi), color='orange', alpha=0.1, zorder=0)
        #    #mch_lines = []
        #    #for sigma in mc_hopkins_simple.trace('sigma')[-1:-501:-20]:
        #    #    mch_lines += ax.plot(logmeandens,tauratio_hopkins(meandens,sigma=sigma),color='red', linewidth=0.5, alpha=0.1)


        ax.legend(loc='best',prop={'size':18})
        ax.axis([-1,7,0,15])
        ax.set_xlabel('$\\log_{10}$($n($H$_2)$ [cm$^{-3}$])',fontsize=24)
        ax.set_ylabel('$\\tau_{1-1}/\\tau_{2-2}$',fontsize=24)
        ax.figure.savefig(savepath+'lognormalsmooth_density_ratio_massweight_withhopkins_logopr%0.1f_abund%i.png' % (np.log10(opr),abundance),bbox_inches='tight')

        ax.errorbar([np.log10(15)],[6.65],xerr=np.array([[0.33,1]]).T,yerr=np.array([[0.5,0.5]]).T,
                    label="G43.16-0.03", color=(0,0,1,0.5), alpha=0.5, marker='o',
                    linewidth=2)
        ax.figure.savefig(savepath+'lognormalsmooth_density_ratio_massweight_withhopkins_logopr%0.1f_abund%i_withG43.png' % (np.log10(opr),abundance),bbox_inches='tight')
